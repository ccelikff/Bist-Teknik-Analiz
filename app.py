"""
BİST TEKNİK ANALİZ — Streamlit Cloud Versiyonu
Sinyal Sistemi + Destek/Direnç
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BİST Teknik Analiz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #0D1117; }
  [data-testid="stSidebar"]          { background-color: #161B22; }
  [data-testid="stHeader"]           { background-color: #0D1117; }
  h1,h2,h3,label,p,span             { color: #C9D1D9 !important; }
  .stButton > button {
    background: #2E75B6; color: white; border: none;
    border-radius: 6px; font-weight: 700; width: 100%;
    padding: 10px; font-size: 15px;
  }
  .stButton > button:hover { background: #1F5FA6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# İNDİKATÖRLER
# ─────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_macd(c, fast=12, slow=26, sig=9):
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, sig)
    return m.round(4), s.round(4), (m - s).round(4)

def calc_rsi(c, n=10):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).round(2)

def calc_cci(h, l, c, n=14):
    tp = (h + l + c) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return ((tp - ma) / (0.015 * md)).round(2)

def calc_mom(c, n=10):
    return (c - c.shift(n)).round(4)

def calc_stoch(h, l, c, k=14, d=3, smooth=3):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    K = raw.rolling(smooth).mean().round(2)
    D = K.rolling(d).mean().round(2)
    return K, D

def calc_atr(h, l, c, n=14):
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().round(4)

# ─────────────────────────────────────────────
# SİNYAL SİSTEMİ
# ─────────────────────────────────────────────
def calc_signals(close, high, low, rsi_period, mom_period):
    macd_l, macd_s, _ = calc_macd(close)
    rsi               = calc_rsi(close, rsi_period)
    stk_k, stk_d      = calc_stoch(high, low, close)
    mom               = calc_mom(close, mom_period)
    e20               = ema(close, 20)
    e50               = ema(close, 50)

    score = pd.Series(0, index=close.index, dtype=float)

    for i in range(1, len(close)):
        s = 0
        # MACD kesişimi
        if macd_l.iloc[i-1] < macd_s.iloc[i-1] and macd_l.iloc[i] > macd_s.iloc[i]:
            s += 1
        elif macd_l.iloc[i-1] > macd_s.iloc[i-1] and macd_l.iloc[i] < macd_s.iloc[i]:
            s -= 1
        # RSİ aşırı bölge çıkışı
        if rsi.iloc[i-1] < 30 and rsi.iloc[i] >= 30:
            s += 1
        elif rsi.iloc[i-1] > 70 and rsi.iloc[i] <= 70:
            s -= 1
        # Stokastik kesişimi
        if stk_k.iloc[i-1] < stk_d.iloc[i-1] and stk_k.iloc[i] > stk_d.iloc[i] and stk_k.iloc[i] < 40:
            s += 1
        elif stk_k.iloc[i-1] > stk_d.iloc[i-1] and stk_k.iloc[i] < stk_d.iloc[i] and stk_k.iloc[i] > 60:
            s -= 1
        # EMA kesişimi
        if e20.iloc[i-1] < e50.iloc[i-1] and e20.iloc[i] > e50.iloc[i]:
            s += 1
        elif e20.iloc[i-1] > e50.iloc[i-1] and e20.iloc[i] < e50.iloc[i]:
            s -= 1
        # Momentum sıfır geçişi
        if mom.iloc[i-1] < 0 and mom.iloc[i] >= 0:
            s += 1
        elif mom.iloc[i-1] > 0 and mom.iloc[i] <= 0:
            s -= 1
        score.iloc[i] = s

    return score

def signal_summary(close, high, low, rsi_period, mom_period):
    macd_l, macd_s, _ = calc_macd(close)
    rsi               = calc_rsi(close, rsi_period)
    stk_k, stk_d      = calc_stoch(high, low, close)
    mom               = calc_mom(close, mom_period)
    e20               = ema(close, 20)
    e50               = ema(close, 50)

    rows = []
    # MACD
    if macd_l.iloc[-1] > macd_s.iloc[-1]:
        rows.append(("MACD-AS", "📗 YUKARI",  "MACD sinyal üzerinde", "al"))
    else:
        rows.append(("MACD-AS", "📕 AŞAĞI",   "MACD sinyal altında",  "sat"))
    # RSİ
    rv = float(rsi.iloc[-1])
    if rv < 30:
        rows.append(("RSİ", f"📗 {rv:.1f} AŞIRI SATIM", "Alım bölgesi", "al"))
    elif rv > 70:
        rows.append(("RSİ", f"📕 {rv:.1f} AŞIRI ALIM",  "Satım bölgesi", "sat"))
    else:
        rows.append(("RSİ", f"⬜ {rv:.1f} NÖTR", "30-70 arası", "notr"))
    # Stokastik
    kv = float(stk_k.iloc[-1])
    if stk_k.iloc[-1] > stk_d.iloc[-1] and kv < 80:
        rows.append(("STOKASTİK", f"📗 {kv:.1f} YUKARI", "%K sinyal üzerinde", "al"))
    elif stk_k.iloc[-1] < stk_d.iloc[-1] and kv > 20:
        rows.append(("STOKASTİK", f"📕 {kv:.1f} AŞAĞI",  "%K sinyal altında",  "sat"))
    else:
        rows.append(("STOKASTİK", f"⬜ {kv:.1f} NÖTR", "Aşırı bölge", "notr"))
    # EMA
    if e20.iloc[-1] > e50.iloc[-1]:
        rows.append(("EMA 20/50", "📗 YUKARI TREND", "EMA20 > EMA50", "al"))
    else:
        rows.append(("EMA 20/50", "📕 AŞAĞI TREND",  "EMA20 < EMA50", "sat"))
    # Momentum
    mv = float(mom.iloc[-1])
    if mv > 0:
        rows.append(("MOMENTUM", f"📗 +{mv:.2f}", "Pozitif ivme", "al"))
    else:
        rows.append(("MOMENTUM", f"📕 {mv:.2f}",  "Negatif ivme", "sat"))

    return rows

# ─────────────────────────────────────────────
# DESTEK / DİRENÇ
# ─────────────────────────────────────────────
def find_support_resistance(high, low, close, window=10, max_levels=5, tolerance=0.015):
    supports, resistances = [], []
    n = len(close)
    for i in range(window, n - window):
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            supports.append(float(low.iloc[i]))
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            resistances.append(float(high.iloc[i]))

    def cluster(levels, tol):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lv in levels[1:]:
            if (lv - clusters[-1][-1]) / clusters[-1][-1] < tol:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        return [np.mean(c) for c in clusters]

    last_price = float(close.iloc[-1])
    sup = sorted(cluster(supports,    tolerance), key=lambda x: abs(x - last_price))[:max_levels]
    res = sorted(cluster(resistances, tolerance), key=lambda x: abs(x - last_price))[:max_levels]
    return sorted(sup), sorted(res)

# ─────────────────────────────────────────────
# VERİ ÇEK
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch(ticker, days, interval):
    symbol = ticker if ticker.endswith(".IS") else ticker + ".IS"
    df = yf.download(symbol, period=f"{days}d", interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df.astype(float)

# ─────────────────────────────────────────────
# GRAFİK
# ─────────────────────────────────────────────
def build_figure(df, ticker, label, rsi_period, mom_period, show_signals, show_sr):
    close, high, low, op = df["Close"], df["High"], df["Low"], df["Open"]
    vol = df["Volume"] if "Volume" in df.columns else None

    macd_l, macd_s, macd_h = calc_macd(close)
    rsi    = calc_rsi(close, rsi_period)
    rsi_ma = rsi.rolling(8).mean().round(2)
    cci    = calc_cci(high, low, close)
    cci_ma = cci.rolling(8).mean().round(2)
    mom    = calc_mom(close, mom_period)
    mom_ma = mom.rolling(8).mean().round(4)
    stk_k, stk_d = calc_stoch(high, low, close)
    atr    = calc_atr(high, low, close)
    signals = calc_signals(close, high, low, rsi_period, mom_period)
    sup_levels, res_levels = find_support_resistance(high, low, close)

    fig = make_subplots(
        rows=8, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.25, 0.09, 0.12, 0.11, 0.11, 0.11, 0.11, 0.10],
        subplot_titles=[
            "FİYAT",
            "ATR (14)",
            "MACD-AS (12,26,9)",
            f"RSİ ({rsi_period})",
            "CCI (14)",
            f"MOMENTUM ({mom_period})",
            "STOKASTİK (14,3,3)",
            "HACİM",
        ],
    )

    # ── 1. FİYAT ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=op, high=high, low=low, close=close, name="Fiyat",
        increasing=dict(line=dict(color="#26A69A"), fillcolor="#26A69A"),
        decreasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema(close,20), name="EMA 20",
        line=dict(color="#FFA726", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema(close,50), name="EMA 50",
        line=dict(color="#7E57C2", width=1.2)), row=1, col=1)

    # Destek/Direnç
    if show_sr:
        for lv in sup_levels:
            fig.add_hline(y=lv, line=dict(color="#26A69A", width=1, dash="dot"),
                annotation_text=f"  D {lv:.2f}",
                annotation_font=dict(color="#26A69A", size=9),
                annotation_position="right", row=1, col=1)
        for lv in res_levels:
            fig.add_hline(y=lv, line=dict(color="#EF5350", width=1, dash="dot"),
                annotation_text=f"  R {lv:.2f}",
                annotation_font=dict(color="#EF5350", size=9),
                annotation_position="right", row=1, col=1)

    # Sinyaller
    if show_signals:
        buy_idx  = signals >= 2
        sell_idx = signals <= -2
        wbuy_idx = signals == 1
        wsell_idx= signals == -1

        if buy_idx.any():
            fig.add_trace(go.Scatter(x=df.index[buy_idx], y=low[buy_idx]*0.985,
                mode="markers", name="💪 Güçlü Alım",
                marker=dict(symbol="triangle-up", size=14, color="#00E676",
                            line=dict(color="#00C853", width=1))), row=1, col=1)
        if sell_idx.any():
            fig.add_trace(go.Scatter(x=df.index[sell_idx], y=high[sell_idx]*1.015,
                mode="markers", name="💪 Güçlü Satım",
                marker=dict(symbol="triangle-down", size=14, color="#FF1744",
                            line=dict(color="#D50000", width=1))), row=1, col=1)
        if wbuy_idx.any():
            fig.add_trace(go.Scatter(x=df.index[wbuy_idx], y=low[wbuy_idx]*0.990,
                mode="markers", name="Zayıf Alım",
                marker=dict(symbol="triangle-up", size=9,
                            color="rgba(38,166,154,0.5)")), row=1, col=1)
        if wsell_idx.any():
            fig.add_trace(go.Scatter(x=df.index[wsell_idx], y=high[wsell_idx]*1.010,
                mode="markers", name="Zayıf Satım",
                marker=dict(symbol="triangle-down", size=9,
                            color="rgba(239,83,80,0.5)")), row=1, col=1)

    # ── 2. ATR ──
    fig.add_trace(go.Scatter(x=df.index, y=atr, name="ATR(14)",
        line=dict(color="#00BCD4", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.1)"), row=2, col=1)

    # ── 3. MACD-AS ──
    ch = ["#26A69A" if v >= 0 else "#EF5350" for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=macd_h, name="AS",
        marker_color=ch, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_l, name="MACD-AS",
        line=dict(color="#2196F3", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_s, name="Sinyal",
        line=dict(color="#FF5722", width=1.5, dash="dot")), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dash"), row=3, col=1)

    if show_signals:
        for i in range(1, len(macd_l)):
            if macd_l.iloc[i-1] < macd_s.iloc[i-1] and macd_l.iloc[i] > macd_s.iloc[i]:
                fig.add_trace(go.Scatter(x=[df.index[i]], y=[macd_l.iloc[i]],
                    mode="markers", marker=dict(symbol="triangle-up", size=10, color="#00E676"),
                    showlegend=False), row=3, col=1)
            elif macd_l.iloc[i-1] > macd_s.iloc[i-1] and macd_l.iloc[i] < macd_s.iloc[i]:
                fig.add_trace(go.Scatter(x=[df.index[i]], y=[macd_l.iloc[i]],
                    mode="markers", marker=dict(symbol="triangle-down", size=10, color="#FF1744"),
                    showlegend=False), row=3, col=1)

    # ── 4. RSİ ──
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name=f"RSİ({rsi_period})",
        line=dict(color="#FF9800", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.07)"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_ma, name="RSİ MA(8)",
        line=dict(color="#FFFFFF", width=1.2, dash="dot")), row=4, col=1)
    fig.add_hline(y=70, line=dict(color="#EF5350", width=1, dash="dash"), row=4, col=1)
    fig.add_hline(y=30, line=dict(color="#26A69A", width=1, dash="dash"), row=4, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0, row=4, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=4, col=1)

    # ── 5. CCI ──
    fig.add_trace(go.Scatter(x=df.index, y=cci, name="CCI(14)",
        line=dict(color="#EF5350", width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=cci_ma, name="CCI MA(8)",
        line=dict(color="#FFEB3B", width=1.2, dash="dot")), row=5, col=1)
    fig.add_hline(y=100,  line=dict(color="#EF5350", width=1, dash="dash"), row=5, col=1)
    fig.add_hline(y=-100, line=dict(color="#26A69A", width=1, dash="dash"), row=5, col=1)
    fig.add_hline(y=0,    line=dict(color="gray", width=0.5, dash="dot"), row=5, col=1)

    # ── 6. MOMENTUM ──
    mc = ["#26A69A" if v >= 0 else "#EF5350" for v in mom.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=mom, name=f"Mom({mom_period})",
        marker_color=mc, opacity=0.6), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=mom_ma, name="Mom MA(8)",
        line=dict(color="#CE93D8", width=1.5)), row=6, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dash"), row=6, col=1)

    # ── 7. STOKASTİK ──
    fig.add_trace(go.Scatter(x=df.index, y=stk_k, name="%K",
        line=dict(color="#1565C0", width=1.5)), row=7, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stk_d, name="%D",
        line=dict(color="#E53935", width=1.5, dash="dot")), row=7, col=1)
    fig.add_hline(y=80, line=dict(color="#EF5350", width=1, dash="dash"), row=7, col=1)
    fig.add_hline(y=20, line=dict(color="#26A69A", width=1, dash="dash"), row=7, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0, row=7, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=7, col=1)

    # ── 8. HACİM ──
    if vol is not None:
        vc = ["#26A69A" if c >= o else "#EF5350"
              for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=vol, name="Hacim",
            marker_color=vc, opacity=0.6), row=8, col=1)

    fig.update_layout(
        title=dict(text=f"<b>IST:{ticker}</b> — {label} Teknik Analiz",
            font=dict(size=16, color="#E0E0E0"), x=0.5, y=0.99),
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font=dict(family="Arial", color="#C9D1D9", size=10),
        height=1250,
        margin=dict(l=60, r=180, t=40, b=30),
        legend=dict(
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            bgcolor="#161B22",
            bordercolor="#30363D",
            borderwidth=1,
            font=dict(size=11, color="#C9D1D9"),
            tracegroupgap=4,
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="#8B949E",
            activecolor="#58A6FF",
            remove=["lasso2d", "select2d", "autoScale2d",
                    "hoverClosestCartesian", "hoverCompareCartesian",
                    "toggleSpikelines"],
        ),
    )

    ax = dict(gridcolor="#21262D", gridwidth=0.5,
              zerolinecolor="#30363D", zerolinewidth=1, tickfont=dict(size=8))
    for i in range(1, 9):
        fig.update_yaxes(ax, row=i, col=1)
        fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5,
                         showticklabels=(i == 8), row=i, col=1)

    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=7, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = "#58A6FF"
        ann.font.size  = 11
        ann.font.family = "Arial"
        ann.x = 0.01
        ann.xanchor = "left"
        ann.xref = "paper"

    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 BİST Teknik Analiz")
    st.markdown("---")

    ticker = st.text_input("Hisse Kodu", value="THYAO",
        placeholder="THYAO, SISE, GARAN...").upper().strip()

    st.markdown("**Veri Aralığı**")
    gunluk_gun   = st.slider("Günlük (gün)",   90,  730,  365)
    haftalik_gun = st.slider("Haftalık (gün)", 180, 1460, 730)

    st.markdown("**İndikatör Ayarları**")
    rsi_period = st.number_input("RSİ Periyot",      2, 50, 10)
    mom_period = st.number_input("Momentum Periyot", 2, 50, 10)

    st.markdown("**Göster / Gizle**")
    show_signals = st.toggle("🔔 Alım/Satım Sinyalleri", value=True)
    show_sr      = st.toggle("📐 Destek / Direnç",       value=True)

    st.markdown("---")
    analiz_btn = st.button("🚀 ANALİZ ET")
    st.markdown("---")
    st.markdown("""<div style='font-size:11px;color:#484F58;text-align:center'>
    Veriler yfinance aracılığıyla<br>çekilmektedir.<br>⚠️ Yatırım tavsiyesi değildir.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANA ALAN
# ─────────────────────────────────────────────
if analiz_btn:
    st.session_state.update({
        "last_ticker": ticker, "last_gd": gunluk_gun, "last_hd": haftalik_gun,
        "last_rsi": rsi_period, "last_mom": mom_period,
        "last_signals": show_signals, "last_sr": show_sr,
    })

t  = st.session_state.get("last_ticker", "")
gd = st.session_state.get("last_gd",  gunluk_gun)
hd = st.session_state.get("last_hd",  haftalik_gun)
rp = st.session_state.get("last_rsi", rsi_period)
mp = st.session_state.get("last_mom", mom_period)
ss = st.session_state.get("last_signals", show_signals)
sr = st.session_state.get("last_sr",      show_sr)

if not t:
    st.markdown("""<div style='text-align:center;padding:80px;color:#484F58;'>
      <div style='font-size:48px;'>📈</div>
      <div style='font-size:20px;margin-top:16px;'>
        Sol panelden hisse kodu girin ve <b>ANALİZ ET</b> butonuna basın
      </div></div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner(f"📥 {t} verisi çekiliyor..."):
    df_d = fetch(t, gd,  "1d")
    df_w = fetch(t, hd, "1wk")

if df_d is None or df_w is None:
    st.error(f"❌ **{t}** için veri bulunamadı.")
    st.stop()

# ── Fiyat başlık kartı ──
last    = df_d.iloc[-1]
prev    = df_d.iloc[-2]
chg     = float(last["Close"]) - float(prev["Close"])
chg_pct = chg / float(prev["Close"]) * 100
sign    = "▲" if chg >= 0 else "▼"
clr     = "#26A69A" if chg >= 0 else "#EF5350"

st.markdown(f"""
<div style='background:#161B22;border:1px solid #30363D;border-radius:10px;
            padding:16px 24px;margin-bottom:12px;
            display:flex;align-items:center;gap:32px;flex-wrap:wrap;'>
  <div>
    <div style='font-size:28px;font-weight:900;color:#58A6FF;'>IST:{t}</div>
    <div style='font-size:11px;color:#8B949E;'>Borsa İstanbul</div>
  </div>
  <div>
    <div style='font-size:26px;font-weight:800;color:#E0E0E0;'>₺{float(last["Close"]):.2f}</div>
    <div style='font-size:16px;font-weight:700;color:{clr};'>{sign} ₺{abs(chg):.2f} ({chg_pct:+.2f}%)</div>
  </div>
  <div style='font-size:12px;color:#8B949E;line-height:1.9'>
    Açılış: <b style='color:#E0E0E0'>₺{float(last["Open"]):.2f}</b> &nbsp;|&nbsp;
    Yüksek: <b style='color:#26A69A'>₺{float(last["High"]):.2f}</b> &nbsp;|&nbsp;
    Düşük: <b style='color:#EF5350'>₺{float(last["Low"]):.2f}</b> &nbsp;|&nbsp;
    Tarih: <b style='color:#E0E0E0'>{df_d.index[-1].strftime('%d.%m.%Y')}</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sinyal Özet Paneli ──
if ss:
    summary   = signal_summary(df_d["Close"], df_d["High"], df_d["Low"], rp, mp)
    al_count  = sum(1 for r in summary if r[3] == "al")
    sat_count = sum(1 for r in summary if r[3] == "sat")
    total     = len(summary)

    if al_count > sat_count:
        oclr = "#26A69A"; otxt = f"📗 GENEL: ALIM EĞİLİMİ  ({al_count}/{total} indikatör)"
    elif sat_count > al_count:
        oclr = "#EF5350"; otxt = f"📕 GENEL: SATIM EĞİLİMİ  ({sat_count}/{total} indikatör)"
    else:
        oclr = "#8B949E"; otxt = f"⬜ GENEL: NÖTR  ({al_count} alım / {sat_count} satım)"

    with st.expander("🔔 GÜNLÜK SİNYAL ÖZETİ", expanded=True):
        st.markdown(f"""<div style='background:rgba(0,0,0,0.3);border:2px solid {oclr};
            border-radius:8px;padding:10px 16px;margin-bottom:10px;
            font-size:15px;font-weight:700;color:{oclr};'>{otxt}</div>""",
            unsafe_allow_html=True)

        cols = st.columns(len(summary))
        for col, (ind, durum, aciklama, tip) in zip(cols, summary):
            bg  = "rgba(38,166,154,0.12)" if tip=="al" else \
                  "rgba(239,83,80,0.12)"  if tip=="sat" else "rgba(50,50,50,0.2)"
            brd = "#26A69A" if tip=="al" else "#EF5350" if tip=="sat" else "#30363D"
            col.markdown(f"""<div style='background:{bg};border:1px solid {brd};
                border-radius:8px;padding:10px;text-align:center;'>
                <div style='font-size:11px;color:#8B949E;margin-bottom:4px;'>{ind}</div>
                <div style='font-size:12px;font-weight:700;'>{durum}</div>
                <div style='font-size:10px;color:#8B949E;margin-top:4px;'>{aciklama}</div>
                </div>""", unsafe_allow_html=True)

# ── Destek/Direnç Paneli ──
if sr:
    sup_d, res_d = find_support_resistance(df_d["High"], df_d["Low"], df_d["Close"])
    last_price   = float(df_d["Close"].iloc[-1])

    with st.expander("📐 DESTEK / DİRENÇ SEVİYELERİ", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 Destek Seviyeleri**")
            for lv in sorted(sup_d, reverse=True):
                uzaklik = (last_price - lv) / last_price * 100
                st.markdown(f"""<div style='background:rgba(38,166,154,0.1);
                    border-left:3px solid #26A69A;padding:6px 12px;margin:4px 0;
                    border-radius:4px;display:flex;justify-content:space-between;'>
                    <span style='font-weight:700;color:#26A69A;'>₺{lv:.2f}</span>
                    <span style='color:#8B949E;font-size:11px;'>%{uzaklik:.1f} uzakta</span>
                    </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("**🔴 Direnç Seviyeleri**")
            for lv in sorted(res_d):
                uzaklik = (lv - last_price) / last_price * 100
                st.markdown(f"""<div style='background:rgba(239,83,80,0.1);
                    border-left:3px solid #EF5350;padding:6px 12px;margin:4px 0;
                    border-radius:4px;display:flex;justify-content:space-between;'>
                    <span style='font-weight:700;color:#EF5350;'>₺{lv:.2f}</span>
                    <span style='color:#8B949E;font-size:11px;'>%{uzaklik:.1f} uzakta</span>
                    </div>""", unsafe_allow_html=True)

# ── Grafikler ──
tab1, tab2 = st.tabs(["📈 Günlük", "📊 Haftalık"])
with tab1:
    st.plotly_chart(build_figure(df_d, t, "GÜNLÜK",   rp, mp, ss, sr), use_container_width=True)
with tab2:
    st.plotly_chart(build_figure(df_w, t, "HAFTALIK",  rp, mp, ss, sr), use_container_width=True)
