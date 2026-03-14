"""
BİST TEKNİK ANALİZ — Streamlit Cloud Versiyonu
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────
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
  h1, h2, h3, label, p, span        { color: #C9D1D9 !important; }
  .stButton > button {
    background: #2E75B6; color: white; border: none;
    border-radius: 6px; font-weight: 700; width: 100%;
    padding: 10px; font-size: 15px;
  }
  .stButton > button:hover { background: #1F5FA6; }
  .metric-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 8px; padding: 12px 16px; text-align: center;
  }
  .metric-val  { font-size: 22px; font-weight: 800; }
  .metric-lbl  { font-size: 11px; color: #8B949E; margin-top: 2px; }
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
# VERİ ÇEK
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)   # 5 dakika cache
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
def build_figure(df, ticker, label, rsi_period, mom_period):
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

    row_heights = [0.25, 0.09, 0.12, 0.11, 0.11, 0.11, 0.11, 0.10]

    fig = make_subplots(
        rows=8, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.022,
        row_heights=row_heights,
        subplot_titles=[
            "📈 FİYAT (Mum) + EMA 20/50",
            "📏 ATR (14) — Volatilite",
            "📊 MACD-AS (12, 26, 9)",
            f"📉 RSİ ({rsi_period}) + MA 8",
            "📐 CCI (14) + MA 8",
            f"⚡ MOMENTUM ({mom_period}) + MA 8",
            "🔄 STOKASTİK (14, 3, 3)",
            "📦 HACİM",
        ],
    )

    # 1. FİYAT
    fig.add_trace(go.Candlestick(
        x=df.index, open=op, high=high, low=low, close=close, name="Fiyat",
        increasing=dict(line=dict(color="#26A69A"), fillcolor="#26A69A"),
        decreasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema(close, 20), name="EMA 20",
        line=dict(color="#FFA726", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema(close, 50), name="EMA 50",
        line=dict(color="#7E57C2", width=1.2)), row=1, col=1)

    # 2. ATR
    fig.add_trace(go.Scatter(x=df.index, y=atr, name="ATR(14)",
        line=dict(color="#00BCD4", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.1)"), row=2, col=1)

    # 3. MACD-AS
    colors_hist = ["#26A69A" if v >= 0 else "#EF5350" for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=macd_h, name="AS",
        marker_color=colors_hist, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_l, name="MACD-AS",
        line=dict(color="#2196F3", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_s, name="Sinyal",
        line=dict(color="#FF5722", width=1.5, dash="dot")), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dash"), row=3, col=1)

    # 4. RSİ
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name=f"RSİ({rsi_period})",
        line=dict(color="#FF9800", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.07)"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_ma, name="RSİ MA(8)",
        line=dict(color="#FFFFFF", width=1.2, dash="dot")), row=4, col=1)
    fig.add_hline(y=70, line=dict(color="#EF5350", width=1, dash="dash"), row=4, col=1)
    fig.add_hline(y=30, line=dict(color="#26A69A", width=1, dash="dash"), row=4, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0, row=4, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=4, col=1)

    # 5. CCI
    fig.add_trace(go.Scatter(x=df.index, y=cci, name="CCI(14)",
        line=dict(color="#EF5350", width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=cci_ma, name="CCI MA(8)",
        line=dict(color="#FFEB3B", width=1.2, dash="dot")), row=5, col=1)
    fig.add_hline(y=100,  line=dict(color="#EF5350", width=1, dash="dash"), row=5, col=1)
    fig.add_hline(y=-100, line=dict(color="#26A69A", width=1, dash="dash"), row=5, col=1)
    fig.add_hline(y=0,    line=dict(color="gray", width=0.5, dash="dot"), row=5, col=1)

    # 6. MOMENTUM
    mom_colors = ["#26A69A" if v >= 0 else "#EF5350" for v in mom.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=mom, name=f"Mom({mom_period})",
        marker_color=mom_colors, opacity=0.6), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=mom_ma, name="Mom MA(8)",
        line=dict(color="#CE93D8", width=1.5)), row=6, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dash"), row=6, col=1)

    # 7. STOKASTİK
    fig.add_trace(go.Scatter(x=df.index, y=stk_k, name="%K",
        line=dict(color="#1565C0", width=1.5)), row=7, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stk_d, name="%D",
        line=dict(color="#E53935", width=1.5, dash="dot")), row=7, col=1)
    fig.add_hline(y=80, line=dict(color="#EF5350", width=1, dash="dash"), row=7, col=1)
    fig.add_hline(y=20, line=dict(color="#26A69A", width=1, dash="dash"), row=7, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0, row=7, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=7, col=1)

    # 8. HACİM
    if vol is not None:
        vc = ["#26A69A" if c >= o else "#EF5350"
              for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=vol, name="Hacim",
            marker_color=vc, opacity=0.6), row=8, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>IST:{ticker}</b> — {label} Teknik Analiz",
            font=dict(size=18, color="#E0E0E0"), x=0.5,
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#161B22",
        font=dict(family="Arial", color="#C9D1D9", size=10),
        height=1200,
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0.3)", font=dict(size=9),
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    ax = dict(gridcolor="#21262D", gridwidth=0.5,
              zerolinecolor="#30363D", zerolinewidth=1,
              tickfont=dict(size=8))
    for i in range(1, 9):
        fig.update_yaxes(ax, row=i, col=1)
        fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5,
                         showticklabels=(i == 8), row=i, col=1)

    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=7, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = "#8B949E"
        ann.font.size  = 10

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
    gunluk_gun   = st.slider("Günlük (gün)",   90, 730, 365)
    haftalik_gun = st.slider("Haftalık (gün)", 180, 1460, 730)

    st.markdown("**İndikatör Ayarları**")
    rsi_period = st.number_input("RSİ Periyot", 2, 50, 10)
    mom_period = st.number_input("Momentum Periyot", 2, 50, 10)

    st.markdown("---")
    analiz_btn = st.button("🚀 ANALİZ ET")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#484F58; text-align:center'>
    Veriler yfinance aracılığıyla<br>çekilmektedir.<br>
    Yatırım tavsiyesi değildir.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANA ALAN
# ─────────────────────────────────────────────
if analiz_btn or "last_ticker" not in st.session_state:
    if analiz_btn:
        st.session_state["last_ticker"]      = ticker
        st.session_state["last_gunluk_gun"]  = gunluk_gun
        st.session_state["last_haftalik_gun"]= haftalik_gun
        st.session_state["last_rsi"]         = rsi_period
        st.session_state["last_mom"]         = mom_period

t   = st.session_state.get("last_ticker", ticker)
gd  = st.session_state.get("last_gunluk_gun", gunluk_gun)
hd  = st.session_state.get("last_haftalik_gun", haftalik_gun)
rp  = st.session_state.get("last_rsi", rsi_period)
mp  = st.session_state.get("last_mom", mom_period)

if t:
    with st.spinner(f"📥 {t} verisi çekiliyor..."):
        df_d = fetch(t, gd,  "1d")
        df_w = fetch(t, hd, "1wk")

    if df_d is None or df_w is None:
        st.error(f"❌ **{t}** için veri bulunamadı. Ticker kodunu kontrol edin.")
    else:
        # Özet metrikler
        last = df_d.iloc[-1]
        prev = df_d.iloc[-2]
        chg  = float(last["Close"]) - float(prev["Close"])
        chg_pct = chg / float(prev["Close"]) * 100
        sign    = "▲" if chg >= 0 else "▼"
        clr     = "#26A69A" if chg >= 0 else "#EF5350"

        st.markdown(f"""
        <div style='background:#161B22; border:1px solid #30363D; border-radius:10px;
                    padding:16px 24px; margin-bottom:16px;
                    display:flex; align-items:center; gap:32px;'>
          <div>
            <div style='font-size:28px; font-weight:900; color:#58A6FF;'>IST:{t}</div>
            <div style='font-size:11px; color:#8B949E;'>Borsa İstanbul</div>
          </div>
          <div>
            <div style='font-size:26px; font-weight:800; color:#E0E0E0;'>₺{float(last["Close"]):.2f}</div>
            <div style='font-size:16px; font-weight:700; color:{clr};'>
              {sign} ₺{abs(chg):.2f} ({chg_pct:+.2f}%)
            </div>
          </div>
          <div style='font-size:12px; color:#8B949E; line-height:1.8'>
            Açılış: <b style='color:#E0E0E0'>₺{float(last["Open"]):.2f}</b> &nbsp;|&nbsp;
            Yüksek: <b style='color:#26A69A'>₺{float(last["High"]):.2f}</b> &nbsp;|&nbsp;
            Düşük: <b style='color:#EF5350'>₺{float(last["Low"]):.2f}</b> &nbsp;|&nbsp;
            Tarih: <b style='color:#E0E0E0'>{df_d.index[-1].strftime('%d.%m.%Y')}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Sekmeler
        tab1, tab2 = st.tabs(["📈 Günlük", "📊 Haftalık"])

        with tab1:
            fig_d = build_figure(df_d, t, "GÜNLÜK", rp, mp)
            st.plotly_chart(fig_d, use_container_width=True)

        with tab2:
            fig_w = build_figure(df_w, t, "HAFTALIK", rp, mp)
            st.plotly_chart(fig_w, use_container_width=True)
else:
    st.markdown("""
    <div style='text-align:center; padding:80px; color:#484F58;'>
      <div style='font-size:48px;'>📈</div>
      <div style='font-size:20px; margin-top:16px;'>
        Sol panelden hisse kodu girin ve <b>ANALİZ ET</b> butonuna basın
      </div>
    </div>
    """, unsafe_allow_html=True)
