"""
BİST & EMTİA TEKNİK ANALİZ
- Çoklu Zaman Dilimi Karşılaştırma
- Divergence Tespiti
- Emtia (Altın, Gümüş, Platin, Paladyum)
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
# EMTİA TANIMLARI
# ─────────────────────────────────────────────
EMTIALAR = {
    "🥇 Altın":     {"ticker": "GC=F",  "sembol": "ALTIN",   "para": "$", "renk": "#FFD700"},
    "🥈 Gümüş":    {"ticker": "SI=F",  "sembol": "GUMUS",   "para": "$", "renk": "#C0C0C0"},
    "💎 Platin":    {"ticker": "PL=F",  "sembol": "PLATIN",  "para": "$", "renk": "#E5E4E2"},
    "🔮 Paladyum":  {"ticker": "PA=F",  "sembol": "PALADYUM","para": "$", "renk": "#CED0DD"},
}

# ─────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Teknik Analiz Platformu",
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
  div[data-baseweb="tab-list"] { background: #161B22; gap: 4px; }
  div[data-baseweb="tab"]      { background: #21262D; color: #8B949E;
                                  border-radius: 6px 6px 0 0; }
  div[aria-selected="true"]    { background: #2E75B6 !important; color: white !important; }
  .diverg-bull { background:rgba(38,166,154,0.15); border:1px solid #26A69A;
                 border-radius:6px; padding:8px 12px; margin:4px 0; }
  .diverg-bear { background:rgba(239,83,80,0.15);  border:1px solid #EF5350;
                 border-radius:6px; padding:8px 12px; margin:4px 0; }
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
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().round(4)

def calc_diosc(h, l, c, n=14):
    """DIOSC = DI+ - DI- (DMI Osilatörü)"""
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    up   = h - h.shift()
    down = l.shift() - l
    pdm  = up.where((up > down) & (up > 0), 0.0)
    ndm  = down.where((down > up) & (down > 0), 0.0)
    atr14 = tr.ewm(span=n, adjust=False).mean()
    pdi   = 100 * pdm.ewm(span=n, adjust=False).mean() / atr14.replace(0, np.nan)
    ndi   = 100 * ndm.ewm(span=n, adjust=False).mean() / atr14.replace(0, np.nan)
    return (pdi - ndi).round(2)

# ─────────────────────────────────────────────
# İNDİKATÖR BAZLI DİVERGENCE (herhangi bir indikatör)
# ─────────────────────────────────────────────
def check_indicator_divergence(close, indicator, lookback=60, window=5):
    """
    (+) Pozitif uyuşmazlık: Fiyat düşük dip, indikatör yüksek dip → AL
    (-) Negatif uyuşmazlık: Fiyat yüksek tepe, indikatör düşük tepe → SAT
    Yok: divergence yok
    """
    n     = len(close)
    start = max(0, n - lookback)
    c_seg = close.iloc[start:].reset_index(drop=True)
    i_seg = indicator.iloc[start:].reset_index(drop=True)

    def local_pivots(s, w):
        highs, lows = [], []
        for idx in range(w, len(s) - w):
            if s.iloc[idx] == s.iloc[idx-w:idx+w+1].max(): highs.append(idx)
            if s.iloc[idx] == s.iloc[idx-w:idx+w+1].min(): lows.append(idx)
        return highs, lows

    c_highs, c_lows = local_pivots(c_seg, window)
    i_highs, i_lows = local_pivots(i_seg, window)

    # Pozitif (bullish) — diplerde
    if len(c_lows) >= 2 and len(i_lows) >= 2:
        ci1, ci2 = c_lows[-2], c_lows[-1]
        # en yakın indikatör dibi
        near = [k for k in i_lows if k <= ci2 + window]
        if len(near) >= 2:
            ii2 = near[-1]; ii1 = near[-2]
            p1, p2 = float(c_seg.iloc[ci1]), float(c_seg.iloc[ci2])
            v1, v2 = float(i_seg.iloc[ii1]), float(i_seg.iloc[ii2])
            if p2 < p1 and v2 > v1:
                return "(+)"

    # Negatif (bearish) — tepelerde
    if len(c_highs) >= 2 and len(i_highs) >= 2:
        ci1, ci2 = c_highs[-2], c_highs[-1]
        near = [k for k in i_highs if k <= ci2 + window]
        if len(near) >= 2:
            ii2 = near[-1]; ii1 = near[-2]
            p1, p2 = float(c_seg.iloc[ci1]), float(c_seg.iloc[ci2])
            v1, v2 = float(i_seg.iloc[ii1]), float(i_seg.iloc[ii2])
            if p2 > p1 and v2 < v1:
                return "(-)"

    return "Yok"

# ─────────────────────────────────────────────
# GÖSTERGE ŞABLON TABLOSU
# ─────────────────────────────────────────────
def build_indicator_table(df, rsi_period, mom_period):
    """
    Tablodaki sütunlar:
    İNDİKATÖR | SİNYAL VAR MI? / SON SİNYAL | YÖN | KONUM | UYUŞMAZLIK
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    macd_l, macd_s, macd_h = calc_macd(close)
    rsi     = calc_rsi(close, rsi_period)
    cci     = calc_cci(high, low, close)
    stk_k, stk_d = calc_stoch(high, low, close)
    mom     = calc_mom(close, mom_period)
    diosc   = calc_diosc(high, low, close)

    rows = []

    # ── MACD ──
    macd_val  = float(macd_l.iloc[-1])
    macd_sig  = float(macd_s.iloc[-1])
    macd_prev = float(macd_l.iloc[-2])
    macd_sprev= float(macd_s.iloc[-2])
    # Son sinyal
    if macd_prev < macd_sprev and macd_val > macd_sig:
        sinyal = "Alış"
    elif macd_prev > macd_sprev and macd_val < macd_sig:
        sinyal = "Satış"
    else:
        sinyal = "Alış" if macd_val > macd_sig else "Satış"
    yon    = "↑" if macd_h.iloc[-1] > macd_h.iloc[-2] else "↓"
    konum  = "0 Üzeri" if macd_val > 0 else "0 Altı"
    uyus   = check_indicator_divergence(close, macd_l)
    rows.append(("MACD", sinyal, yon, konum, uyus))

    # ── CCI ──
    cci_val  = float(cci.iloc[-1])
    cci_prev = float(cci.iloc[-2])
    if cci_prev < -100 and cci_val >= -100:   sinyal = "Alış"
    elif cci_prev > 100 and cci_val <= 100:   sinyal = "Satış"
    elif cci_val > 100:                       sinyal = "Aşırı Alım"
    elif cci_val < -100:                      sinyal = "Aşırı Satım"
    else:                                     sinyal = "Nötr"
    yon   = "↑" if cci_val > cci_prev else "↓"
    if cci_val > 100:     konum = "+100 Üzeri"
    elif cci_val < -100:  konum = "-100 Altı"
    else:                 konum = "±100 Bölgesi"
    uyus  = check_indicator_divergence(close, cci)
    rows.append(("CCI", sinyal, yon, konum, uyus))

    # ── RSİ ──
    rsi_val  = float(rsi.iloc[-1])
    rsi_prev = float(rsi.iloc[-2])
    if rsi_prev < 30 and rsi_val >= 30:    sinyal = "Alış"
    elif rsi_prev > 70 and rsi_val <= 70:  sinyal = "Satış"
    elif rsi_val < 30:                     sinyal = "Aşırı Satım"
    elif rsi_val > 70:                     sinyal = "Aşırı Alım"
    else:                                  sinyal = "Nötr"
    yon   = "↑" if rsi_val > rsi_prev else "↓"
    konum = f"{rsi_val:.0f}"
    uyus  = check_indicator_divergence(close, rsi)
    rows.append(("RSİ", sinyal, yon, konum, uyus))

    # ── STOKASTİK ──
    k_val  = float(stk_k.iloc[-1])
    k_prev = float(stk_k.iloc[-2])
    d_val  = float(stk_d.iloc[-1])
    d_prev = float(stk_d.iloc[-2])
    if k_prev < d_prev and k_val >= d_val and k_val < 40:   sinyal = "Alış Sin."
    elif k_prev > d_prev and k_val <= d_val and k_val > 60: sinyal = "Satış Sin."
    elif k_val < 20:   sinyal = "Aşırı Satım"
    elif k_val > 80:   sinyal = "Aşırı Alım"
    else:              sinyal = "Nötr"
    yon   = "↑" if k_val > k_prev else "↓"
    konum = f"{k_val:.0f}"
    uyus  = check_indicator_divergence(close, stk_k)
    rows.append(("STOKASTİK", sinyal, yon, konum, uyus))

    # ── MOMENTUM ──
    mom_val  = float(mom.iloc[-1])
    mom_prev = float(mom.iloc[-2])
    if mom_prev < 0 and mom_val >= 0:   sinyal = "Alımda"
    elif mom_prev > 0 and mom_val <= 0: sinyal = "Satışta"
    elif mom_val > 0:                   sinyal = "Alımda"
    else:                               sinyal = "Satışta"
    yon   = "↑" if mom_val > mom_prev else "↓"
    konum = "+" if mom_val > 0 else "-"
    uyus  = check_indicator_divergence(close, mom)
    rows.append(("MOMENTUM", sinyal, yon, konum, uyus))

    # ── DIOSC ──
    dio_val  = float(diosc.iloc[-1])
    dio_prev = float(diosc.iloc[-2])
    if dio_prev < 0 and dio_val >= 0:   sinyal = "Alımda"
    elif dio_prev > 0 and dio_val <= 0: sinyal = "Satışta"
    elif dio_val > 0:                   sinyal = "Alımda"
    else:                               sinyal = "Satışta"
    yon   = "↑" if dio_val > dio_prev else "↓"
    konum = "+" if dio_val > 0 else "-"
    uyus  = check_indicator_divergence(close, diosc)
    rows.append(("DIOSC", sinyal, yon, konum, uyus))

    return rows

def render_indicator_table(rows, label, key_prefix):
    """Streamlit native tablo olarak göster."""
    st.markdown(f"""
    <div style='background:#2E75B6;color:white;font-weight:700;font-size:13px;
                padding:8px 16px;border-radius:6px 6px 0 0;margin-bottom:2px;'>
        {label}
    </div>
    """, unsafe_allow_html=True)

    headers = ["İNDİKATÖR", "SİNYAL / SON SİNYAL", "YÖN", "KONUM", "UYUŞMAZLIK"]

    # Her satırı ayrı st.columns ile çiz
    col_w = [1.2, 1.4, 0.5, 1.0, 1.0]

    # Başlık satırı
    hcols = st.columns(col_w)
    for hc, h in zip(hcols, headers):
        hc.markdown(
            f"<div style='background:#1F3864;color:#58A6FF;font-weight:700;"
            f"font-size:11px;padding:7px 6px;text-align:center;"
            f"border-bottom:2px solid #2E75B6;'>{h}</div>",
            unsafe_allow_html=True
        )

    for i, (ind, sinyal, yon, konum, uyus) in enumerate(rows):
        bg = "#161B22" if i % 2 == 0 else "#0D1117"

        # Sinyal rengi
        if any(x in sinyal for x in ["Alış", "Alımda", "Aşırı Satım"]):
            sc = "#26A69A"
        elif any(x in sinyal for x in ["Satış", "Satışta", "Aşırı Alım"]):
            sc = "#EF5350"
        else:
            sc = "#8B949E"

        yon_clr = "#26A69A" if yon == "↑" else "#EF5350"

        if uyus == "(+)":   uc = "#26A69A"; ubg = "rgba(38,166,154,0.15)"
        elif uyus == "(-)": uc = "#EF5350"; ubg = "rgba(239,83,80,0.15)"
        else:               uc = "#484F58"; ubg = "transparent"

        cell = (f"padding:8px 6px;text-align:center;background:{bg};"
                f"border-bottom:1px solid #21262D;font-size:12px;")

        rcols = st.columns(col_w)
        rcols[0].markdown(
            f"<div style='{cell}text-align:left;padding-left:10px;"
            f"font-weight:700;color:#C9D1D9;'>{ind}</div>",
            unsafe_allow_html=True)
        rcols[1].markdown(
            f"<div style='{cell}color:{sc};font-weight:600;'>{sinyal}</div>",
            unsafe_allow_html=True)
        rcols[2].markdown(
            f"<div style='{cell}font-size:18px;color:{yon_clr};font-weight:700;'>{yon}</div>",
            unsafe_allow_html=True)
        rcols[3].markdown(
            f"<div style='{cell}color:#C9D1D9;'>{konum}</div>",
            unsafe_allow_html=True)
        rcols[4].markdown(
            f"<div style='{cell}background:{ubg};color:{uc};"
            f"font-weight:700;font-size:14px;'>{uyus}</div>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DİVERGENCE TESPİTİ
# ─────────────────────────────────────────────
def find_pivots(series, window=5):
    """Yerel tepe ve dip indekslerini döndür."""
    highs, lows = [], []
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i-window:i+window+1].max():
            highs.append(i)
        if series.iloc[i] == series.iloc[i-window:i+window+1].min():
            lows.append(i)
    return highs, lows

def detect_divergence(close, rsi, window=5, lookback=60):
    """
    Bullish: Fiyat daha düşük dip, RSİ daha yüksek dip → alım sinyali
    Bearish: Fiyat daha yüksek tepe, RSİ daha düşük tepe → satım sinyali
    Son 'lookback' mum içinde arar.
    """
    n = len(close)
    start = max(0, n - lookback)
    c_seg = close.iloc[start:]
    r_seg = rsi.iloc[start:]

    price_highs, price_lows = find_pivots(c_seg, window)
    rsi_highs,   rsi_lows   = find_pivots(r_seg, window)

    bulls, bears = [], []

    # Bullish divergence: son iki fiyat dibi
    for j in range(1, len(price_lows)):
        i1, i2 = price_lows[j-1], price_lows[j]
        # RSİ diplerinden en yakınını bul
        near = [k for k in rsi_lows if abs(k - i2) <= window*2]
        if not near:
            continue
        r2_idx = min(near, key=lambda k: abs(k - i2))
        near1  = [k for k in rsi_lows if k < r2_idx]
        if not near1:
            continue
        r1_idx = near1[-1]

        p1, p2 = float(c_seg.iloc[i1]), float(c_seg.iloc[i2])
        r1, r2 = float(r_seg.iloc[r1_idx]), float(r_seg.iloc[r2_idx])

        if p2 < p1 and r2 > r1 and r2 < 50:   # fiyat düştü, RSİ yükseldi
            date = c_seg.index[i2]
            bulls.append({
                "tarih": date.strftime("%d.%m.%Y"),
                "fiyat1": p1, "fiyat2": p2,
                "rsi1": r1, "rsi2": r2,
                "idx": start + i2,
            })

    # Bearish divergence: son iki fiyat tepesi
    for j in range(1, len(price_highs)):
        i1, i2 = price_highs[j-1], price_highs[j]
        near = [k for k in rsi_highs if abs(k - i2) <= window*2]
        if not near:
            continue
        r2_idx = min(near, key=lambda k: abs(k - i2))
        near1  = [k for k in rsi_highs if k < r2_idx]
        if not near1:
            continue
        r1_idx = near1[-1]

        p1, p2 = float(c_seg.iloc[i1]), float(c_seg.iloc[i2])
        r1, r2 = float(r_seg.iloc[r1_idx]), float(r_seg.iloc[r2_idx])

        if p2 > p1 and r2 < r1 and r2 > 50:   # fiyat yükseldi, RSİ düştü
            date = c_seg.index[i2]
            bears.append({
                "tarih": date.strftime("%d.%m.%Y"),
                "fiyat1": p1, "fiyat2": p2,
                "rsi1": r1, "rsi2": r2,
                "idx": start + i2,
            })

    return bulls[-3:], bears[-3:]   # son 3'er tane

# ─────────────────────────────────────────────
# SİNYAL SİSTEMİ
# ─────────────────────────────────────────────
def calc_signals(close, high, low, rsi_period, mom_period):
    macd_l, macd_s, _ = calc_macd(close)
    rsi  = calc_rsi(close, rsi_period)
    stk_k, stk_d = calc_stoch(high, low, close)
    mom  = calc_mom(close, mom_period)
    e20  = ema(close, 20)
    e50  = ema(close, 50)
    score = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(close)):
        s = 0
        if macd_l.iloc[i-1] < macd_s.iloc[i-1] and macd_l.iloc[i] > macd_s.iloc[i]: s += 1
        elif macd_l.iloc[i-1] > macd_s.iloc[i-1] and macd_l.iloc[i] < macd_s.iloc[i]: s -= 1
        if rsi.iloc[i-1] < 30 and rsi.iloc[i] >= 30: s += 1
        elif rsi.iloc[i-1] > 70 and rsi.iloc[i] <= 70: s -= 1
        if stk_k.iloc[i-1] < stk_d.iloc[i-1] and stk_k.iloc[i] > stk_d.iloc[i] and stk_k.iloc[i] < 40: s += 1
        elif stk_k.iloc[i-1] > stk_d.iloc[i-1] and stk_k.iloc[i] < stk_d.iloc[i] and stk_k.iloc[i] > 60: s -= 1
        if e20.iloc[i-1] < e50.iloc[i-1] and e20.iloc[i] > e50.iloc[i]: s += 1
        elif e20.iloc[i-1] > e50.iloc[i-1] and e20.iloc[i] < e50.iloc[i]: s -= 1
        if mom.iloc[i-1] < 0 and mom.iloc[i] >= 0: s += 1
        elif mom.iloc[i-1] > 0 and mom.iloc[i] <= 0: s -= 1
        score.iloc[i] = s
    return score

def signal_summary(close, high, low, rsi_period, mom_period):
    macd_l, macd_s, _ = calc_macd(close)
    rsi  = calc_rsi(close, rsi_period)
    stk_k, stk_d = calc_stoch(high, low, close)
    mom  = calc_mom(close, mom_period)
    e20  = ema(close, 20)
    e50  = ema(close, 50)
    rows = []
    rows.append(("MACD-AS", "📗 YUKARI" if macd_l.iloc[-1]>macd_s.iloc[-1] else "📕 AŞAĞI",
                 "Sinyal üzeri" if macd_l.iloc[-1]>macd_s.iloc[-1] else "Sinyal altı",
                 "al" if macd_l.iloc[-1]>macd_s.iloc[-1] else "sat"))
    rv = float(rsi.iloc[-1])
    if rv < 30:   rows.append(("RSİ", f"📗 {rv:.1f} AŞ.SATIM", "Alım bölgesi", "al"))
    elif rv > 70: rows.append(("RSİ", f"📕 {rv:.1f} AŞ.ALIM",  "Satım bölgesi","sat"))
    else:         rows.append(("RSİ", f"⬜ {rv:.1f} NÖTR",     "30-70 arası",  "notr"))
    kv = float(stk_k.iloc[-1])
    if stk_k.iloc[-1]>stk_d.iloc[-1] and kv<80:   rows.append(("STOCH",f"📗 {kv:.0f} YUKARI","%K üstte","al"))
    elif stk_k.iloc[-1]<stk_d.iloc[-1] and kv>20: rows.append(("STOCH",f"📕 {kv:.0f} AŞAĞI", "%K altta","sat"))
    else:                                           rows.append(("STOCH",f"⬜ {kv:.0f} NÖTR","Aşırı bölge","notr"))
    rows.append(("EMA 20/50",
                 "📗 YUKARI" if e20.iloc[-1]>e50.iloc[-1] else "📕 AŞAĞI",
                 "EMA20>50"  if e20.iloc[-1]>e50.iloc[-1] else "EMA20<50",
                 "al" if e20.iloc[-1]>e50.iloc[-1] else "sat"))
    mv = float(mom.iloc[-1])
    rows.append(("MOMENTUM", f"📗 +{mv:.2f}" if mv>0 else f"📕 {mv:.2f}",
                 "Pozitif" if mv>0 else "Negatif", "al" if mv>0 else "sat"))
    return rows

# ─────────────────────────────────────────────
# DESTEK / DİRENÇ
# ─────────────────────────────────────────────
def find_support_resistance(high, low, close, window=10, max_levels=5, tolerance=0.015):
    supports, resistances = [], []
    n = len(close)
    for i in range(window, n - window):
        if low.iloc[i]  == low.iloc[i-window:i+window+1].min():  supports.append(float(low.iloc[i]))
        if high.iloc[i] == high.iloc[i-window:i+window+1].max(): resistances.append(float(high.iloc[i]))
    def cluster(levels, tol):
        if not levels: return []
        levels = sorted(levels)
        cls = [[levels[0]]]
        for lv in levels[1:]:
            if (lv - cls[-1][-1]) / cls[-1][-1] < tol: cls[-1].append(lv)
            else: cls.append([lv])
        return [np.mean(c) for c in cls]
    lp  = float(close.iloc[-1])
    sup = sorted(cluster(supports, tolerance),    key=lambda x: abs(x-lp))[:max_levels]
    res = sorted(cluster(resistances, tolerance), key=lambda x: abs(x-lp))[:max_levels]
    return sorted(sup), sorted(res)

# ─────────────────────────────────────────────
# VERİ ÇEK
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_bist(ticker, days, interval):
    symbol = ticker if ticker.endswith(".IS") else ticker + ".IS"
    df = yf.download(symbol, period=f"{days}d", interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df.astype(float)

@st.cache_data(ttl=300)
def fetch_commodity(ticker, days, interval):
    df = yf.download(ticker, period=f"{days}d", interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[cols].dropna()
    if "Volume" not in df.columns: df["Volume"] = 0
    df.index = pd.to_datetime(df.index)
    return df.astype(float)

# ─────────────────────────────────────────────
# GRAFİK OLUŞTURUCU (tek zaman dilimi)
# ─────────────────────────────────────────────
def build_figure(df, name, label, rsi_period, mom_period, show_signals, show_sr, para="₺"):
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

    # Divergence işaretleri RSİ grafiğinde
    bull_divs, bear_divs = detect_divergence(close, rsi, lookback=80)

    fig = make_subplots(
        rows=8, cols=1, shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.25, 0.09, 0.12, 0.11, 0.11, 0.11, 0.11, 0.10],
        subplot_titles=["FİYAT","ATR (14)","MACD-AS (12,26,9)",
                        f"RSİ ({rsi_period})","CCI (14)",
                        f"MOMENTUM ({mom_period})","STOKASTİK (14,3,3)","HACİM"],
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

    if show_sr:
        for lv in sup_levels:
            fig.add_hline(y=lv, line=dict(color="#26A69A", width=1, dash="dot"),
                annotation_text=f"D {lv:.2f}", annotation_font=dict(color="#26A69A", size=9),
                annotation_position="right", row=1, col=1)
        for lv in res_levels:
            fig.add_hline(y=lv, line=dict(color="#EF5350", width=1, dash="dot"),
                annotation_text=f"R {lv:.2f}", annotation_font=dict(color="#EF5350", size=9),
                annotation_position="right", row=1, col=1)

    if show_signals:
        for mask, y_fn, sym, clr, nm in [
            (signals>=2,  lambda: low[signals>=2]*0.985,  "triangle-up",   "#00E676", "💪 Güçlü Alım"),
            (signals<=-2, lambda: high[signals<=-2]*1.015,"triangle-down", "#FF1744", "💪 Güçlü Satım"),
            (signals==1,  lambda: low[signals==1]*0.990,  "triangle-up",   "rgba(38,166,154,0.5)", "Zayıf Alım"),
            (signals==-1, lambda: high[signals==-1]*1.010,"triangle-down", "rgba(239,83,80,0.5)",  "Zayıf Satım"),
        ]:
            if mask.any():
                fig.add_trace(go.Scatter(x=df.index[mask], y=y_fn(), mode="markers", name=nm,
                    marker=dict(symbol=sym, size=13 if "Güçlü" in nm else 8, color=clr)), row=1, col=1)

    # Divergence okları fiyat grafiğinde
    for d in bull_divs:
        fig.add_annotation(x=df.index[d["idx"]], y=float(close.iloc[d["idx"]]) * 0.97,
            text="🔼 BULL DIV", font=dict(color="#00E676", size=9),
            showarrow=True, arrowhead=2, arrowcolor="#00E676",
            ax=0, ay=30, row=1, col=1)
    for d in bear_divs:
        fig.add_annotation(x=df.index[d["idx"]], y=float(close.iloc[d["idx"]]) * 1.03,
            text="🔽 BEAR DIV", font=dict(color="#FF1744", size=9),
            showarrow=True, arrowhead=2, arrowcolor="#FF1744",
            ax=0, ay=-30, row=1, col=1)

    # ── 2. ATR ──
    fig.add_trace(go.Scatter(x=df.index, y=atr, name="ATR(14)",
        line=dict(color="#00BCD4", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.1)"), row=2, col=1)

    # ── 3. MACD-AS ──
    ch = ["#26A69A" if v>=0 else "#EF5350" for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=macd_h, name="AS", marker_color=ch, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_l, name="MACD-AS", line=dict(color="#2196F3", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_s, name="Sinyal",  line=dict(color="#FF5722", width=1.5, dash="dot")), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dash"), row=3, col=1)
    if show_signals:
        for i in range(1, len(macd_l)):
            if macd_l.iloc[i-1]<macd_s.iloc[i-1] and macd_l.iloc[i]>macd_s.iloc[i]:
                fig.add_trace(go.Scatter(x=[df.index[i]], y=[macd_l.iloc[i]], mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#00E676"), showlegend=False), row=3, col=1)
            elif macd_l.iloc[i-1]>macd_s.iloc[i-1] and macd_l.iloc[i]<macd_s.iloc[i]:
                fig.add_trace(go.Scatter(x=[df.index[i]], y=[macd_l.iloc[i]], mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="#FF1744"), showlegend=False), row=3, col=1)

    # ── 4. RSİ + Divergence bölgeleri ──
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name=f"RSİ({rsi_period})",
        line=dict(color="#FF9800", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.07)"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_ma, name="RSİ MA(8)",
        line=dict(color="#FFFFFF", width=1.2, dash="dot")), row=4, col=1)
    fig.add_hline(y=70, line=dict(color="#EF5350", width=1, dash="dash"), row=4, col=1)
    fig.add_hline(y=30, line=dict(color="#26A69A", width=1, dash="dash"), row=4, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)",   line_width=0, row=4, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)",  line_width=0, row=4, col=1)
    # Divergence noktaları RSİ grafiğinde
    for d in bull_divs:
        fig.add_trace(go.Scatter(x=[df.index[d["idx"]]], y=[rsi.iloc[d["idx"]]],
            mode="markers+text", text=["▲"], textposition="bottom center",
            textfont=dict(color="#00E676", size=11),
            marker=dict(symbol="diamond", size=10, color="#00E676"),
            name="Bullish Div", showlegend=False), row=4, col=1)
    for d in bear_divs:
        fig.add_trace(go.Scatter(x=[df.index[d["idx"]]], y=[rsi.iloc[d["idx"]]],
            mode="markers+text", text=["▼"], textposition="top center",
            textfont=dict(color="#FF1744", size=11),
            marker=dict(symbol="diamond", size=10, color="#FF1744"),
            name="Bearish Div", showlegend=False), row=4, col=1)

    # ── 5. CCI ──
    fig.add_trace(go.Scatter(x=df.index, y=cci, name="CCI(14)",   line=dict(color="#EF5350", width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=cci_ma, name="CCI MA(8)", line=dict(color="#FFEB3B", width=1.2, dash="dot")), row=5, col=1)
    fig.add_hline(y=100,  line=dict(color="#EF5350", width=1, dash="dash"), row=5, col=1)
    fig.add_hline(y=-100, line=dict(color="#26A69A", width=1, dash="dash"), row=5, col=1)
    fig.add_hline(y=0,    line=dict(color="gray",    width=0.5, dash="dot"), row=5, col=1)

    # ── 6. MOMENTUM ──
    mc = ["#26A69A" if v>=0 else "#EF5350" for v in mom.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=mom, name=f"Mom({mom_period})", marker_color=mc, opacity=0.6), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=mom_ma, name="Mom MA(8)", line=dict(color="#CE93D8", width=1.5)), row=6, col=1)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5, dash="dash"), row=6, col=1)

    # ── 7. STOKASTİK ──
    fig.add_trace(go.Scatter(x=df.index, y=stk_k, name="%K", line=dict(color="#1565C0", width=1.5)), row=7, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stk_d, name="%D", line=dict(color="#E53935", width=1.5, dash="dot")), row=7, col=1)
    fig.add_hline(y=80, line=dict(color="#EF5350", width=1, dash="dash"), row=7, col=1)
    fig.add_hline(y=20, line=dict(color="#26A69A", width=1, dash="dash"), row=7, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,83,80,0.06)",  line_width=0, row=7, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=7, col=1)

    # ── 8. HACİM ──
    if vol is not None:
        vc = ["#26A69A" if c>=o else "#EF5350" for c,o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=vol, name="Hacim", marker_color=vc, opacity=0.6), row=8, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{name}</b> — {label} Teknik Analiz",
            font=dict(size=16, color="#E0E0E0"), x=0.5, y=0.99),
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font=dict(family="Arial", color="#C9D1D9", size=10),
        height=1250,
        margin=dict(l=60, r=180, t=40, b=30),
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top",
            bgcolor="#161B22", bordercolor="#30363D", borderwidth=1,
            font=dict(size=11, color="#C9D1D9"), tracegroupgap=4),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#8B949E", activecolor="#58A6FF",
            remove=["lasso2d","select2d","autoScale2d",
                    "hoverClosestCartesian","hoverCompareCartesian","toggleSpikelines"]),
    )
    ax = dict(gridcolor="#21262D", gridwidth=0.5, zerolinecolor="#30363D", zerolinewidth=1, tickfont=dict(size=8))
    for i in range(1, 9):
        fig.update_yaxes(ax, row=i, col=1)
        fig.update_xaxes(gridcolor="#21262D", gridwidth=0.5, showticklabels=(i==8), row=i, col=1)
    fig.update_yaxes(range=[0,100], row=4, col=1)
    fig.update_yaxes(range=[0,100], row=7, col=1)
    for ann in fig.layout.annotations:
        ann.font.color="#58A6FF"; ann.font.size=11; ann.font.family="Arial"
        ann.x=0.01; ann.xanchor="left"; ann.xref="paper"
    return fig

# ─────────────────────────────────────────────
# ÇOKLU ZAMAN DİLİMİ — Yan Yana Fiyat + RSİ
# ─────────────────────────────────────────────
def build_mtf_figure(df_d, df_w, df_m, name, rsi_period, para="₺"):
    """Günlük / Haftalık / Aylık fiyat + RSİ yan yana karşılaştırma."""
    datasets = [
        ("Günlük",  df_d, "#2196F3"),
        ("Haftalık",df_w, "#FFA726"),
        ("Aylık",   df_m, "#AB47BC"),
    ]

    fig = make_subplots(
        rows=2, cols=3,
        shared_xaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=[
            "📈 GÜNLÜK Fiyat", "📈 HAFTALIK Fiyat", "📈 AYLIK Fiyat",
            f"📉 GÜNLÜK RSİ ({rsi_period})", f"📉 HAFTALIK RSİ ({rsi_period})", f"📉 AYLIK RSİ ({rsi_period})",
        ],
    )

    for col, (lbl, df, clr) in enumerate(datasets, start=1):
        if df is None or df.empty:
            continue
        close = df["Close"]
        rsi   = calc_rsi(close, rsi_period)
        op, high, low = df["Open"], df["High"], df["Low"]

        # Fiyat mumu
        fig.add_trace(go.Candlestick(
            x=df.index, open=op, high=high, low=low, close=close, name=f"{lbl} Fiyat",
            increasing=dict(line=dict(color="#26A69A"), fillcolor="#26A69A"),
            decreasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
            showlegend=False,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(x=df.index, y=ema(close,20), name=f"{lbl} EMA20",
            line=dict(color="#FFA726", width=1), showlegend=False), row=1, col=col)

        # RSİ
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name=f"{lbl} RSİ",
            line=dict(color=clr, width=1.5),
            fill="tozeroy", fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.08)",
            showlegend=False), row=2, col=col)
        fig.add_hline(y=70, line=dict(color="#EF5350", width=0.8, dash="dash"), row=2, col=col)
        fig.add_hline(y=30, line=dict(color="#26A69A", width=0.8, dash="dash"), row=2, col=col)

    fig.update_layout(
        title=dict(text=f"<b>{name}</b> — Çoklu Zaman Dilimi Karşılaştırma",
            font=dict(size=16, color="#E0E0E0"), x=0.5),
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font=dict(family="Arial", color="#C9D1D9", size=10),
        height=700,
        margin=dict(l=50, r=30, t=60, b=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
    )
    ax = dict(gridcolor="#21262D", gridwidth=0.5, tickfont=dict(size=8))
    for r in range(1, 3):
        for c in range(1, 4):
            fig.update_yaxes(ax, row=r, col=c)
            fig.update_xaxes(ax, row=r, col=c)
    fig.update_yaxes(range=[0,100], row=2, col=1)
    fig.update_yaxes(range=[0,100], row=2, col=2)
    fig.update_yaxes(range=[0,100], row=2, col=3)
    for ann in fig.layout.annotations:
        ann.font.color="#58A6FF"; ann.font.size=11
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Teknik Analiz")
    st.markdown("---")

    mod = st.radio("Piyasa Seçin", ["🇹🇷 BİST Hisseleri", "🏅 Emtia"], horizontal=False)
    st.markdown("---")

    if mod == "🇹🇷 BİST Hisseleri":
        ticker_input = st.text_input("Hisse Kodu", value="THYAO",
            placeholder="THYAO, SISE, GARAN...").upper().strip()
        is_commodity = False
        commodity_key = None
    else:
        commodity_key = st.selectbox("Emtia Seçin", list(EMTIALAR.keys()))
        ticker_input  = EMTIALAR[commodity_key]["ticker"]
        is_commodity  = True

    st.markdown("**Veri Aralığı**")
    gunluk_gun   = st.slider("Günlük (gün)",   90,  730,  365)
    haftalik_gun = st.slider("Haftalık (gün)", 180, 1460, 730)

    st.markdown("**İndikatör Ayarları**")
    rsi_period = st.number_input("RSİ Periyot",      2, 50, 10)
    mom_period = st.number_input("Momentum Periyot", 2, 50, 10)

    st.markdown("**Göster / Gizle**")
    show_signals = st.toggle("🔔 Alım/Satım Sinyalleri", value=True)
    show_sr      = st.toggle("📐 Destek / Direnç",       value=True)
    show_div     = st.toggle("🔀 Divergence Tespiti",     value=True)
    show_mtf     = st.toggle("🕐 Çoklu Zaman Dilimi",    value=True)

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
        "last_ticker": ticker_input, "last_gd": gunluk_gun, "last_hd": haftalik_gun,
        "last_rsi": rsi_period, "last_mom": mom_period,
        "last_signals": show_signals, "last_sr": show_sr,
        "last_div": show_div, "last_mtf": show_mtf,
        "last_is_comm": is_commodity, "last_comm_key": commodity_key,
    })

t    = st.session_state.get("last_ticker", "")
gd   = st.session_state.get("last_gd",  gunluk_gun)
hd   = st.session_state.get("last_hd",  haftalik_gun)
rp   = st.session_state.get("last_rsi", rsi_period)
mp   = st.session_state.get("last_mom", mom_period)
ss   = st.session_state.get("last_signals", show_signals)
sr   = st.session_state.get("last_sr",      show_sr)
sd   = st.session_state.get("last_div",     show_div)
smtf = st.session_state.get("last_mtf",     show_mtf)
is_c = st.session_state.get("last_is_comm", is_commodity)
ck   = st.session_state.get("last_comm_key", commodity_key)

if not t:
    st.markdown("""<div style='text-align:center;padding:80px;color:#484F58;'>
      <div style='font-size:52px;'>📈</div>
      <div style='font-size:20px;margin-top:16px;'>
        Sol panelden seçim yapın ve <b>ANALİZ ET</b> butonuna basın
      </div></div>""", unsafe_allow_html=True)
    st.stop()

# Veriyi çek
with st.spinner(f"📥 {t} verisi çekiliyor..."):
    if is_c:
        df_d = fetch_commodity(t, gd,   "1d")
        df_w = fetch_commodity(t, hd,   "1wk")
        df_m = fetch_commodity(t, 1825, "1mo")
        para      = EMTIALAR[ck]["para"] if ck else "$"
        disp_name = EMTIALAR[ck]["sembol"] if ck else t
        borsа     = "Vadeli Emtia"
    else:
        df_d = fetch_bist(t, gd,   "1d")
        df_w = fetch_bist(t, hd,   "1wk")
        df_m = fetch_bist(t, 1825, "1mo")
        para      = "₺"
        disp_name = f"IST:{t}"
        borsа     = "Borsa İstanbul"

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
    <div style='font-size:28px;font-weight:900;color:#58A6FF;'>{disp_name}</div>
    <div style='font-size:11px;color:#8B949E;'>{borsа}</div>
  </div>
  <div>
    <div style='font-size:26px;font-weight:800;color:#E0E0E0;'>{para}{float(last["Close"]):.2f}</div>
    <div style='font-size:16px;font-weight:700;color:{clr};'>{sign} {para}{abs(chg):.2f} ({chg_pct:+.2f}%)</div>
  </div>
  <div style='font-size:12px;color:#8B949E;line-height:1.9'>
    Açılış: <b style='color:#E0E0E0'>{para}{float(last["Open"]):.2f}</b> &nbsp;|&nbsp;
    Yüksek: <b style='color:#26A69A'>{para}{float(last["High"]):.2f}</b> &nbsp;|&nbsp;
    Düşük: <b style='color:#EF5350'>{para}{float(last["Low"]):.2f}</b> &nbsp;|&nbsp;
    Tarih: <b style='color:#E0E0E0'>{df_d.index[-1].strftime("%d.%m.%Y")}</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sinyal Özeti ──
if ss:
    summary   = signal_summary(df_d["Close"], df_d["High"], df_d["Low"], rp, mp)
    al_count  = sum(1 for r in summary if r[3]=="al")
    sat_count = sum(1 for r in summary if r[3]=="sat")
    total     = len(summary)
    if al_count > sat_count:   oclr="#26A69A"; otxt=f"📗 GENEL: ALIM EĞİLİMİ ({al_count}/{total})"
    elif sat_count > al_count: oclr="#EF5350"; otxt=f"📕 GENEL: SATIM EĞİLİMİ ({sat_count}/{total})"
    else:                      oclr="#8B949E"; otxt=f"⬜ GENEL: NÖTR ({al_count}/{sat_count})"
    with st.expander("🔔 GÜNLÜK SİNYAL ÖZETİ", expanded=True):
        st.markdown(f"""<div style='background:rgba(0,0,0,0.3);border:2px solid {oclr};
            border-radius:8px;padding:10px 16px;margin-bottom:10px;
            font-size:15px;font-weight:700;color:{oclr};'>{otxt}</div>""", unsafe_allow_html=True)
        cols = st.columns(len(summary))
        for col, (ind, durum, aciklama, tip) in zip(cols, summary):
            bg  = "rgba(38,166,154,0.12)" if tip=="al" else "rgba(239,83,80,0.12)" if tip=="sat" else "rgba(50,50,50,0.2)"
            brd = "#26A69A" if tip=="al" else "#EF5350" if tip=="sat" else "#30363D"
            col.markdown(f"""<div style='background:{bg};border:1px solid {brd};border-radius:8px;
                padding:10px;text-align:center;'>
                <div style='font-size:11px;color:#8B949E;margin-bottom:4px;'>{ind}</div>
                <div style='font-size:12px;font-weight:700;'>{durum}</div>
                <div style='font-size:10px;color:#8B949E;margin-top:4px;'>{aciklama}</div>
                </div>""", unsafe_allow_html=True)

# ── Divergence Özeti ──
if sd:
    rsi_d = calc_rsi(df_d["Close"], rp)
    bull_divs, bear_divs = detect_divergence(df_d["Close"], rsi_d, lookback=80)
    if bull_divs or bear_divs:
        with st.expander("🔀 DİVERGENCE TESPİTİ", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 Bullish Divergence (Alım Sinyali)**")
                if bull_divs:
                    for d in reversed(bull_divs):
                        st.markdown(f"""<div class='diverg-bull'>
                            <b style='color:#00E676'>📅 {d["tarih"]}</b><br>
                            Fiyat: {para}{d["fiyat1"]:.2f} → {para}{d["fiyat2"]:.2f} ↓ &nbsp;|&nbsp;
                            RSİ: {d["rsi1"]:.1f} → {d["rsi2"]:.1f} ↑<br>
                            <small style='color:#8B949E'>Fiyat düştü ama RSİ yükseldi → Güç toplanıyor</small>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Son 80 mumda bullish divergence yok")
            with c2:
                st.markdown("**🔴 Bearish Divergence (Satım Sinyali)**")
                if bear_divs:
                    for d in reversed(bear_divs):
                        st.markdown(f"""<div class='diverg-bear'>
                            <b style='color:#FF1744'>📅 {d["tarih"]}</b><br>
                            Fiyat: {para}{d["fiyat1"]:.2f} → {para}{d["fiyat2"]:.2f} ↑ &nbsp;|&nbsp;
                            RSİ: {d["rsi1"]:.1f} → {d["rsi2"]:.1f} ↓<br>
                            <small style='color:#8B949E'>Fiyat yükseldi ama RSİ düştü → Güç zayıflıyor</small>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Son 80 mumda bearish divergence yok")

# ── Destek/Direnç ──
if sr:
    sup_d, res_d = find_support_resistance(df_d["High"], df_d["Low"], df_d["Close"])
    lp = float(df_d["Close"].iloc[-1])
    with st.expander("📐 DESTEK / DİRENÇ SEVİYELERİ", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 Destek**")
            for lv in sorted(sup_d, reverse=True):
                uz = (lp - lv) / lp * 100
                st.markdown(f"""<div style='background:rgba(38,166,154,0.1);border-left:3px solid #26A69A;
                    padding:6px 12px;margin:4px 0;border-radius:4px;display:flex;justify-content:space-between;'>
                    <span style='font-weight:700;color:#26A69A;'>{para}{lv:.2f}</span>
                    <span style='color:#8B949E;font-size:11px;'>%{uz:.1f} uzakta</span></div>""",
                    unsafe_allow_html=True)
        with c2:
            st.markdown("**🔴 Direnç**")
            for lv in sorted(res_d):
                uz = (lv - lp) / lp * 100
                st.markdown(f"""<div style='background:rgba(239,83,80,0.1);border-left:3px solid #EF5350;
                    padding:6px 12px;margin:4px 0;border-radius:4px;display:flex;justify-content:space-between;'>
                    <span style='font-weight:700;color:#EF5350;'>{para}{lv:.2f}</span>
                    <span style='color:#8B949E;font-size:11px;'>%{uz:.1f} uzakta</span></div>""",
                    unsafe_allow_html=True)

# ── Gösterge Şablon Tablosu ──
with st.expander("📋 GÖSTERGE ŞABLON TABLOSU", expanded=True):
    col_g, col_h = st.columns(2)
    with col_g:
        try:
            rows_g = build_indicator_table(df_d, rp, mp)
            render_indicator_table(rows_g, "📅 GÜNLÜK GÖSTERGE ŞABLONU", "gunluk")
        except Exception as e:
            st.warning(f"Günlük tablo hesaplanamadı: {e}")
    with col_h:
        try:
            rows_h = build_indicator_table(df_w, rp, mp)
            render_indicator_table(rows_h, "📆 HAFTALIK GÖSTERGE ŞABLONU", "haftalik")
        except Exception as e:
            st.warning(f"Haftalık tablo hesaplanamadı: {e}")
    st.markdown("""
    <div style='margin-top:10px;font-size:11px;color:#484F58;padding:6px 12px;
                background:#161B22;border-radius:6px;border-left:3px solid #2E75B6;'>
      <b style='color:#58A6FF'>Uyuşmazlık:</b> &nbsp;
      <span style='color:#26A69A'><b>(+) Pozitif</b> = Diplerde AL sinyali</span> &nbsp;|&nbsp;
      <span style='color:#EF5350'><b>(-) Negatif</b> = Tepelerde SAT sinyali</span> &nbsp;|&nbsp;
      <span style='color:#484F58'>Yok = Uyuşmazlık yok</span>
    </div>
    """, unsafe_allow_html=True)

# ── Sekmeler ──
tabs = st.tabs(["📈 Günlük", "📊 Haftalık", "🕐 Çoklu Zaman Dilimi"])

with tabs[0]:
    st.plotly_chart(build_figure(df_d, disp_name, "GÜNLÜK",   rp, mp, ss, sr, para), use_container_width=True)

with tabs[1]:
    st.plotly_chart(build_figure(df_w, disp_name, "HAFTALIK", rp, mp, ss, sr, para), use_container_width=True)

with tabs[2]:
    if smtf:
        st.plotly_chart(build_mtf_figure(df_d, df_w, df_m, disp_name, rp, para), use_container_width=True)
        st.markdown("""<div style='background:#161B22;border:1px solid #30363D;border-radius:8px;
            padding:12px 20px;font-size:12px;color:#8B949E;margin-top:8px;'>
            💡 <b style='color:#C9D1D9'>Çoklu Zaman Dilimi Okuma:</b>
            Aylık grafik genel trendi, haftalık orta vadeli yönü, günlük ise giriş zamanlamasını gösterir.
            Üç zaman diliminde de aynı yönde sinyal varsa <b style='color:#58A6FF'>en güçlü sinyaldir.</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Çoklu zaman dilimini görmek için sol panelden toggle'ı açın.")
