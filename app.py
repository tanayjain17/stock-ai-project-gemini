import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from textblob import TextBlob
import feedparser
import urllib.parse
from datetime import datetime, time as dt_time
import time
import tensorflow as tf
import pytz 
import random 

# --- 0. SEED SETTING ---
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Market Pulse AI", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stStatusWidget { visibility: hidden; }
    .metric-card { background-color: #1e2330; border: 1px solid #2a2f3d; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    .status-live { background-color: #00e676; color: black; box-shadow: 0 0 8px rgba(0, 230, 118, 0.4); }
    .status-closed { background-color: #ff1744; color: white; }
    .trade-plan { border: 1px solid #4caf50; background-color: #1b2e1b; padding: 15px; border-radius: 10px; margin-top: 10px; }
    .news-card { background-color: #151922; padding: 15px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #2962ff; transition: transform 0.2s; }
    .news-card:hover { transform: translateX(5px); background-color: #1a1f2b; }
    .news-meta { font-size: 11px; color: #888; display: flex; justify-content: space-between; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET DATABASE ---
NIFTY_100_TICKERS = {
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "L&T": "LT.NS",
    "Kotak Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "Tata Motors": "TATAMOTORS.NS",
    "Maruti Suzuki": "MARUTI.NS", "Bajaj Finance": "BAJFINANCE.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Titan": "TITAN.NS", "NTPC": "NTPC.NS", "Power Grid": "POWERGRID.NS", "ONGC": "ONGC.NS",
    "Zomato": "ZOMATO.NS", "Adani Ent": "ADANIENT.NS", "DLF": "DLF.NS", "HAL": "HAL.NS",
    "Tata Steel": "TATASTEEL.NS", "Hindalco": "HINDALCO.NS", "Jio Financial": "JIOFIN.NS"
}

ETFS_MFS = {
    "Nifty 50 ETF": "NIFTYBEES.NS", "Bank Nifty ETF": "BANKBEES.NS", "IT ETF": "ITBEES.NS",
    "Gold ETF": "GOLDBEES.NS", "Silver ETF": "SILVERBEES.NS", "Nasdaq 100": "MON100.NS", "CPSE ETF": "CPSEETF.NS"
}

COMMODITIES_GLOBAL = {
    "Gold (Global)": "GC=F", "Silver (Global)": "SI=F", "Crude Oil": "CL=F", "Natural Gas": "NG=F", "Copper": "HG=F"
}

# --- 3. CORE LOGIC ---

def is_market_open(ticker):
    if not (ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^")): return True, "Global"
    utc_now = datetime.now(pytz.utc)
    ist_now = utc_now.astimezone(pytz.timezone('Asia/Kolkata'))
    if ist_now.weekday() >= 5: return False, "Weekend"
    curr_time = ist_now.time()
    if dt_time(9, 15) <= curr_time <= dt_time(15, 30): return True, "Open"
    return False, "Closed"

def get_currency(ticker):
    return "₹" if (ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^")) else "$"

def get_live_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.fast_info.last_price
        prev = stock.fast_info.previous_close
        return price, price - prev, (price - prev)/prev * 100
    except: return 0.0, 0.0, 0.0

def get_news(query):
    try:
        cb = random.randint(1, 10000)
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en&cb={cb}"
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:8]:
            blob = TextBlob(e.title)
            sent = "🟢 Bullish" if blob.sentiment.polarity > 0.05 else "🔴 Bearish" if blob.sentiment.polarity < -0.05 else "⚪ Neutral"
            try: ts = time.mktime(e.published_parsed)
            except: ts = 0
            items.append({'title': e.title, 'link': e.link, 'source': e.get('source', {}).get('title', 'News'), 'date': e.get('published','')[:16], 'ts': ts, 'sent': sent})
        return sorted(items, key=lambda x: x['ts'], reverse=True)
    except: return []

def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def train_ai(df):
    data = df[['Close', 'RSI', 'SMA_50', 'EMA_20']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)
    if len(scaled) < 60: return None, None
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 4)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, batch_size=16, epochs=5, verbose=0) 
    last_60 = scaled[-60:].reshape(1, 60, 4)
    pred_scaled = model.predict(last_60)
    dummy = np.zeros((1, 4))
    dummy[0, 0] = pred_scaled[0, 0]
    pred_price = scaler.inverse_transform(dummy)[0, 0]
    last_atr = df['ATR'].iloc[-1]
    return pred_price, last_atr

# --- 4. NAVIGATION ---
st.sidebar.title("⚡ Flash Scanner Pro")
nav_options = ["🏠 Market Dashboard", "📈 Stock Analyzer", "🏦 ETFs & Mutual Funds", "🛢️ Global Commodities"]
view = st.sidebar.radio("Go to:", nav_options)

selected_ticker = "RELIANCE.NS"
is_bse = False

# SELECTORS
if view == "📈 Stock Analyzer":
    t_name = st.sidebar.selectbox("Nifty 100 List", list(NIFTY_100_TICKERS.keys()))
    selected_ticker = NIFTY_100_TICKERS[t_name]
    is_bse = st.sidebar.checkbox("Switch to BSE?", value=False)
    if is_bse: selected_ticker = selected_ticker.replace(".NS", ".BO")
    st.sidebar.markdown("---")
    custom = st.sidebar.text_input("Or Search Stock (e.g. ZOMATO)")
    if custom: selected_ticker = f"{custom.upper()}.BO" if is_bse else f"{custom.upper()}.NS"

elif view == "🏦 ETFs & Mutual Funds":
    t_name = st.sidebar.selectbox("Popular ETFs", list(ETFS_MFS.keys()))
    selected_ticker = ETFS_MFS[t_name]
    st.sidebar.markdown("---")
    is_bse = st.sidebar.checkbox("Switch to BSE?", value=False, key="etf_bse")
    if is_bse: selected_ticker = selected_ticker.replace(".NS", ".BO")
    custom_etf = st.sidebar.text_input("Or Search ETF (e.g. MAFANG)")
    if custom_etf: selected_ticker = f"{custom_etf.upper()}.BO" if is_bse else f"{custom_etf.upper()}.NS"

elif view == "🛢️ Global Commodities":
    t_name = st.sidebar.selectbox("Global Assets", list(COMMODITIES_GLOBAL.keys()))
    selected_ticker = COMMODITIES_GLOBAL[t_name]

# RESTORED SETTINGS
ai_horizon = "Short Term (Intraday)"
chart_range = "1 Day"

if view != "🏠 Market Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    chart_range = st.sidebar.selectbox("Chart History", ["1 Day", "5 Days", "1 Month", "6 Months", "YTD", "1 Year", "5 Years", "Max"])
    ai_horizon = st.sidebar.radio("AI Prediction Horizon", ["Short Term (Intraday)", "Long Term (Delivery)"])


# --- 5. VISUALIZATION FRAGMENTS ---

@st.fragment(run_every=2)
def render_fast_price_header(ticker):
    is_open, status_txt = is_market_open(ticker)
    curr_sym = get_currency(ticker)
    lp, lc, lpct = get_live_data(ticker)
    color = "#00e676" if lc >= 0 else "#ff1744"
    badge_html = f'<span class="status-badge status-live">🟢 LIVE</span>' if is_open else f'<span class="status-badge status-closed">🔴 CLOSED</span>'

    st.markdown(f"""
    <div style="background:#1e2330; padding:20px; border-radius:12px; margin-bottom:10px; border-left: 6px solid {color};">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <div style="color:#aaa; font-size:12px; margin-bottom:5px;">MARKET STATUS &nbsp; {badge_html}</div>
                <div style="font-size:42px; font-weight:bold; color:white;">{curr_sym}{lp:,.2f}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:24px; font-weight:bold; color:{color};">{lc:+.2f}</div>
                <div style="background:{color}20; color:{color}; padding:4px 10px; border-radius:8px; font-weight:bold; display:inline-block;">
                    {lpct:+.2f}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.fragment(run_every=60)
def render_stable_chart(ticker, range_choice):
    range_map = {
        "1 Day": {"p": "1d", "i": "1m"}, "5 Days": {"p": "5d", "i": "15m"},
        "1 Month": {"p": "1mo", "i": "1h"}, "6 Months": {"p": "6mo", "i": "1d"},
        "YTD": {"p": "ytd", "i": "1d"}, "1 Year": {"p": "1y", "i": "1d"},
        "5 Years": {"p": "5y", "i": "1wk"}, "Max": {"p": "max", "i": "1wk"}
    }
    params = range_map[range_choice]
    try:
        df = yf.download(ticker, period=params["p"], interval=params["i"], progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if not df.empty:
            df = add_indicators(df)
            fig = go.Figure()
            
            # --- THE FIX FOR BROKEN CHARTS ---
            # using type='category' removes weekend gaps
            fig.update_xaxes(
                type='category', 
                tickmode='auto', 
                nticks=8,  # Limit ticks to prevent overcrowding
                showgrid=False
            )
            
            fig.add_trace(go.Candlestick(x=df.index.strftime('%Y-%m-%d %H:%M'), open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            if len(df) > 20: fig.add_trace(go.Scatter(x=df.index.strftime('%Y-%m-%d %H:%M'), y=df['EMA_20'], line=dict(color='#2979ff', width=2), name="EMA 20"))
            
            fig.update_layout(height=450, margin=dict(t=30,b=0,l=0,r=0), template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#1e2330", plot_bgcolor="#1e2330", title=dict(text=f"{range_choice} Chart ({ticker})", font=dict(color="#aaa", size=12)))
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No data for this range.")
    except: st.error("Chart Unavailable.")

@st.fragment(run_every=900)
def render_news_feed(query_ticker):
    clean_ticker = query_ticker.replace(".NS","").replace(".BO","")
    st.subheader("📰 Latest News (Auto-Updates)")
    with st.spinner("Updating..."):
        news_items = get_news(clean_ticker)
    if news_items:
        for n in news_items[:6]: 
            st.markdown(f"""
            <div class="news-card">
                <div class="news-meta"><span>🏛️ {n['source']}</span><span>🕒 {n['date']}</span></div>
                <a href="{n['link']}" target="_blank" style="color:#e0e0e0; text-decoration:none; font-weight:600; font-size:15px; display:block; margin-bottom:8px;">{n['title']}</a>
                <div style="font-size:12px; font-weight:bold;">{n['sent']}</div>
            </div>""", unsafe_allow_html=True)
    else: st.info("No recent news found.")

# --- 6. PAGE ROUTING ---

if view == "🏠 Market Dashboard":
    st.title("🌏 Market Dashboard")
    @st.fragment(run_every=5)
    def render_dashboard():
        c1, c2, c3 = st.columns(3)
        indices = [("NIFTY 50", "^NSEI", c1), ("SENSEX", "^BSESN", c2), ("BANK NIFTY", "^NSEBANK", c3)]
        for name, sym, col in indices:
            is_op, _ = is_market_open(sym)
            p, c, pct = get_live_data(sym)
            clr = "#00e676" if c >= 0 else "#ff1744"
            dot = "🟢" if is_op else "🔴"
            with col:
                st.markdown(f'<div class="metric-card" style="border-top: 3px solid {clr};"><div style="font-size:12px; color:#888;">{dot} {name}</div><div style="font-size:26px; font-weight:bold;">₹{p:,.2f}</div><div style="color:{clr}; font-weight:bold;">{c:+.2f} ({pct:+.2f}%)</div></div>', unsafe_allow_html=True)
    render_dashboard()
    st.markdown("---")
    render_news_feed("Indian Stock Market")

else:
    st.title(f"📊 Analysis: {selected_ticker}")
    render_fast_price_header(selected_ticker)
    render_stable_chart(selected_ticker, chart_range)
    st.markdown("---")
    
    c_ai, c_news = st.columns([1, 1])
    
    with c_ai:
        st.subheader("🤖 AI Prediction")
        if st.button(f"Run {ai_horizon} Prediction"):
            with st.spinner("Training Custom Model..."):
                try:
                    # DYNAMIC DATA LOADING BASED ON SELECTION
                    if ai_horizon == "Short Term (Intraday)":
                        df_train = yf.download(selected_ticker, period="5d", interval="5m", progress=False)
                    else:
                        df_train = yf.download(selected_ticker, period="2y", interval="1d", progress=False)
                    
                    if isinstance(df_train.columns, pd.MultiIndex): df_train.columns = df_train.columns.droplevel(1)
                    
                    if len(df_train) > 60:
                        df_train = add_indicators(df_train)
                        target, atr = train_ai(df_train)
                        
                        if target:
                            curr, _, _ = get_live_data(selected_ticker)
                            diff = target - curr
                            sig = "BUY 🚀" if diff > 0 else "SELL 🔻"
                            col = "green" if diff > 0 else "red"
                            curr_sym = get_currency(selected_ticker)
                            
                            st.success(f"AI Target ({ai_horizon}): **{curr_sym}{target:.2f}**")
                            st.markdown(f"Signal: **:{col}[{sig}]** (Potential: {diff:+.2f})")
                            
                            with st.expander("🔐 Unlock Pro Trade Setup"):
                                multiplier = 1.0 if "Intraday" in ai_horizon else 2.0
                                sl = curr - (1.5 * atr * multiplier) if diff > 0 else curr + (1.5 * atr * multiplier)
                                t1 = curr + (1 * atr * multiplier) if diff > 0 else curr - (1 * atr * multiplier)
                                st.markdown(f"""
                                <div class="trade-plan">
                                    <h4 style="margin:0; color:#4caf50;">{ai_horizon} Plan</h4>
                                    <hr style="border-color:#555;">
                                    <p><strong>🔵 ENTRY:</strong> {curr:.2f}</p>
                                    <p><strong>🛑 STOP LOSS:</strong> {sl:.2f}</p>
                                    <p><strong>🎯 TARGET:</strong> {t1:.2f}</p>
                                </div>""", unsafe_allow_html=True)
                        else: st.error("Model failed.")
                    else: st.warning("Not enough data for this timeframe.")
                except Exception as e: st.error(f"Error: {e}")

    with c_news:
        render_news_feed(selected_ticker)
