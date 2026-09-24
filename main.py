import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="AlphaPulse Workstation", layout="wide")

# 1. Initialize Persistent Session States
if "cash_balance" not in st.session_state:
    st.session_state.cash_balance = 100000.00

if "portfolio" not in st.session_state:
    # Structure: {'AAPL': {'shares': 10, 'avg_cost': 180.00}}
    st.session_state.portfolio = {}

if "watchlists" not in st.session_state:
    st.session_state.watchlists = {
        "Tech Titans": ["AAPL", "MSFT", "NVDA", "GOOGL"],
        "Growth / Momentum": ["TSLA", "PLTR", "AMD", "COIN"]
    }

# 2. Sidebar Navigation
st.sidebar.title("⚡ AlphaPulse")
menu_choice = st.sidebar.radio(
    "Navigation",
    ["📡 Scanner", "📋 Watchlists", "💼 Paper Portfolio", "📈 Historical Charts", "🤖 AI Predictor"]
)

# 3. Page Routing
if menu_choice == "📡 Scanner":
    st.title("📡 Swing Trading Scanner")
    # Scanner logic here...

elif menu_choice == "📋 Watchlists":
    st.title("📋 Watchlist Manager (Max 8 Lists)")
    st.write(f"Active Watchlists: {len(st.session_state.watchlists)}/8")
    # Watchlist CRUD operations...

elif menu_choice == "💼 Paper Portfolio":
    st.title("💼 Paper Trading Simulator")
    st.metric("Available Cash", f"${st.session_state.cash_balance:,.2f}")
    # Buy/Sell execution & holdings table...

elif menu_choice == "📈 Historical Charts":
    st.title("📈 Technical Deep-Dive")
    # Plotly candlestick logic...

elif menu_choice == "🤖 AI Predictor":
    st.title("🤖 Fundamental & Technical Price Modeler")
    # Predictive model / scoring engine...

# --- Core Trading Logic ---

class MarketData:
    @staticmethod
    def fetch_single(ticker, period="6mo"):
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                return ticker, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return ticker, df
        except Exception:
            return ticker, None

    @classmethod
    def fetch_batch(cls, tickers, period="6mo"):
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(cls.fetch_single, t, period): t for t in tickers}
            for future in futures:
                ticker, df = future.result()
                if df is not None:
                    results[ticker] = df
        return results

class SwingStrategy:
    def __init__(self, ema_short=20, ema_long=50, rsi_period=14, rsi_min=50, rsi_max=70):
        self.params = {'ema_s': ema_short, 'ema_l': ema_long, 'rsi_p': rsi_period, 'rsi_min': rsi_min, 'rsi_max': rsi_max}

    def apply_indicators(self, df):
        p = self.params
        df['EMA_Short'] = ta.trend.EMAIndicator(df['Close'], window=p['ema_s']).ema_indicator()
        df['EMA_Long'] = ta.trend.EMAIndicator(df['Close'], window=p['ema_l']).ema_indicator()
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=p['rsi_p']).rsi()
        df['Vol_SMA'] = df['Volume'].rolling(window=p['ema_s']).mean()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        return df.dropna()

    def get_signal(self, row):
        p = self.params
        if (row['EMA_Short'] > row['EMA_Long'] and row['MACD'] > row['MACD_Signal'] and 
            p['rsi_min'] < row['RSI'] < p['rsi_max'] and row['Volume'] > row['Vol_SMA']):
            return 'BUY 🟢'
        if (row['EMA_Short'] < row['EMA_Long'] or (row['MACD'] < row['MACD_Signal'] and row['RSI'] < 45)):
            return 'SELL/EXIT 🔴'
        return 'HOLD ⚪'

# --- Interactive Plotting ---

def plot_interactive_chart(df, ticker, sl_mult=1.5, tp_mult=3.0):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Daily Candles & EMAs", "RSI (14)")
    )

    # 1. Price Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Price'
    ), row=1, col=1)

    # 2. EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Short'], line=dict(color='yellow', width=1.5), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Long'], line=dict(color='cyan', width=1.5), name='EMA 50'), row=1, col=1)

    # 3. RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=30, r=30, t=40, b=30)
    )
    return fig

# --- App UI & Execution ---

st.title("📈 Swing Trading Signal Generator")

# Sidebar
st.sidebar.header("Scanner Settings")
default_tickers = "AAPL, MSFT, NVDA, TSLA, AMD, META, AMZN, GOOGL, PLTR, COIN"
ticker_input = st.sidebar.text_area("Watchlist", default_tickers)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Parameters")
sl_multiplier = st.sidebar.slider("Stop Loss Multiplier (ATR)", 1.0, 3.0, 1.5, 0.1)
tp_multiplier = st.sidebar.slider("Take Profit Multiplier (ATR)", 1.0, 5.0, 3.0, 0.1)

# Preserve state so dropdowns don't wipe scanner data
if "market_data" not in st.session_state:
    st.session_state.market_data = {}
if "report_df" not in st.session_state:
    st.session_state.report_df = pd.DataFrame()

if st.sidebar.button("Run Scanner 🚀") or st.session_state.report_df.empty:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    with st.spinner("Analyzing market data..."):
        raw_data = MarketData.fetch_batch(tickers)
        strategy = SwingStrategy()
        
        results = []
        processed_data = {}
        for ticker, df in raw_data.items():
            df = strategy.apply_indicators(df)
            if df.empty:
                continue
            processed_data[ticker] = df
            latest = df.iloc[-1]
            signal = strategy.get_signal(latest)
            
            results.append({
                'Ticker': ticker,
                'Close': f"${latest['Close']:.2f}",
                'Signal': signal,
                'Stop_Loss': f"${latest['Close'] - (latest['ATR'] * sl_multiplier):.2f}" if 'BUY' in signal else '-',
                'Target': f"${latest['Close'] + (latest['ATR'] * tp_multiplier):.2f}" if 'BUY' in signal else '-',
                'RSI': round(latest['RSI'], 1),
                '_raw_signal': signal
            })

        st.session_state.market_data = processed_data
        st.session_state.report_df = pd.DataFrame(results)

# Display Dashboard
if not st.session_state.report_df.empty:
    df_display = st.session_state.report_df
    
    # Summary Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scanned Tickers", len(df_display))
    col2.metric("BUY Alerts", len(df_display[df_display['_raw_signal'].str.contains('BUY')]))
    col3.metric("SELL Alerts", len(df_display[df_display['_raw_signal'].str.contains('SELL')]))
    col4.metric("HOLD Status", len(df_display[df_display['_raw_signal'].str.contains('HOLD')]))

    # Signal Table
    st.markdown("### Scanner Summary")
    st.dataframe(df_display.drop(columns=['_raw_signal']), use_container_width=True, hide_index=True)

    # Technical Deep-Dive Charting
    st.markdown("---")
    st.markdown("### Technical Deep Dive")
    selected_ticker = st.selectbox("Select a ticker to view chart", df_display['Ticker'].tolist())

    if selected_ticker in st.session_state.market_data:
        ticker_df = st.session_state.market_data[selected_ticker]
        fig = plot_interactive_chart(ticker_df, selected_ticker, sl_multiplier, tp_multiplier)
        st.plotly_chart(fig, use_container_width=True)
