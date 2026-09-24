import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from concurrent.futures import ThreadPoolExecutor

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
        df['MACD'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
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

class Screener:
    def __init__(self, strategy, sl_mult=1.5, tp_mult=3.0):
        self.strategy = strategy
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def run(self, data_dict):
        results = []
        for ticker, df in data_dict.items():
            df = self.strategy.apply_indicators(df)
            if df.empty:
                continue
            latest = df.iloc[-1]
            signal = self.strategy.get_signal(latest)
            results.append({
                'Ticker': ticker,
                'Close': f"${latest['Close']:.2f}",
                'Signal': signal,
                'Stop_Loss': f"${latest['Close'] - (latest['ATR'] * self.sl_mult):.2f}" if 'BUY' in signal else '-',
                'Target': f"${latest['Close'] + (latest['ATR'] * self.tp_mult):.2f}" if 'BUY' in signal else '-',
                'RSI': round(latest['RSI'], 1)
            })
        return pd.DataFrame(results)

# --- Streamlit Web App UI ---

st.set_page_config(page_title="Swing Trading Scanner", layout="wide")

st.title("📈 Swing Trading Signal Generator")
st.markdown("Automated multi-factor confirmation using EMA, MACD, RSI, and Volume.")

# Sidebar Controls
st.sidebar.header("Scanner Settings")
default_tickers = "AAPL, MSFT, NVDA, TSLA, AMD, META, AMZN, GOOGL, PLTR, COIN"
ticker_input = st.sidebar.text_area("Watchlist (comma separated)", default_tickers)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Management (ATR)")
sl_multiplier = st.sidebar.slider("Stop Loss Multiplier", 1.0, 3.0, 1.5, 0.1)
tp_multiplier = st.sidebar.slider("Take Profit Multiplier", 1.0, 5.0, 3.0, 0.1)

# Run Button
if st.sidebar.button("Run Scanner 🚀"):
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    with st.spinner(f"Fetching market data for {len(tickers)} tickers..."):
        market_data = MarketData.fetch_batch(tickers)
    
    with st.spinner("Calculating technical indicators..."):
        strategy = SwingStrategy()
        screener = Screener(strategy=strategy, sl_mult=sl_multiplier, tp_mult=tp_multiplier)
        report_df = screener.run(market_data)
        
    if not report_df.empty:
        st.success("Scan Complete!")
        # Display as an interactive table
        st.dataframe(report_df, use_container_width=True, hide_index=True)
    else:
        st.error("No data found for the provided tickers.")