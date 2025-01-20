import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys
from pathlib import Path


SYMBOLS = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "TSLA",  # Tesla, Inc.
    "ETH-USD",  # Ethereum (Cryptocurrency)
    "GOOG",  # Alphabet Inc. (Class C)
    "AMZN",  # Amazon.com, Inc.
    "META",  # Meta Platforms, Inc. (formerly Facebook)
    "NFLX",  # Netflix, Inc.
    "NVDA",  # NVIDIA Corporation
    "JPM",  # JPMorgan Chase & Co.
    "V",  # Visa Inc.
    "JNJ",  # Johnson & Johnson
    "WMT",  # Walmart Inc.
    "PG",  # The Procter & Gamble Company
    "MA",  # Mastercard Incorporated
    "DIS",  # The Walt Disney Company
    "PYPL",  # PayPal Holdings, Inc.
    "BAC",  # Bank of America Corporation
    "INTC",  # Intel Corporation
    "CMCSA",  # Comcast Corporation
    "ADBE",  # Adobe Inc.
    "PFE",  # Pfizer Inc.
    "KO",  # The Coca-Cola Company
    "PEP",  # PepsiCo, Inc.
    "CSCO",  # Cisco Systems, Inc.
    "XOM",  # Exxon Mobil Corporation
    "NKE",  # NIKE, Inc.
    "MRK",  # Merck & Co., Inc.
    "ABT",  # Abbott Laboratories
    "T",  # AT&T Inc.
    "CVX",  # Chevron Corporation
    "VZ",  # Verizon Communications Inc.
    "COST",  # Costco Wholesale Corporation
    "DHR",  # Danaher Corporation
    "MCD",  # McDonald's Corporation
    "MDT",  # Medtronic plc
    "ACN",  # Accenture plc
    "AVGO",  # Broadcom Inc.
    "QCOM",  # QUALCOMM Incorporated
    "TXN",  # Texas Instruments Incorporated
    "UNH",  # UnitedHealth Group Incorporated
    "LLY",  # Eli Lilly and Company
    "NEE",  # NextEra Energy, Inc.
    "BMY",  # Bristol Myers Squibb Company
    "ORCL",  # Oracle Corporation
    "SBUX",  # Starbucks Corporation
    "IBM",  # International Business Machines Corporation
    "MMM",  # 3M Company
    "GE",  # General Electric Company
    "F",  # Ford Motor Company
]

START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
INTERVAL = "1d"

RESULTS_CSV = "poc_results.csv"
DATA_DIR = Path("data")  # Directory to save processed data

# -----------------------------
# DATA FETCHING FUNCTION
# -----------------------------
def fetch_data(symbols, start, end, interval="1d"):
    """
    Fetch OHLC data from yfinance for each symbol
    and return a dict of {symbol: dataframe}.
    """
    data_dict = {}
    for sym in symbols:
        print(f"Fetching data for {sym}...")
        # Use group_by="column" to avoid multi-index columns
        df = yf.download(sym, start=start, end=end, interval=interval, group_by="column")

        if df.empty:
            print(f"Warning: No data fetched for {sym}. Skipping.")
            continue

        # 1) Flatten if multiindex
        if isinstance(df.columns, pd.MultiIndex):
            print(f"Info: Flattening MultiIndex columns for {sym}.")
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            print(f"Columns after flattening: {df.columns.tolist()}")

        # 2) Rename columns e.g. 'Close_AAPL' -> 'Close'
        renamed_columns = {}
        suffix = f"_{sym}"
        for col in df.columns:
            if col.endswith(suffix):
                new_col = col.replace(suffix, "")
                renamed_columns[col] = new_col

        if renamed_columns:
            df.rename(columns=renamed_columns, inplace=True)
            print(f"Columns after renaming for {sym}: {df.columns.tolist()}")

        # 3) Check for required columns (excluding Adj Close)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns {missing_cols} for {sym}. Skipping.")
            continue

        df.dropna(inplace=True)
        data_dict[sym] = df
    return data_dict

# -----------------------------
# BOLLINGER BAND CALCULATION
# -----------------------------
def add_bollinger_bands(df, window=20, num_std=2):
    """
    Add Bollinger Bands columns to the DataFrame:
      - 'SMA': Simple Moving Average
      - 'UpperBand'
      - 'LowerBand'
    Drops rows until the rolling window is available.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column to compute Bollinger Bands.")

    df['SMA'] = df['Close'].rolling(window).mean()
    df['StdDev'] = df['Close'].rolling(window).std()
    df['UpperBand'] = df['SMA'] + (num_std * df['StdDev'])
    df['LowerBand'] = df['SMA'] - (num_std * df['StdDev'])
    df.dropna(inplace=True)
    return df


def bollinger_reversal_backtest(df, symbol):
    """
    Bollinger Reversal Strategy Backtest:
      - BUY if price < 0.97 * LowerBand
      - SELL if price >= UpperBand
    """
    trades = []
    position = False
    buy_price = None
    buy_date = None

    for idx, row in df.iterrows():
        close = row['Close']
        lower = row['LowerBand']
        upper = row['UpperBand']

        if not position and close < 0.97 * lower:
            position = True
            buy_price = close
            buy_date = idx
            print(f"[{symbol}] BUY at {buy_price} on {buy_date}")

        elif position and close >= upper:
            profit_pct = (close - buy_price) / buy_price * 100
            trades.append({
                'symbol': symbol,
                'date_in': buy_date,
                'buy_price': buy_price,
                'date_out': idx,
                'sell_price': close,
                'profit_percentage': profit_pct
            })
            print(f"[{symbol}] SELL at {close} on {idx} | Profit: {profit_pct:.2f}%")
            position = False
            buy_price = None
            buy_date = None

    # Close any open position at the end
    if position:
        last_close = df['Close'].iloc[-1]
        last_date = df.index[-1]
        profit_pct = (last_close - buy_price) / buy_price * 100
        trades.append({
            'symbol': symbol,
            'date_in': buy_date,
            'buy_price': buy_price,
            'date_out': last_date,
            'sell_price': last_close,
            'profit_percentage': profit_pct
        })
        print(f"[{symbol}] SELL (End) at {last_close} on {last_date} | Profit: {profit_pct:.2f}%")
    return trades


def run_backtest(symbols, start_date, end_date, interval="1d"):
    """
    1. Fetch data
    2. Calculate Bollinger
    3. Run strategy
    4. Save trades
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data_dict = fetch_data(symbols, start_date, end_date, interval)
    all_trades = []

    for sym, df in data_dict.items():
        try:
            df_bb = add_bollinger_bands(df)
            print(f"\n{sym} Bollinger Bands DataFrame Head:")
            print(df_bb.head())

            # Save processed data in data/{symbol}/data.csv
            symbol_folder = DATA_DIR / sym
            symbol_folder.mkdir(parents=True, exist_ok=True)
            symbol_path = symbol_folder / "data.csv"
            df_bb.to_csv(symbol_path)
            print(f"Processed data for {sym} saved to {symbol_path}")

            # Run the strategy
            trades = bollinger_reversal_backtest(df_bb, sym)
            all_trades.extend(trades)
        except Exception as e:
            print(f"Error processing {sym}: {e}")
            continue

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df['date_in'] = pd.to_datetime(trades_df['date_in'])
        trades_df['date_out'] = pd.to_datetime(trades_df['date_out'])
        trades_df.sort_values(by='date_in', inplace=True)
    else:
        print("No trades were generated during the backtest.")

    trades_df.to_csv(RESULTS_CSV, index=False)
    print(f"All trades saved to {RESULTS_CSV}")
    return trades_df


def show_streamlit_app(trades_df, data_dict):
    """
    Minimal Streamlit web app to:
      - Show trades
      - Let user select a symbol to plot
      - Display Bollinger chart with buy/sell markers
    """
    st.title("POC Bollinger Reversal Backtest")
    if trades_df.empty:
        st.warning("No trades generated. Please check the backtest.")
        return
    else:
        st.subheader("Trade Results")
        st.dataframe(trades_df.reset_index(drop=True))

    unique_symbols = trades_df['symbol'].unique()
    selected_symbol = st.selectbox("Select a symbol to visualize", unique_symbols)

    if selected_symbol:
        if selected_symbol not in data_dict:
            st.error(f"No data available for {selected_symbol}.")
            return

        df = data_dict[selected_symbol].copy()
        df_bb = add_bollinger_bands(df)

        # Identify signals
        buy_signals, sell_signals = [], []
        position = False
        for idx, row in df_bb.iterrows():
            close = row['Close']
            lower = row['LowerBand']
            upper = row['UpperBand']
            if not position and close < 0.97 * lower:
                buy_signals.append((idx, close))
                position = True
            elif position and close >= upper:
                sell_signals.append((idx, close))
                position = False

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_bb.index, df_bb['Close'], label='Close', color='blue')
        ax.plot(df_bb.index, df_bb['UpperBand'], color='red', label='UpperBand')
        ax.plot(df_bb.index, df_bb['LowerBand'], color='green', label='LowerBand')
        ax.plot(df_bb.index, df_bb['SMA'], color='orange', label='SMA', linestyle='--')
        if buy_signals:
            b_dates, b_prices = zip(*buy_signals)
            ax.scatter(b_dates, b_prices, marker='^', color='green', s=100, label='Buy')
        if sell_signals:
            s_dates, s_prices = zip(*sell_signals)
            ax.scatter(s_dates, s_prices, marker='v', color='red', s=100, label='Sell')
        ax.set_title(f"{selected_symbol} Bollinger Bands")
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "backtest":
        print("Running backtest...")
        trades_df = run_backtest(SYMBOLS, START_DATE, END_DATE, INTERVAL)
        if not trades_df.empty:
            print("\nBacktest Trades:")
            print(trades_df)
        else:
            print("No trades were generated during the backtest.")
    else:
        if not Path(RESULTS_CSV).exists():
            st.warning("No existing results found. Running backtest now...")
            trades_df = run_backtest(SYMBOLS, START_DATE, END_DATE, INTERVAL)
            if trades_df.empty:
                st.warning("Backtest completed but no trades were generated.")
        else:
            try:
                trades_df = pd.read_csv(RESULTS_CSV)
                trades_df['date_in'] = pd.to_datetime(trades_df['date_in'])
                trades_df['date_out'] = pd.to_datetime(trades_df['date_out'])
                st.success(f"Loaded backtest results from {RESULTS_CSV}")
            except Exception as e:
                st.error(f"Error loading {RESULTS_CSV}: {e}")
                st.stop()

        # Re-fetch or load processed data for visualization
        data_dict = {}
        for sym in SYMBOLS:
            symbol_folder = DATA_DIR / sym
            symbol_data_path = symbol_folder / "data.csv"
            if symbol_data_path.exists():
                df_sym = pd.read_csv(symbol_data_path, index_col=0, parse_dates=True)
                data_dict[sym] = df_sym
            else:
                # If data doesn't exist, attempt to fetch again
                df_fetch = yf.download(sym, start=START_DATE, end=END_DATE, interval=INTERVAL, group_by="column")
                if not df_fetch.empty:
                    # Flatten / rename
                    if isinstance(df_fetch.columns, pd.MultiIndex):
                        df_fetch.columns = ['_'.join(col).strip() for col in df_fetch.columns.values]
                    suffix = f"_{sym}"
                    rename_map = {}
                    for col in df_fetch.columns:
                        if col.endswith(suffix):
                            rename_map[col] = col.replace(suffix, "")
                    if rename_map:
                        df_fetch.rename(columns=rename_map, inplace=True)
                    # Add Bollinger
                    df_fetch.dropna(inplace=True)
                    data_dict[sym] = add_bollinger_bands(df_fetch)
                else:
                    st.warning(f"No data fetched for {sym}, skipping in visualization.")

        if not data_dict:
            st.error("No data to visualize.")
            st.stop()


        show_streamlit_app(trades_df, data_dict)
