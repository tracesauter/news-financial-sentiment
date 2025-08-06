import logging
from datetime import datetime, timedelta, date
import pandas as pd
from polygon import RESTClient
from typing import Optional, List # Added List for type hinting

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

# Load API key from .env file
def read_api_key(file_path="secrets/key.txt"):
    try:
        with open(file_path, "r") as file:
            # Read the file and strip any whitespace
            api_key = file.read().strip()
        return api_key
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error reading the API key: {e}")
        return None

# Usage
API_KEY = read_api_key()

if not API_KEY:
    logging.error("Polygon API key not found. Set POLYGON_API_KEY env var or in .env file.")
    raise ValueError("Missing POLYGON_API_KEY")

# client = RESTClient(api_key=API_KEY, read_timeout_seconds=60)
client = RESTClient(api_key=API_KEY)

# --- Constants ---
YEARS_OF_HISTORY = 35
# Raw columns expected directly from the polygon client's Agg object attributes
EXPECTED_RAW_COLUMNS_FROM_API = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp', 'transactions', 'otc']
# Define how we want to rename the raw columns for the final DataFrame
FINAL_COLUMN_MAP = {
    'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
    'volume': 'volume', 'vwap': 'vwap', 'transactions': 'transactions',
}
# Define the final columns we want in our output DataFrame *before* adding ticker/return
FINAL_OUTPUT_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
# Define the output CSV filename
OUTPUT_FILENAME = "combined_adjusted_daily_data.csv"

# --- Core Function (Unchanged from previous working version) ---
def get_adjusted_daily_history(ticker: str, years_back: int = YEARS_OF_HISTORY) -> Optional[pd.DataFrame]:
    """
    Fetches adjusted daily historical OHLCV data for a given ticker from Polygon.io.
    (Docstring and function body remain the same as the previous version)
    """
    ticker = ticker.upper()
    logging.info(f"Attempting to fetch adjusted daily data for ticker: {ticker} ({years_back} years back)")
    end_date = date.today()
    start_date = end_date - timedelta(days=years_back * 365.25)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    logging.info(f"Requesting data from {start_date_str} to {end_date_str}")
    try:
        aggs_generator = client.list_aggs(
            ticker=ticker, multiplier=1, timespan="day", from_=start_date_str,
            to=end_date_str, adjusted=True, sort='asc', limit=50000
        )
        agg_list = list(aggs_generator)
        if not agg_list:
            logging.warning(f"No data found for ticker {ticker} in the specified range (API returned empty list).")
            return None
        logging.info(f"Successfully retrieved {len(agg_list)} daily bars for {ticker}.")
        df = pd.DataFrame(agg_list)
        if 'timestamp' not in df.columns:
             logging.error(f"Critical error: 'timestamp' column missing in data for {ticker}.")
             return None
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce').dt.date
        if df['date'].isnull().any():
            logging.warning(f"Found invalid timestamps for {ticker}, resulting rows will be dropped.")
            df = df.dropna(subset=['date'])
        df = df.set_index('date')
        for raw_col_name in FINAL_COLUMN_MAP.keys():
            if raw_col_name not in df.columns:
                df[raw_col_name] = pd.NA
        df = df.rename(columns=FINAL_COLUMN_MAP)
        final_cols_present = [col for col in FINAL_OUTPUT_COLUMNS if col in df.columns]
        df = df[final_cols_present]
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
        for col in numeric_cols:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')
        logging.info(f"Data processed successfully for {ticker}. Final shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"An error occurred fetching or processing data for {ticker}: {e}", exc_info=True)
        return None

# --- Main Execution ---
if __name__ == "__main__":
    tickers_to_analyze = [
        'SPY',
        'VXX'
    ]

    # List to hold individual DataFrames before concatenation
    all_ticker_dataframes: List[pd.DataFrame] = [] # More specific type hint

    print("-" * 60)
    logging.info(f"Starting historical data fetch for {len(tickers_to_analyze)} tickers...")
    print("-" * 60)

    for symbol in tickers_to_analyze:
        print(f"\nProcessing: {symbol}")
        df_data = get_adjusted_daily_history(symbol)

        if df_data is not None and not df_data.empty:
            print(f"Successfully fetched and processed {len(df_data)} data points for {symbol}.")
            print(f"Date range: {df_data.index.min()} to {df_data.index.max()}")

            # --- Add Ticker Column ---
            df_data['ticker'] = symbol

            # --- Calculate Daily Returns ---
            # Calculated here so it's included in the data appended to the list
            df_data['daily_return'] = df_data['close'].pct_change()

            # --- Append to List ---
            all_ticker_dataframes.append(df_data)

            # Optional: Print preview (can be commented out for faster runs)
            print("Data Preview (with ticker and return):")
            preview_cols = ['ticker', 'close', 'daily_return']
            if 'volume' in df_data.columns: preview_cols.insert(2, 'volume')
            print(df_data[preview_cols].tail(3))

        elif df_data is not None and df_data.empty:
             print(f"Fetched data for {symbol}, but the DataFrame is empty after processing.")
        else:
            # This branch is hit if get_adjusted_daily_history returned None
            print(f"Failed to fetch or process data for {symbol} (function returned None).")

    print("-" * 60)
    logging.info("Finished fetching individual ticker data.")
    print("-" * 60)

    # --- Combine and Save ---
    if all_ticker_dataframes:
        logging.info(f"Combining data for {len(all_ticker_dataframes)} successfully fetched tickers...")
        try:
            # Concatenate all DataFrames vertically
            combined_df = pd.concat(all_ticker_dataframes, ignore_index=False) # Keep the date index

            # Optional: Reorder columns for better readability in the CSV
            # Put 'ticker' first, then OHLCV etc., then 'daily_return'
            desired_order = ['ticker'] + FINAL_OUTPUT_COLUMNS + ['daily_return']
            # Ensure we only try to order columns that actually exist in the combined DF
            cols_present_in_combined = [col for col in desired_order if col in combined_df.columns]
            combined_df = combined_df[cols_present_in_combined]

            # Ensure the index has a name for the CSV header
            combined_df.index.name = 'date'

            logging.info(f"Saving combined data ({len(combined_df)} rows) to {OUTPUT_FILENAME}...")
            combined_df.to_csv(OUTPUT_FILENAME, index=True) # index=True keeps the 'date' column
            logging.info(f"Successfully saved combined data to {OUTPUT_FILENAME}")
            print(f"\n✅ Combined data for all successful tickers saved to: {OUTPUT_FILENAME}")

        except Exception as e:
            logging.error(f"Failed to combine or save data to CSV: {e}", exc_info=True)
            print(f"\n❌ Error combining or saving data: {e}")

    else:
        logging.warning("No data was successfully fetched for any ticker. CSV file not created.")
        print("\n⚠️ No data fetched, so no CSV file was created.")

    # The 'historical_data' dictionary is no longer populated in this version,
    # as we now collect into 'all_ticker_dataframes' and then create 'combined_df'.
    # If you needed individual DFs later, you could reconstruct the dictionary
    # from 'all_ticker_dataframes' or reload from the CSV and group by ticker.