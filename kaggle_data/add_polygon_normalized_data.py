import pandas as pd

existing_data = pd.read_csv("kaggle_headlines_with_sentiment.csv")
polygon_data = pd.read_csv("pull_data/combined_adjusted_daily_data.csv")

existing_data['Date'] = pd.to_datetime(existing_data['Date'])

polygon_sp500 = polygon_data[polygon_data['ticker'] == 'SPY'].copy()
polygon_sp500['Date'] = pd.to_datetime(polygon_sp500['date'])
polygon_sp500 = polygon_sp500[['Date', 'close']].rename(columns={'close': 'SP500_Adj_Close'})
polygon_sp500 = polygon_sp500.sort_values(by='Date', ascending=True).reset_index(drop=True)
polygon_sp500['SP500_1_ahead'] = polygon_sp500['SP500_Adj_Close'].shift(-1)
polygon_sp500['SP500_2_ahead'] = polygon_sp500['SP500_Adj_Close'].shift(-2)
polygon_sp500['SP500_3_ahead'] = polygon_sp500['SP500_Adj_Close'].shift(-3)
polygon_sp500['SP500_1_ago'] = polygon_sp500['SP500_Adj_Close'].shift(1)
polygon_sp500['SP500_2_ago'] = polygon_sp500['SP500_Adj_Close'].shift(2)
polygon_sp500['SP500_3_ago'] = polygon_sp500['SP500_Adj_Close'].shift(3)
polygon_sp500['SP500_1_day_return_forward'] = polygon_sp500['SP500_1_ahead'] / polygon_sp500['SP500_Adj_Close'] - 1
polygon_sp500['SP500_3_day_return_forward'] = polygon_sp500['SP500_3_ahead'] / polygon_sp500['SP500_Adj_Close'] - 1
polygon_sp500['SP500_3_day_return_backward'] = polygon_sp500['SP500_Adj_Close'] / polygon_sp500['SP500_3_ago'] - 1
polygon_sp500['SP500_4_day_return_straddle'] = polygon_sp500['SP500_2_ahead'] / polygon_sp500['SP500_2_ago'] - 1
polygon_sp500['SP500_1_day_return_lag_1'] = polygon_sp500['SP500_1_day_return_forward'].shift(1)
polygon_sp500['SP500_1_day_return_lag_2'] = polygon_sp500['SP500_1_day_return_forward'].shift(2)
polygon_sp500['SP500_1_day_return_lag_3'] = polygon_sp500['SP500_1_day_return_forward'].shift(3)

polygon_VIX_etf = polygon_data[polygon_data['ticker'] == 'VXX'].copy()
polygon_VIX_etf['Date'] = pd.to_datetime(polygon_VIX_etf['date'])
polygon_VIX_etf = polygon_VIX_etf[['Date', 'close']].rename(columns={'close': 'VIX_ETF_Adj_Close'})
polygon_VIX_etf = polygon_VIX_etf.sort_values(by='Date', ascending=True).reset_index(drop=True)
polygon_VIX_etf['VIX_ETF_1_ahead'] = polygon_VIX_etf['VIX_ETF_Adj_Close'].shift(-1)
polygon_VIX_etf['VIX_ETF_2_ahead'] = polygon_VIX_etf['VIX_ETF_Adj_Close'].shift(-2)
polygon_VIX_etf['VIX_ETF_3_ahead'] = polygon_VIX_etf['VIX_ETF_Adj_Close'].shift(-3)
polygon_VIX_etf['VIX_ETF_1_ago'] = polygon_VIX_etf['VIX_ETF_Adj_Close'].shift(1)
polygon_VIX_etf['VIX_ETF_2_ago'] = polygon_VIX_etf['VIX_ETF_Adj_Close'].shift(2)
polygon_VIX_etf['VIX_ETF_3_ago'] = polygon_VIX_etf['VIX_ETF_Adj_Close'].shift(3)
polygon_VIX_etf['VIX_ETF_1_day_return_forward'] = polygon_VIX_etf['VIX_ETF_1_ahead'] / polygon_VIX_etf['VIX_ETF_Adj_Close'] - 1
polygon_VIX_etf['VIX_ETF_3_day_return_forward'] = polygon_VIX_etf['VIX_ETF_3_ahead'] / polygon_VIX_etf['VIX_ETF_Adj_Close'] - 1
polygon_VIX_etf['VIX_ETF_3_day_return_backward'] = polygon_VIX_etf['VIX_ETF_Adj_Close'] / polygon_VIX_etf['VIX_ETF_3_ago'] - 1
polygon_VIX_etf['VIX_ETF_4_day_return_straddle'] = polygon_VIX_etf['VIX_ETF_2_ahead'] / polygon_VIX_etf['VIX_ETF_2_ago'] - 1
polygon_VIX_etf['VIX_ETF_1_day_return_lag_1'] = polygon_VIX_etf['VIX_ETF_1_day_return_forward'].shift(1)
polygon_VIX_etf['VIX_ETF_1_day_return_lag_2'] = polygon_VIX_etf['VIX_ETF_1_day_return_forward'].shift(2)
polygon_VIX_etf['VIX_ETF_1_day_return_lag_3'] = polygon_VIX_etf['VIX_ETF_1_day_return_forward'].shift(3)

merged_data = (
    existing_data
    .merge(polygon_sp500, on='Date', how='left')
    .merge(polygon_VIX_etf, on='Date', how='left')
)

merged_data.to_csv("kaggle_headlines_with_sentiment_and_derived_market_features_and_targets.csv", index=False)
