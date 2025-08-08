import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('kalman_data_test_x.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['SP500_Adj_Close_lag_1'] = df['SP500_Adj_Close'].shift(-1)
df = df.dropna().reset_index(drop=True)
df['positive_return'] = (df['SP500_Adj_Close'] > df['SP500_Adj_Close_lag_1']).astype(int)
df['positive_expected_mean_reversion'] = (df['posterior_estimated_ou_exponent'] < 0).astype(int)

print(df['positive_return'].value_counts())
print('\n--------------------\n')
print(df['positive_expected_mean_reversion'].value_counts())
print('\n--------------------\n')
print(df[['positive_return', 'positive_expected_mean_reversion']].value_counts())