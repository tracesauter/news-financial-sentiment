import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_csv('ssm1_gbm_ou_multiplier/ssm_estimates_chunk_2_3.csv')
df['SP500_Adj_Close_1_behind'] = df['SP500_Adj_Close'].shift(1)
df['SP500_Adj_Close_1_ahead'] = df['SP500_Adj_Close'].shift(-1)
df['Date'] = pd.to_datetime(df['Date'])
df['next_date'] = df['Date'].shift(-1)
df['days_until_next_date'] = (df['next_date'] - df['Date']).dt.days
df['lag_log_return'] = np.log(df['SP500_Adj_Close']) - np.log(df['SP500_Adj_Close_1_behind'])
df['target'] = np.log(df['SP500_Adj_Close_1_ahead']) - np.log(df['SP500_Adj_Close'])
df['expected_sum_of_latent_states'] = (
    df['prior_estimated_log_fair_value'] + df['prior_estimated_ou_exponent']
)
df['expected_sum_of_latent_states_minus_log_current_fair_value'] = (
    df['expected_sum_of_latent_states'] - np.log(df['SP500_Adj_Close'])
)
df['var_sum_of_latent_states'] = (
    df['prior_variance_log_fair_value'] + df['prior_variance_ou_exponent'] + 2 * df['prior_covariance_log_fair_value_ou_exponent']
)
df['expected_e_to_sum_of_latent_states'] = np.exp(
    df['expected_sum_of_latent_states'] + 0.5 * df['var_sum_of_latent_states']
)
df['expected_ratio'] = (
    df['expected_e_to_sum_of_latent_states'] / np.exp(df['SP500_Adj_Close'])
)
df['variance_e_to_sum_of_latent_states'] = (
    (np.exp(df['var_sum_of_latent_states']) - 1) *
    np.exp(2 * df['expected_sum_of_latent_states'] + df['var_sum_of_latent_states'])
)

train_split = df[df['which_split'] == 'train'].reset_index(drop=True)
test_split = df[df['which_split'] == 'test'].reset_index(drop=True)

train_df = train_split[[
    'lag_log_return',
    'expected_sum_of_latent_states_minus_log_current_fair_value',
    'var_sum_of_latent_states',
    'expected_ratio',
    # 'variance_e_to_sum_of_latent_states',
    # 'days_until_next_date',
    'target',
]]
test_df = test_split[[
    'lag_log_return',
    'expected_sum_of_latent_states_minus_log_current_fair_value',
    'var_sum_of_latent_states',
    'expected_ratio',
    # 'variance_e_to_sum_of_latent_states',
    # 'days_until_next_date',
    'target',
]]

print(test_df.head())
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)


train_X = train_df.drop(columns=['target'])
train_y = train_df['target']
test_X = test_df.drop(columns=['target'])
test_y = test_df['target']
train_dmatrix = xgb.DMatrix(data=train_X, label=train_y)
test_dmatrix = xgb.DMatrix(data=test_X, label=test_y)
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'rmse',
    'seed': 42
}
num_rounds = 200
model = xgb.train(params, train_dmatrix, num_rounds, evals=[(test_dmatrix, 'test')], early_stopping_rounds=50)
predictions = model.predict(test_dmatrix)
mse = mean_squared_error(test_y, predictions)
correlation = np.corrcoef(test_y.to_numpy(), predictions)[0, 1]
print(f'Mean Squared Error: {mse}')
print(f'Correlation: {correlation}')
