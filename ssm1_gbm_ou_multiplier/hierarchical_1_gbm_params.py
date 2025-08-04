import pandas as pd
import numpy as np
import statsmodels.api as sm

kaggle_headline_data = pd.read_csv('kaggle_data/kaggle_headlines_with_sentiment_and_derived_market_features_and_targets.csv')
synthetic_data = pd.read_csv("synthetic_data/consolidated_data/consolidated_with_sentiment.txt", sep="|")
print(kaggle_headline_data.head())
print(kaggle_headline_data.columns)
characters_to_replace_with_dash = ['–', '—']
characters_to_replace_with_single_quote = ['`', '‘', '’', '•', '…']
characters_to_replace_with_double_quote = ['“', '”', '″']
characters_to_replace_with_delimiter = ['[', ']', '{', '}', '(', ')']
characters_to_replace_with_comparison = ['<', '>', '=', '≠', '≤', '≥']
nonstandard_characters = ['`', '¥', '®', 'ا', 'ج', 'ح', 'ع', 'ف', 'ل', 'م', 'ن', 'ي', 'अ', 'आ', 'इ', 'ए', 'औ', 'क', 'ख', 'ग', 'च', 'ज', 'ट', 'ड', 'त', 'थ', 'द', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'स', 'ह', 'ा', 'ि', 'ी', 'ो', '–', '—', '‘', '’', '“', '”', '•', '…', '″', '€', '™']

# pretreat the 'Title' data to replace the prescribed characters
for char in characters_to_replace_with_dash:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '-', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '-', regex=False)
for char in characters_to_replace_with_single_quote:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, "'", regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, "'", regex=False)
for char in characters_to_replace_with_double_quote:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '"', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '"', regex=False)
for char in nonstandard_characters:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<UNK>', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '<UNK>', regex=False)
for char in characters_to_replace_with_delimiter:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<DELIM>', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '<DELIM>', regex=False)
for char in characters_to_replace_with_comparison:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<COMPAR>', regex=False)
    synthetic_data['headline'] = synthetic_data['headline'].str.replace(char, '<COMPAR>', regex=False)

kaggle_headline_data['Date'] = pd.to_datetime(kaggle_headline_data['Date'])
bad_nans = kaggle_headline_data[[
    'SP500_Adj_Close',
    'SP500_1_ago',
    'SP500_2_ago',
    'SP500_3_ago',
    'SP500_1_ahead',
    'SP500_2_ahead',
    'SP500_3_ahead',
    'SP500_1_day_return_lag_1',
    'SP500_1_day_return_lag_2',
    'SP500_1_day_return_lag_3',
]].isna().any(axis=1)
kaggle_headline_data = kaggle_headline_data[~bad_nans].reset_index(drop=True)

kaggle_headline_data = kaggle_headline_data.sort_values(by='Date', ascending=True).reset_index(drop=True)

date_56_pctile = kaggle_headline_data['Date'].quantile(0.56)
date_80_pctile = kaggle_headline_data['Date'].quantile(0.80)

holdout_data = kaggle_headline_data[kaggle_headline_data['Date'] > date_80_pctile].reset_index(drop=True)
test_data = kaggle_headline_data[(kaggle_headline_data['Date'] > date_56_pctile) & (kaggle_headline_data['Date'] <= date_80_pctile)].reset_index(drop=True)
train_data = kaggle_headline_data[kaggle_headline_data['Date'] <= date_56_pctile].reset_index(drop=True)

train_x = train_data[[
    'Date',
    'Title',
    'SP500_Adj_Close',
    'SP500_1_day_return_forward',
]]
test_x = test_data[[
    'Date',
    'Title',
    'SP500_Adj_Close',
    'SP500_1_day_return_forward',
]]

train_x = train_x[['Date','SP500_Adj_Close','SP500_1_day_return_forward']].copy().drop_duplicates().reset_index(drop=True)
test_x = test_x[['Date','SP500_Adj_Close','SP500_1_day_return_forward']].copy().drop_duplicates().reset_index(drop=True)

train_x['days_elapsed'] = (train_x['Date'] - train_x['Date'].min()).dt.days

# regress SP500_Adj_Close on days_elapsed
X_train = sm.add_constant(train_x['days_elapsed'])
y_train = train_x['SP500_Adj_Close']
log_y_train = np.log(y_train)
model = sm.OLS(log_y_train, X_train).fit()
train_x['log_SP500_Adj_Close_predicted'] = model.predict(X_train)
# print coefficients and intercept
print("Coefficients:", model.params)
print("Intercept:", model.params['const'])
# print variance of residuals
train_x['residuals'] = np.log(y_train) - train_x['log_SP500_Adj_Close_predicted']
train_x['Delta_t'] = train_x['days_elapsed'].diff().fillna(0)
train_x['square_root_Delta_t'] = train_x['Delta_t'].apply(lambda x: x**0.5 if x >= 0 else 0)
train_x['residuals_scaled'] = train_x['residuals'] / train_x['square_root_Delta_t'].replace(0, 1)  # avoid division by zero

# print dayes_elapsed coefficient from model
print("Days elapsed coefficient:", model.params['days_elapsed'])
# Delta ln(X_t) \sim N((mu - (1/2)sigma^2)Delta t, sigma^2 Delta t)
sigma = train_x[train_x['Delta_t'] != 0]['residuals_scaled'].std()
mu = model.params['days_elapsed'] + 0.5 * sigma**2
print("Estimated mu:", mu)
print("Estimated sigma:", sigma)
