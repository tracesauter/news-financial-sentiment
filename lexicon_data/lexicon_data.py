import numpy as np
import pandas as pd
import re
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
prices = pd.read_csv('kaggle_data/sp500_headlines_2008_2024.csv', parse_dates=['Date'])
lexicon = pd.read_csv('kaggle_data/financial_sentiment_lexicon.csv')
target = pd.read_csv('kaggle_data/kaggle_headlines_with_sentiment_and_derived_market_features_and_targets.csv')

# Prepare lexicon: map words to sentiment scores
lexicon = lexicon.set_index('Word_or_Phrase')['Sentiment_Score'].to_dict()

# Merge price data using date
data = prices.copy()

def headline_sentiment(headline):
    tokens = re.findall(r'\w+', str(headline).lower())
    scores = [lexicon.get(token, 0) for token in tokens]
    total = sum(scores)
    avg = np.mean(scores) if scores else 0.0
    return pd.Series([total, avg])

# Apply sentiment function row-wise and assign the result to two new columns
data[['sent_sum', 'sent_avg']] = data['Title'].apply(headline_sentiment)

grouped1 = data.groupby('Date').agg({'sent_avg': 'mean'}).reset_index()# Get target data

grouped2 = target.groupby('Date').agg({
    'SP500_1_ahead': 'mean',
    'SP500_Adj_Close': 'mean'
}).reset_index()

# Merge on 'Date'
grouped1['Date'] = pd.to_datetime(grouped1['Date'])
grouped2['Date'] = pd.to_datetime(grouped2['Date'])
grouped = pd.merge(grouped1, grouped2, on='Date', how='inner')

# Calculate target column
grouped['target'] = np.log(grouped['SP500_1_ahead']) - np.log(grouped['SP500_Adj_Close'])

grouped.to_csv('./kaggle_data/add_lexicon_sentiment_to_headlines.csv', index=False)

grouped['Date'] = pd.to_datetime(grouped['Date'])
grouped.sort_values(by='Date', ascending=True).reset_index()
date_80_pctile = grouped['Date'].quantile(0.8)
data_holdout = grouped[data['Date'] > date_80_pctile].reset_index()
grouped = grouped[grouped['Date'] <= date_80_pctile].reset_index()

features = ['sent_avg']
X = grouped[features]
y = grouped['target']

tscv = TimeSeriesSplit(n_splits=4)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.01),
    'MLP Regressor': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

for name, model in models.items():
    print(f"\n=== {name} ===")
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        print(f"Fold {fold}: MSE = {mse:.6f}, R² = {r2:.4f}")