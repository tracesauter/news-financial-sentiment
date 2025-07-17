import numpy as np
import pandas as pd
import re
from sklearn.metrics import classification_report

# Load datasets
prices = pd.read_csv('kaggle_data/sp500_headlines_2008_2024.csv', parse_dates=['Date'])
lexicon = pd.read_csv('kaggle_data/financial_sentiment_lexicon.csv')

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

# Combine all headlines into one string per row, then apply
headline_cols = [col for col in data.columns if col.lower().startswith('headline')]
data['all_headlines'] = data[headline_cols].fillna('').agg(' '.join, axis=1)

# Apply sentiment function row-wise and assign the result to two new columns
data[['sent_sum', 'sent_avg']] = data['all_headlines'].apply(headline_sentiment)

# Label: next-day return direction (up/down)
data['next_close'] = data['CP'].shift(-1)
data['target'] = (data['next_close'] > data['CP']).astype(int)

# Price features
data['return'] = data['CP'].pct_change()
data['ma5'] = data['CP'].rolling(5).mean()
data.dropna(inplace=True)

from sklearn.model_selection import train_test_split

features = ['sent_sum', 'sent_avg', 'return', 'ma5']
X = data[features]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # maintain time order
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred,zero_division=0))