import pandas as pd
import re

finance_news = pd.read_csv('./kaggle_headlines_with_sentiment.csv')
sentiment_lexicon = pd.read_csv('./financial_sentiment_lexicon.csv')

sentiment_lexicon['phrase_lower'] = sentiment_lexicon['Word_or_Phrase'].str.lower()
phrase_to_score_map = dict(zip(sentiment_lexicon['phrase_lower'], sentiment_lexicon['Sentiment_Score']))

def create_optimized_regex(phrases):
    # sort phrases by length (desc) so we match phrases with multiple words first (like matching "sharp decline" and not picking up "decline")
    phrases_sorted = sorted(phrases, key=len, reverse=True)
    
    escaped_phrases = [re.escape(p) for p in phrases_sorted]
    
    # create a regex pattern with word boundaries (\b) so we match whole words/phrases.
    # we don't want to match 'bull' in 'bulletin'
    pattern = r'\b(' + '|'.join(escaped_phrases) + r')\b'
    return pattern


print("Calculating 'summed_sentiment'...")
all_phrases_pattern = create_optimized_regex(sentiment_lexicon['phrase_lower'].unique())
found_phrases = finance_news['Title'].str.findall(all_phrases_pattern, flags=re.IGNORECASE)
finance_news['summed_sentiment'] = found_phrases.apply(
    lambda matches: sum(phrase_to_score_map.get(match.lower(), 0) for match in matches)
)

print("Calculating phrase count features...")

# dictionary that maps sentiment categories to masks for rows of the sentiment lexicon that match a condition
sentiment_categories = {
    "number_of_strongly_negative_phrases": (sentiment_lexicon['Sentiment_Score'] < -0.5),
    "number_of_mildly_negative_phrases": (sentiment_lexicon['Sentiment_Score'] >= -0.5) & (sentiment_lexicon['Sentiment_Score'] < 0.0),
    "number_of_mildly_positive_phrases": (sentiment_lexicon['Sentiment_Score'] > 0.0) & (sentiment_lexicon['Sentiment_Score'] <= 0.5),
    "number_of_strongly_positive_phrases": (sentiment_lexicon['Sentiment_Score'] > 0.5)
}

for col_name, condition in sentiment_categories.items():
    print(f"  - Creating '{col_name}'")
    
    category_phrases = sentiment_lexicon.loc[condition, 'phrase_lower'].unique()
    
    if len(category_phrases) > 0:
        # create the regex for this category
        pattern_category = create_optimized_regex(category_phrases)
        finance_news[col_name] = finance_news['Title'].str.count(pattern_category, flags=re.IGNORECASE)
    else:
        finance_news[col_name] = 0

finance_news.to_csv('./kaggle_headlines_with_sentiment.csv', index=False)
