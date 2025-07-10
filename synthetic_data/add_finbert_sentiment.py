import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

finance_news = pd.read_csv('consolidated_data/consolidated.txt', sep="|")

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

labels = {0:'neutral', 1:'positive',2:'negative'}

X = finance_news['headline'].to_list()

sent_val = list()
for x in tqdm(X, desc="Processing headlines"):
    inputs = tokenizer(x, return_tensors="pt", padding=True)
    outputs = finbert(**inputs)[0]
   
    val = labels[np.argmax(outputs.detach().numpy())]

    sent_val.append(val)

finance_news['sentiment'] = sent_val

finance_news.to_csv('consolidated_data/consolidated_with_sentiment.txt', sep="|", index=False)