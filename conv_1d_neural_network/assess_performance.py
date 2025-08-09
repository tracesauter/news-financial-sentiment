import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from tokenizer import normalize_and_lowercase
from conv_1d_headline_preprocess_nn_architecture import Conv1DHeadlinePreprocess
import joblib

from dataclasses import dataclass
from tqdm import tqdm

@dataclass(frozen=True)
class Config:
    BATCH_SIZE: int = 64
    EMBEDDING_DIM: int = 5
    OUTPUT_CHANNELS_1: int = 4
    OUTPUT_CHANNELS_2: int = 2
    OUTPUT_SEQUENCE_LENGTH: int = 10
    VOCAB_SIZE: int =80
    KERNEL_SIZE: int = 6
    NUM_SENTIMENTS: int = 3
    LINEAR_OUT_SIZE: int = 16
    NUM_EPOCHS: int = 200
    LEARNING_RATE: float = 3e-3


pretrained_tokenizer = joblib.load('conv_1d_neural_network/bpe_tokenizer.joblib')
kaggle_headline_data = pd.read_csv("kaggle_data/kaggle_headlines_with_sentiment_and_derived_market_features_and_targets.csv")

class MultiHeadlineSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_channels_1, output_channels_2, output_sequence_length, kernel_size, linear_out_size, tokenizer):
        super(MultiHeadlineSentimentModel, self).__init__()
        # initialize this to the pretrained preprocessing layer
        # it takes tokenized headlines, embeds them, runs through 1D convolution and
        # a set of fully connected layers to produce a vector of output per
        # headline
        self.preprocessing_layer = Conv1DHeadlinePreprocess(
            vocab_size,
            embedding_dim,
            output_channels_1,
            output_channels_2,
            output_sequence_length,
            kernel_size,
            linear_out_size,
            tokenizer=tokenizer
        )
        # relu the preprocessed output
        self.relu = nn.ReLU()
        # run the output through fully connected layers that output a single value
        # per headline
        self.fully_connected = nn.Sequential(
            nn.Linear(linear_out_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.post_processing = nn.Sequential(
            nn.Linear(11, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, tokens, finbert_sentiments, lexicon_sentiments, last_day_log_return):
        # the input mini-batch has tokens, finbert sentiments, and lexicon sentiments for
        # each headline in a specific day (there could be 1, there could be many)

        # start by preprocessing tokens
        x_tokens = self.preprocessing_layer(tokens)
        # relu the preprocessed output
        x_tokens = self.relu(x_tokens)
        # run output through fully connected layers
        x_tokens = self.fully_connected(x_tokens) # (mini-batch size, 1)

        # up to this point we have used 1D convolution and fully connected layers to 
        # process each tokenized headline. Now we'll put together features aggregated
        # to the level of the single day from which the headlines were drawn.
        # here, we create a single-row "batch" of features for the day 
        mean = x_tokens.mean()
        std = x_tokens.std(unbiased=False)
        num_finbert_zeros = (finbert_sentiments == 0).sum()
        num_finbert_ones = (finbert_sentiments == 1).sum()
        num_finbert_twos = (finbert_sentiments == 2).sum()
        lexicon_sentiment_averages = lexicon_sentiments.mean(dim=0)
        features = torch.stack([
            mean, std, last_day_log_return,
            num_finbert_zeros, num_finbert_ones, num_finbert_twos,
            *lexicon_sentiment_averages
        ]).reshape(1, -1) # shape (1, 11)
        # now we run those features through a set of fully connected layers
        # to produce a single output value for the day
        out = self.post_processing(features) # (1, 1)
        # we squeeze the output to remove both dimensions to return a single scalar
        out = out.squeeze(0).squeeze(0)
        return out

# create data splits
kaggle_headline_data['Date'] = pd.to_datetime(kaggle_headline_data['Date'])
kaggle_headline_data.sort_values(by='Date', ascending=True).reset_index()
date_20_pctile = kaggle_headline_data['Date'].quantile(0.2)
date_40_pctile = kaggle_headline_data['Date'].quantile(0.4)
date_60_pctile = kaggle_headline_data['Date'].quantile(0.6)
date_80_pctile = kaggle_headline_data['Date'].quantile(0.8)
df_split_1 = kaggle_headline_data[kaggle_headline_data['Date'] <= date_20_pctile].reset_index(drop=True)
df_split_2 = kaggle_headline_data[(kaggle_headline_data['Date'] > date_20_pctile) & (kaggle_headline_data['Date'] <= date_40_pctile)].reset_index(drop=True)
df_split_3 = kaggle_headline_data[(kaggle_headline_data['Date'] > date_40_pctile) & (kaggle_headline_data['Date'] <= date_60_pctile)].reset_index(drop=True)
df_split_4 = kaggle_headline_data[(kaggle_headline_data['Date'] > date_60_pctile) & (kaggle_headline_data['Date'] <= date_80_pctile)].reset_index(drop=True)
df_holdout = kaggle_headline_data[kaggle_headline_data['Date'] > date_80_pctile].reset_index(drop=True)

# create a column in kaggle_headline_data that indicates which split of the data the row belongs to from 1 through 5
kaggle_headline_data['split'] = 0
kaggle_headline_data.loc[kaggle_headline_data['Date'] <= date_20_pctile, 'split'] = 1
kaggle_headline_data.loc[(kaggle_headline_data['Date'] > date_20_pctile) & (kaggle_headline_data['Date'] <= date_40_pctile), 'split'] = 2
kaggle_headline_data.loc[(kaggle_headline_data['Date'] > date_40_pctile) & (kaggle_headline_data['Date'] <= date_60_pctile), 'split'] = 3
kaggle_headline_data.loc[(kaggle_headline_data['Date'] > date_60_pctile) & (kaggle_headline_data['Date'] <= date_80_pctile), 'split'] = 4
kaggle_headline_data.loc[kaggle_headline_data['Date'] > date_80_pctile, 'split'] = 5

class FinancialNewsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
        # self.data is a pandas DataFrame. set its 'Date' column to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values(by='Date', ascending=True).reset_index(drop=True)
        self.data['date_id'] = pd.factorize(self.data['Date'])[0]

    def __len__(self):
        # return the number of unique dates in the dataset
        return self.data['date_id'].nunique()

    def __getitem__(self, idx):
        # filter to the rows corresponding to the date_id
        this_date_rows = self.data[self.data['date_id'] == idx]

        headlines = this_date_rows['Title'].tolist()
        tokenized_headlines = [torch.tensor(self.tokenizer.encode(normalize_and_lowercase(headline)), dtype=torch.long) for headline in headlines]
        tokenized_padded_headlines = nn.utils.rnn.pad_sequence(
            tokenized_headlines,
            batch_first=True,
            padding_value=self.tokenizer.token_to_id['<PAD>']
        ) # (batch_size, max_length)
        finbert_sentiments = torch.tensor(
            this_date_rows['finbert_sentiment'].map({
                'neutral': 0,
                'positive': 1,
                'negative': 2
            }).values,
            dtype=torch.long
        ) # (batch_size,)
        lexicon_sentiments = torch.tensor(
            this_date_rows[['summed_sentiment', 'number_of_strongly_negative_phrases', 'number_of_mildly_negative_phrases', 'number_of_mildly_positive_phrases', 'number_of_strongly_positive_phrases']].values,
            dtype=torch.float32
        ) # (batch_size, 5)

        sp500_adj_close = this_date_rows['SP500_Adj_Close'].values[0]  # SP500_Adj_Close is the same for all rows in this date
        sp500_adj_close_1_ahead = this_date_rows['SP500_1_ahead'].values[0]  # SP500_1_ahead is the same for all rows in this date
        sp500_adj_close_1_ago = this_date_rows['SP500_1_ago'].values[0]  # SP500_1_ago is the same for all rows in this date

        last_day_log_return = torch.log(torch.tensor(sp500_adj_close, dtype=torch.float32)) - torch.log(torch.tensor(sp500_adj_close_1_ago, dtype=torch.float32)) # (1,)
        target = torch.log(torch.tensor(sp500_adj_close_1_ahead, dtype=torch.float32)) - torch.log(torch.tensor(sp500_adj_close, dtype=torch.float32)) # (1,)

        return {
            'tokenized_padded_headlines': tokenized_padded_headlines, # tensor of shape (batch_size, max_length)
            'finbert_sentiments': finbert_sentiments, # tensor of shape (batch_size,)
            'lexicon_sentiments': lexicon_sentiments, # tensor of shape (batch_size, 5)
            'last_day_log_return': last_day_log_return, # tensor of shape (1,)
            'target': target # tensor of shape (1,)
        }

all_data_dataset = FinancialNewsDataset(kaggle_headline_data, pretrained_tokenizer)
all_data_dataloader = DataLoader(
    all_data_dataset,
    batch_size=1,
    shuffle=False,
)

model1 = MultiHeadlineSentimentModel(
    vocab_size=Config.VOCAB_SIZE,
    embedding_dim=Config.EMBEDDING_DIM,
    output_channels_1=Config.OUTPUT_CHANNELS_1,
    output_channels_2=Config.OUTPUT_CHANNELS_2,
    output_sequence_length=Config.OUTPUT_SEQUENCE_LENGTH,
    kernel_size=Config.KERNEL_SIZE,
    linear_out_size=Config.LINEAR_OUT_SIZE,
    tokenizer=pretrained_tokenizer
)

model2 = MultiHeadlineSentimentModel(
    vocab_size=Config.VOCAB_SIZE,
    embedding_dim=Config.EMBEDDING_DIM,
    output_channels_1=Config.OUTPUT_CHANNELS_1,
    output_channels_2=Config.OUTPUT_CHANNELS_2,
    output_sequence_length=Config.OUTPUT_SEQUENCE_LENGTH,
    kernel_size=Config.KERNEL_SIZE,
    linear_out_size=Config.LINEAR_OUT_SIZE,
    tokenizer=pretrained_tokenizer
)

# load model1 state dict from "best_model_trained_on_splits_1_and_2.pth"
model1.load_state_dict(torch.load('conv_1d_neural_network/best_model_trained_on_splits_1_and_2.pth', weights_only=True))
# load model2 state dict from "best_model_trained_on_splits_1_2_and_3.pth"
model2.load_state_dict(torch.load('conv_1d_neural_network/best_model_trained_on_splits_1_2_and_3.pth', weights_only=True))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1.to(device)
model2.to(device)


# construct model1 predictions on all_data_dataset
model1.eval()
model2.eval()
model1_predictions = []
model2_predictions = []
target_values = []
with torch.no_grad():
    for batch in tqdm(all_data_dataloader, desc="Model 1 Predictions"):
        tokens = batch['tokenized_padded_headlines'].to(device)
        tokens = tokens.squeeze(0)  # remove the batch dimension for single sample
        finbert_sentiments = batch['finbert_sentiments'].to(device)
        finbert_sentiments = finbert_sentiments.squeeze(0)  # remove the batch dimension for single sample
        lexicon_sentiments = batch['lexicon_sentiments'].to(device)
        lexicon_sentiments = lexicon_sentiments.squeeze(0)  # remove the batch dimension for single sample
        last_day_log_return = batch['last_day_log_return'].to(device)
        last_day_log_return = last_day_log_return.squeeze(0)  # remove the batch dimension for single sample
        target = batch['target'].to(device)
        target = target.squeeze(0)

        model1_prediction = model1(tokens, finbert_sentiments, lexicon_sentiments, last_day_log_return)
        model1_predictions.append(model1_prediction.item())
        model2_prediction = model2(tokens, finbert_sentiments, lexicon_sentiments, last_day_log_return)
        model2_predictions.append(model2_prediction.item())
        target_values.append(target.item())

# convert predictions to pandas DataFrame
predictions_df = pd.DataFrame({
    'date_id': all_data_dataset.data['date_id'].unique(),
    'model1_prediction': model1_predictions,
    'model2_prediction': model2_predictions,
    'target': target_values
})

# merge in the split information from all_data_dataset
predictions_df = predictions_df.merge(
    all_data_dataset.data[['date_id', 'Date', 'split']].drop_duplicates(),
    on='date_id',
    how='left'
)

# save the predictions DataFrame to a CSV file
predictions_df.to_csv('conv_1d_neural_network/predictions.csv', index=False)

# calculate r squared for model1 and model2 predictions and correlation between predictions and target
# for each split
for split in predictions_df['split'].unique():
    split_df = predictions_df[predictions_df['split'] == split]
    model1_r_squared = 1 - ((split_df['target'] - split_df['model1_prediction']) ** 2).sum() / ((split_df['target'] - split_df['target'].mean()) ** 2).sum()
    model1_correlation = split_df['target'].corr(split_df['model1_prediction'])
    model2_r_squared = 1 - ((split_df['target'] - split_df['model2_prediction']) ** 2).sum() / ((split_df['target'] - split_df['target'].mean()) ** 2).sum()
    model2_correlation = split_df['target'].corr(split_df['model2_prediction'])
    print(f"Split {split}: Model 1 R^2 = {model1_r_squared:.4f}, Model 1 Correlation = {model1_correlation:.4f}, Model 2 R^2 = {model2_r_squared:.4f}, Model 2 Correlation = {model2_correlation:.4f}")

