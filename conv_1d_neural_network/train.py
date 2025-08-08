import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from tokenizer import normalize_and_lowercase
from conv_1d_headline_preprocess_nn_architecture import Conv1DHeadlinePreprocess
import joblib

from dataclasses import dataclass
from tqdm import tqdm

torch.manual_seed(42)

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

characters_to_replace_with_dash = ['–', '—']
characters_to_replace_with_single_quote = ['`', '‘', '’', '•', '…']
characters_to_replace_with_double_quote = ['“', '”', '″']
characters_to_replace_with_delimiter = ['[', ']', '{', '}', '(', ')']
characters_to_replace_with_comparison = ['<', '>', '=', '≠', '≤', '≥']
nonstandard_characters = ['`', '¥', '®', 'ا', 'ج', 'ح', 'ع', 'ف', 'ل', 'م', 'ن', 'ي', 'अ', 'आ', 'इ', 'ए', 'औ', 'क', 'ख', 'ग', 'च', 'ज', 'ट', 'ड', 'त', 'थ', 'द', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'स', 'ह', 'ा', 'ि', 'ी', 'ो', '–', '—', '‘', '’', '“', '”', '•', '…', '″', '€', '™']

# pretreat the 'Title' data to replace the prescribed characters
for char in characters_to_replace_with_dash:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '-', regex=False)
for char in characters_to_replace_with_single_quote:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, "'", regex=False)
for char in characters_to_replace_with_double_quote:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '"', regex=False)
for char in nonstandard_characters:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<UNK>', regex=False)
for char in characters_to_replace_with_delimiter:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<DELIM>', regex=False)
for char in characters_to_replace_with_comparison:
    kaggle_headline_data['Title'] = kaggle_headline_data['Title'].str.replace(char, '<COMPAR>', regex=False)

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


def train_model(
    train_dfs,
    test_dfs,
    pretrained_nn_preprocessing_layer_weights_path,
):
    # train_dfs is a list of DataFrames. we want to supply the concatenated DataFrame to the Dataset constructor
    train_data = pd.concat(train_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)

    train_dataset = FinancialNewsDataset(train_data, pretrained_tokenizer)
    test_dataset = FinancialNewsDataset(test_data, pretrained_tokenizer)

    g = torch.Generator()
    g.manual_seed(42)

    # we will batch these in mini-batches of 1 at a time, because a single element from
    # the __getitem__ method of the FinancialNewsDataset is a single day of data, which
    # itself can have multiple headlines
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=g
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    # create the model
    model = MultiHeadlineSentimentModel(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        output_channels_1=Config.OUTPUT_CHANNELS_1,
        output_channels_2=Config.OUTPUT_CHANNELS_2,
        output_sequence_length=Config.OUTPUT_SEQUENCE_LENGTH,
        kernel_size=Config.KERNEL_SIZE,
        linear_out_size=Config.LINEAR_OUT_SIZE,
        tokenizer=pretrained_tokenizer
    )
    # load the pretrained preprocessing layer weights
    model.preprocessing_layer.load_state_dict(
        torch.load(pretrained_nn_preprocessing_layer_weights_path, weights_only=True)
    )
    
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")
    print(f"After freezing the preprocessing layer, model has {(num_params - sum(p.numel() for p in model.preprocessing_layer.parameters() if p.requires_grad)):,} trainable parameters.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # freeze the preprocessing layer
    for param in model.preprocessing_layer.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(Config.NUM_EPOCHS):
        if epoch == 5:
            # unfreeze the preprocessing layer at epoch 5
            for param in model.preprocessing_layer.parameters():
                param.requires_grad = True
            print("Unfreezing the preprocessing layer at epoch 5.")
        model.train()
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        train_batch_count = 0
        train_batch_losses = []
        for mini_batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}"):
            tokens = mini_batch['tokenized_padded_headlines'].to(device)
            tokens = tokens.squeeze(0)  # remove the batch dimension
            finbert_sentiments = mini_batch['finbert_sentiments'].to(device)
            finbert_sentiments = finbert_sentiments.squeeze(0)  # remove the batch dimension
            lexicon_sentiments = mini_batch['lexicon_sentiments'].to(device)
            lexicon_sentiments = lexicon_sentiments.squeeze(0)  # remove the batch dimension
            last_day_log_return = mini_batch['last_day_log_return'].to(device)
            last_day_log_return = last_day_log_return.squeeze(0)  # remove the batch dimension
            target = mini_batch['target'].to(device)
            target = target.squeeze(0)  # remove the batch dimension

            output = model(tokens, finbert_sentiments, lexicon_sentiments, last_day_log_return)
            loss = criterion(output, target)
            train_batch_losses += [loss]

            train_batch_count += 1
            last_item_of_batch = ((train_batch_count % Config.BATCH_SIZE) == (Config.BATCH_SIZE - 1))
            last_item_of_epoch = (train_batch_count == len(train_dataloader))
            if last_item_of_batch or last_item_of_epoch:
                train_batch_losses_tensor = torch.stack(train_batch_losses)
                batch_loss = torch.mean(train_batch_losses_tensor)
                batch_loss.backward()
                if epoch >= 5:
                    # divide the gradients for the preprocessing layer by 2 so they can learn, but
                    # not overpower the gradients of the rest of the model
                    for param in model.preprocessing_layer.parameters():
                        if param.grad is not None:
                            param.grad /= 2.0
                optimizer.step()
                train_batch_losses = []
                optimizer.zero_grad()
                if last_item_of_batch:
                    elements_in_batch = Config.BATCH_SIZE
                else:
                    elements_in_batch = len(train_batch_losses_tensor)
                epoch_train_loss += batch_loss.item() * elements_in_batch

        model.eval()
        for mini_batch in tqdm(test_dataloader, desc=f"Testing Epoch {epoch + 1}/{Config.NUM_EPOCHS}"):
            tokens = mini_batch['tokenized_padded_headlines'].to(device)
            tokens = tokens.squeeze(0)  # remove the batch dimension
            finbert_sentiments = mini_batch['finbert_sentiments'].to(device)
            finbert_sentiments = finbert_sentiments.squeeze(0)  # remove the batch dimension
            lexicon_sentiments = mini_batch['lexicon_sentiments'].to(device)
            lexicon_sentiments = lexicon_sentiments.squeeze(0)  # remove the batch dimension
            last_day_log_return = mini_batch['last_day_log_return'].to(device)
            last_day_log_return = last_day_log_return.squeeze(0)  # remove the batch dimension
            target = mini_batch['target'].to(device)
            target = target.squeeze(0)  # remove the batch dimension

            with torch.no_grad():
                output = model(tokens, finbert_sentiments, lexicon_sentiments, last_day_log_return)
                loss = criterion(output, target)
                epoch_test_loss += loss.item()
        epoch_train_loss /= len(train_dataset)
        epoch_test_loss /= len(test_dataset)

        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} - Train Loss: {epoch_train_loss:.5f}, Test Loss: {epoch_test_loss:.5f}")
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with loss: {best_loss:.5f}")


if __name__ == "__main__":
    # train the model
    train_model(
        train_dfs=[df_split_1, df_split_2],
        test_dfs=[df_split_3],
        pretrained_nn_preprocessing_layer_weights_path='conv_1d_neural_network/best_conv1d_headline_preprocess.pth'
    )
