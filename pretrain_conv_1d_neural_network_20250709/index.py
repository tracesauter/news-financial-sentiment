import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from tokenizer import BPETokenizer, normalize_and_lowercase
from sklearn.model_selection import train_test_split

import random
from functools import partial

from dataclasses import dataclass
from tqdm import tqdm

@dataclass(frozen=True)
class Config:
    BATCH_SIZE: int = 64
    EMBEDDING_DIM: int = 5
    OUTPUT_CHANNELS_1: int = 4
    OUTPUT_CHANNELS_2: int = 2
    OUTPUT_SEQUENCE_LENGTH: int = 10
    VOCAB_SIZE=80
    KERNEL_SIZE: int = 6
    NUM_SENTIMENTS: int = 3
    LINEAR_OUT_SIZE: int = 16
    NUM_EPOCHS: int = 200
    LEARNING_RATE: float = 3e-3

kaggle_headline_data = pd.read_csv("kaggle_data/sp500_headlines_2008_2024.csv")
financial_sentiment_lexicon = pd.read_csv("kaggle_data/financial_sentiment_lexicon.csv")
synthetic_data = pd.read_csv("synthetic_data/consolidated_data/consolidated_with_sentiment.txt", sep="|")

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

# use train_test_split to set aside holdout data from kaggle headline data
non_holdout_data, holdout_data = train_test_split(kaggle_headline_data, test_size=0.2, random_state=42)
# use train_test_split to spliit non_holdout_data into train and test sets
train_data, test_data = train_test_split(non_holdout_data, test_size=0.3, random_state=42)

# set corpus to a list of headlines from train_data
corpus = train_data['Title'].tolist()

# replace every time there are two double quotes in a row with a single double quote
corpus = [text.replace('""', '"') for text in corpus]

special_tokens = ["<PAD>", "<UNK>", "<MASK>","<DELIM>","<COMPAR>"]

# # take the top 10 and bottom 10 "Word_or_Phrase" values from the financial sentiment lexicon and add them to the special tokens
# special_tokens += financial_sentiment_lexicon['Word_or_Phrase'].tolist()[:4] + financial_sentiment_lexicon['Word_or_Phrase'].tolist()[-4:]
vocab_size = Config.VOCAB_SIZE # The total size of the vocabulary (special + chars + merges)

# initialize and Train the Tokenizer
tokenizer = BPETokenizer(special_tokens=special_tokens)
tokenizer.train(corpus, vocab_size)
print(f"Tokenizer trained with vocabulary size: {len(tokenizer.vocab)}")

synthetic_train_data, synthetic_test_data = train_test_split(synthetic_data, test_size=0.3, random_state=42, stratify=synthetic_data['sentiment'])

class FinancialNewsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headline = self.data.iloc[idx]['headline']
        sentiment = self.data.iloc[idx]['sentiment']
        
        # Normalize and tokenize the headline and add padding tokens to the right up to max_character_length
        headline = normalize_and_lowercase(headline)
        tokens = self.tokenizer.encode(headline)
        # padded_tokens = tokens + [self.tokenizer.token_to_id['<PAD>']] * (max_character_length - len(tokens))
        torch_tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Convert sentiment to a numerical label
        sentiment_label = {'neutral': 0, 'positive': 1, 'negative': 2}[sentiment]
        torch_sentiment_label = torch.tensor(sentiment_label, dtype=torch.long)

        return torch_tokens, torch_sentiment_label
    
def collate_and_mask(batch, pad_token_id, mask_token_id):
    # Separate tokens and sentiments
    token_lists, sentiments = zip(*batch)

    # Pad the token sequences to the length of the longest sequence in the batch
    padded_tokens = nn.utils.rnn.pad_sequence(
        token_lists, batch_first=True, padding_value=pad_token_id
    )

    tokens_with_masking = padded_tokens.clone()
    masked_tokens = torch.zeros(padded_tokens.shape[0], dtype=torch.long)

    # Iterate over each sequence in the batch to apply a mask
    for i in range(padded_tokens.size(0)):
        # Don't mask padding tokens. Find the actual length of the sequence.
        seq_len = (padded_tokens[i] != pad_token_id).sum().item()
        
        # Randomly select an index to mask (from 0 to seq_len - 1)
        mask_idx = random.randint(0, seq_len - 1)

        # Mask the token at the selected index
        masked_tokens[i] = tokens_with_masking[i, mask_idx]
        tokens_with_masking[i, mask_idx] = mask_token_id

    return {
        'tokens': padded_tokens,
        'tokens_with_masking': tokens_with_masking,
        'masked_tokens_ids': masked_tokens,
        'sentiments': torch.tensor(sentiments, dtype=torch.long)
    }

collate_function_with_args = partial(
    collate_and_mask,
    pad_token_id=tokenizer.token_to_id['<PAD>'],
    mask_token_id=tokenizer.token_to_id['<MASK>']
)

# Create datasets and dataloaders
train_dataset = FinancialNewsDataset(synthetic_train_data, tokenizer)
test_dataset = FinancialNewsDataset(synthetic_test_data, tokenizer)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_function_with_args
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_function_with_args
)

class Conv1DHeadlineModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_channels_1, output_channels_2, output_sequence_length, kernel_size, num_sentiments, linear_out_size):
        super(Conv1DHeadlineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.token_to_id['<PAD>'])
        self.conv1d_1 = nn.Conv1d(embedding_dim, output_channels_1, kernel_size=kernel_size, padding=1)
        self.conv1d_2 = nn.Conv1d(output_channels_1, output_channels_2, kernel_size=kernel_size, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_sequence_length)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Sequential(
            nn.Linear(output_channels_2 * output_sequence_length, 2 * linear_out_size),
            nn.ReLU(),
            nn.Linear(2 * linear_out_size, linear_out_size),
            nn.ReLU(),
            nn.Linear(linear_out_size, linear_out_size)
        )
        self.token_out = nn.Sequential(
            nn.Linear(linear_out_size, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, vocab_size)  # Final layer to output vocab size
        )
        self.sentiment_out = nn.Sequential(
            nn.Linear(linear_out_size, 2 * num_sentiments),
            nn.ReLU(),
            nn.Linear(2 * num_sentiments, 2 * num_sentiments),
            nn.ReLU(),
            nn.Linear(2 * num_sentiments, num_sentiments),
        )

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # Shape: (batch_size, embedding_dim, seq_len)
        x = self.conv1d_1(x)  # Shape: (batch_size, output_channels_1, seq_len)
        x = self.relu(x)
        x = self.conv1d_2(x)  # Shape: (batch_size, output_channels_2, seq_len)
        x = self.relu(x)
        x = self.adaptive_pool(x)  # Shape: (batch_size, output_channels_2, output_sequence_length)
        x = x.transpose(1, 2)  # Shape: (batch_size, output_sequence_length, output_channels_2)
        # Flatten the output for the linear layer
        x = x.reshape(x.size(0), -1)  # Shape: (batch_size, output_channels_2 * output_sequence_length)
        x = self.linear(x)  # Shape: (batch_size, linear_out_size)
        x = self.relu(x)
        token_out = self.token_out(x)  # Shape: (batch_size, vocab_size)
        sentiment_out = self.sentiment_out(x)  # Shape: (batch_size, num_sentiments)
        return token_out, sentiment_out

model = Conv1DHeadlineModel(
    vocab_size=len(tokenizer.vocab),
    embedding_dim=Config.EMBEDDING_DIM,
    output_channels_1=Config.OUTPUT_CHANNELS_1,
    output_channels_2=Config.OUTPUT_CHANNELS_2,
    output_sequence_length=Config.OUTPUT_SEQUENCE_LENGTH,
    kernel_size=Config.KERNEL_SIZE,
    num_sentiments=Config.NUM_SENTIMENTS,
    linear_out_size=Config.LINEAR_OUT_SIZE
)

# print number of model parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params} trainable parameters.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to the selected device
model.to(device)

# Define the loss functions for our two tasks
# We use CrossEntropyLoss because this is a multi-class classification problem
# for both token prediction and sentiment analysis.
loss_fn_token = nn.CrossEntropyLoss()
loss_fn_sentiment = nn.CrossEntropyLoss()

# Define the optimizer
# Adam is a good default choice for many deep learning tasks.
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

def train_model(model, train_loader, test_loader, optimizer, loss_fn_token, loss_fn_sentiment, device, epochs=10):
    """
    Main function to train and evaluate the model.
    """
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()  # Set the model to training mode
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training Batches"):
            # Move batch data to the device
            tokens_masked = batch['tokens_with_masking'].to(device)
            true_masked_tokens = batch['masked_tokens_ids'].to(device)
            sentiments = batch['sentiments'].to(device)

            # 1. Forward pass
            token_logits, sentiment_logits = model(tokens_masked)

            # 2. Calculate loss
            # token_multiplier = 3.0 if epoch < 15 else 2.0 if epoch < 30 else 1.95 if epoch < 40 else 1.5 if epoch < 150 else 1.25
            token_multiplier = 3.4 if epoch < 10 else 5.0 if epoch < 30 else 1.5
            loss_token = loss_fn_token(token_logits, true_masked_tokens) * token_multiplier
            # sentiment_multiplier = 0.4 if epoch < 5 else 0.7 if epoch < 15 else 0.9 if epoch < 30 else 1.05 if epoch < 50 else 1.25
            sentiment_multiplier = 1.7 if epoch < 5 else 1.25 if epoch < 15 else 1.0
            loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiments) * sentiment_multiplier
            
            # We combine the losses. A simple sum is a good starting point.
            # You could also weight them, e.g., total_loss = 0.7 * loss_token + 0.3 * loss_sentiment
            total_loss = loss_token + loss_sentiment
            
            # 3. Backward pass and optimization
            optimizer.zero_grad()    # Clear previous gradients
            total_loss.backward()    # Compute gradients
            optimizer.step()         # Update model weights

            total_train_loss += total_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        correct_tokens = 0
        second_choice_correct = 0
        correct_sentiments = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient calculation for efficiency
            for batch in test_loader:
                tokens_masked = batch['tokens_with_masking'].to(device)
                true_masked_tokens = batch['masked_tokens_ids'].to(device)
                sentiments = batch['sentiments'].to(device)

                token_logits, sentiment_logits = model(tokens_masked)
                
                loss_token = loss_fn_token(token_logits, true_masked_tokens)
                loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiments)
                total_val_loss += (loss_token + loss_sentiment).item()

                # Calculate accuracy for token prediction
                pred_tokens = torch.argmax(token_logits, dim=1)
                second_best_tokens = torch.topk(token_logits, 2, dim=1).indices[:, 1]
                correct_tokens += (pred_tokens == true_masked_tokens).sum().item()
                second_choice_correct += (second_best_tokens == true_masked_tokens).sum().item()

                # Calculate accuracy for sentiment prediction
                pred_sentiments = torch.argmax(sentiment_logits, dim=1)
                correct_sentiments += (pred_sentiments == sentiments).sum().item()
                
                total_samples += sentiments.size(0)

        avg_val_loss = total_val_loss / len(test_loader)
        token_accuracy = correct_tokens / total_samples
        token_top_two_accuracy = (correct_tokens + second_choice_correct) / total_samples
        sentiment_accuracy = correct_sentiments / total_samples

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Token Accuracy:     {token_accuracy:.4f}")
        print(f"  Val Token Top-2 Accuracy: {token_top_two_accuracy:.4f}")
        print(f"  Val Sentiment Accuracy: {sentiment_accuracy:.4f}")
        print("-" * 30)

# Start training
if __name__ == '__main__':
    train_model(
        model=model,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        optimizer=optimizer,
        loss_fn_token=loss_fn_token,
        loss_fn_sentiment=loss_fn_sentiment,
        device=device,
        epochs=Config.NUM_EPOCHS
    )

# # 3. Inspect the results
# print(f"Vocabulary Size: {len(tokenizer.vocab)}")
# print(f"Learned Vocabulary: {tokenizer.vocab}\n")
# print(f"Learned Merges ({len(tokenizer.merges)}):")
# print(f"Tokenizer - ID Mapping: {tokenizer.token_to_id}\n")
# for i, (pair, new_token) in enumerate(tokenizer.merges.items()):
#     print(f"{i+1}: {pair} -> {new_token}")
# print("-" * 30)


# # 4. Test Encoding and Decoding
# text_to_test = "This is a popular deep learning algorithm with pytorch."
# print(f"Original Text: '{text_to_test}'")

# # Encode the text
# encoded_ids = tokenizer.encode(text_to_test)
# print(f"Encoded IDs: {encoded_ids}\n")

# # Decode the IDs
# decoded_text = tokenizer.decode(encoded_ids)
# print(f"Decoded Text: '{decoded_text}'")

# # Test with unknown characters and accents
# unknown_text = "résumé with a Z." # 'z' was not in the training corpus
# unknown_text = "résumé ¡Hola, cómo estás? This sentence has special characters and accents. with a Z." # 'z' was not in the training corpus
# print(f"\nOriginal Text with Unknowns: '{unknown_text}'")
# encoded_unknown = tokenizer.encode(unknown_text)
# print(f"Encoded with <UNK> (ID={tokenizer.token_to_id['<UNK>']}): {encoded_unknown}")
# decoded_unknown = tokenizer.decode(encoded_unknown)
# print(f"Decoded Text: '{decoded_unknown}'")