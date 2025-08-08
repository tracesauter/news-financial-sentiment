import torch
import torch.nn as nn


class Conv1DHeadlinePreprocess(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_channels_1, output_channels_2, output_sequence_length, kernel_size, linear_out_size, tokenizer):
        super(Conv1DHeadlinePreprocess, self).__init__()
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
        
        return x