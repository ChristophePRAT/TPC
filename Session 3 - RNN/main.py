import argparse
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(
    description="Train a simple RNN for sentiment analysis."
)
parser.add_argument(
    "--embedding_dim", type=int, default=5, help="Dimension of the word embeddings."
)
parser.add_argument(
    "--hidden_dim", type=int, default=256, help="Dimension of the RNN hidden state."
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer."
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size for training."
)
parser.add_argument(
    "--num_epochs", type=int, default=10, help="Number of training epochs."
)
args = parser.parse_args()

# 1. Device Configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sentiment" in df.columns:
        df["sent"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": 0})
    else:
        df["sent"] = None

    df = df.dropna(subset=["text", "sent"]).copy()

    df["text"] = df["text"].astype(str).str.lower()
    df["text"] = df["text"].replace(r"[^a-z0-9\s]", "", regex=True)

    df["sent"] = df["sent"].astype(int)

    return df


train_df = pd.read_csv("train.csv", encoding="latin1", on_bad_lines="skip")

train_df = preprocess_data(train_df)

test_df = pd.read_csv("test.csv", encoding="latin1", on_bad_lines="skip")

test_df = preprocess_data(test_df)

x_train, y_train = train_df["text"], train_df["sent"]
x_test, y_test = test_df["text"], test_df["sent"]

print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

words = set()


def tokenize_safe(review):
    # Guard against non-string and empty values
    if review is None:
        return []
    if not isinstance(review, str):
        try:
            review = str(review)
        except Exception:
            return []
    review = review.strip()
    if review == "":
        return []
    return review.split()


for review in x_train:
    words.update(tokenize_safe(review))

for review in x_test:
    words.update(tokenize_safe(review))

word_to_idx = {word: idx + 1 for idx, word in enumerate(sorted(words))}
word_to_idx["<PAD>"] = 0
print(f"Vocabulary size: {len(word_to_idx)}")


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        # Use a learnable embedding matrix
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # Index into embedding matrix (x must be long)
        return self.weight[x]


class RNN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.Whh = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.Wxh = torch.nn.Parameter(torch.randn(embedding_dim, hidden_dim) * 0.01)
        self.Who = torch.nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)

        self.Bh = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.Bo = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        batch_size = x.size(0)
        seq_len = x.size(1)

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.tanh(
                torch.matmul(x_t, self.Wxh) + torch.matmul(h, self.Whh) + self.Bh
            )

        # Classify based on the final hidden state
        return torch.matmul(h, self.Who) + self.Bo


class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = RNN(embedding_dim, hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.rnn(embedded)
        return output


# Hyperparameters
vocab_size = len(word_to_idx)
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
output_dim = 2  # binary classification (positive vs negative)
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs


def texts_to_padded_tensor(texts, vocab):
    sequences = []
    for review in texts:
        tokens = tokenize_safe(review)
        indices = [vocab.get(w, 0) for w in tokens]
        if len(indices) == 0:
            indices = [0]
        sequences.append(torch.tensor(indices, dtype=torch.long))
    if len(sequences) == 0:
        return torch.empty((0, 0), dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded


x_train_tensor = texts_to_padded_tensor(x_train, word_to_idx)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

x_test_tensor = texts_to_padded_tensor(x_test, word_to_idx)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(
    device
)


def train():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_samples = x_train_tensor.size(0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0

        # Shuffle indices for each epoch
        perm = torch.randperm(num_samples)
        x_shuffled = x_train_tensor[perm]
        y_shuffled = y_train_tensor[perm]

        for i in range(0, num_samples, batch_size):
            x_batch = x_shuffled[i : i + batch_size].to(device)
            y_batch = y_shuffled[i : i + batch_size].to(device)

            optimizer.zero_grad()

            output = model(x_batch)
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_correct += (output.argmax(dim=1) == y_batch).sum().item()

        accuracy = total_correct / num_samples if num_samples > 0 else 0.0
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def evaluate():
    model.eval()
    with torch.no_grad():
        x_test_batch = x_test_tensor.to(device)
        y_test_batch = y_test_tensor.to(device)

        output = model(x_test_batch)
        predicted = output.argmax(dim=1)
        accuracy = (predicted == y_test_batch).float().mean().item()
        print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    if len(x_train) == 0 or len(x_test) == 0:
        print("One of the datasets is empty after preprocessing. Exiting.")
        sys.exit(1)
    train()

    evaluate()
