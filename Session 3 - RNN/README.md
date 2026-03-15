TPC/Session 3 - RNN/README.md#L1-200
# Session #3 - Simple RNN for Sentiment Analysis

This is the third session project: a minimal educational RNN that performs sentiment classification on CSV datasets. The training and evaluation entrypoint is `main.py` in this folder.

## Installation

To install the required dependencies for the project (using the workspace helper `uv`), run:

```bash
cd .. && uv sync
```

(That will install the dependencies declared for the workspace. Alternatively install locally with `pip install torch pandas numpy`.)

## Running the code

You can run the training + evaluation script with `uv`:

```bash
uv run main.py
```

By default the script will look for `train.csv` and `test.csv` in the same directory and will print training progress and a final test accuracy.

## Custom hyperparameters

The script accepts command line arguments. Example usage with custom hyperparameters:

```bash
usage: main.py [-h] [--embedding_dim EMBEDDING_DIM] [--hidden_dim HIDDEN_DIM]
               [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
               [--num_epochs NUM_EPOCHS]

Train a simple RNN for sentiment analysis.

options:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimension of the word embeddings (default: 5)
  --hidden_dim HIDDEN_DIM
                        Dimension of the RNN hidden state (default: 256)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 1e-4)
  --batch_size BATCH_SIZE
                        Batch size for training (manual in-memory batching)
                        (default: 128)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 10)
```

Example invocation with custom values:

```bash
uv run main.py -- --embedding_dim 10 --hidden_dim 128 --learning_rate 0.001 --batch_size 64 --num_epochs 20
```

Note: `uv run main.py --` passes the following flags to the script.

## What the script does

- Loads and preprocesses `train.csv` and `test.csv`.
- Builds a vocabulary from train + test texts.
- Converts datasets into padded integer tensors (token indices).
- Trains a small custom RNN (learned embedding matrix + manual RNN loop) using in-memory batching.
- Evaluates and prints test accuracy.

## Example workflow

1. Go to the `Session 3 - RNN` directory.
2. From the session directory run:
```bash
uv run main.py -- --num_epochs 5
```
3. Monitor printed training loss/accuracy and final test accuracy.

## Further reading

- See Session #2 README for a similar CLI documentation style.
- Consider reading PyTorch docs for `nn.Embedding`, `nn.LSTM`, and tokenization best practices.
