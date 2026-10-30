text = "Large language models understand text"

tokens = text.split()
print(tokens)

vocab = {
    "<UNK>": 0,
    "Large": 1,
    "language": 2,
    "models": 3,
    "understand": 4,
    "text": 5,
}

token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
print(token_ids)
