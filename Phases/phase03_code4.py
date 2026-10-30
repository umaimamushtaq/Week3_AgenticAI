text = "hello"

tokens = list(text)
print(tokens)

vocab = {"h": 0, "e": 1, "l": 2, "o": 3}

token_ids = [vocab[ch] for ch in text]
print(token_ids)
