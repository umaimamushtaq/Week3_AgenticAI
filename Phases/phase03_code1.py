from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Large language models are powerful."

tokens = tokenizer.tokenize(text)
print("Tokens:")
print(tokens)

token_ids = tokenizer.encode(text)
print("Token IDs:")
print(token_ids)
