from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Large language models are powerful."

print("Original text:")
print(text)

tokens = tokenizer.tokenize(text)
print("\nTokens:")
print(tokens)

token_ids = tokenizer.encode(text, add_special_tokens=True)
print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))

decoded = tokenizer.decode(token_ids)
print("\nDecoded text:")
print(decoded)
