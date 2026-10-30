from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Large language models are powerful."

tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

decoded = tokenizer.decode(token_ids)
print("Decoded text:", decoded)

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bert_count = len(tokenizer.encode(text, add_special_tokens=False))
gpt2_count = len(gpt2_tokenizer.encode(text, add_special_tokens=False))
print(f"\nToken count comparison on '{text}':")
print("BERT:", bert_count)
print("GPT-2:", gpt2_count)
