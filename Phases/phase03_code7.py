from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokens = tokenizer.tokenize("Hello world")
token_ids = tokenizer.encode("Hello world", add_special_tokens=True)

print(tokens)
print(token_ids)
print(tokenizer.cls_token)
print(tokenizer.sep_token)
print(tokenizer.cls_token_id)
print(tokenizer.sep_token_id)
