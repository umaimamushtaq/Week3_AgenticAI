from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Playing with language models."

tokens = tokenizer.tokenize(text)
print(tokens)
