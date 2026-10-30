from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = """Artificial intelligence is transforming software engineering and many other fields."""

tokens = tokenizer.encode(text, add_special_tokens=False)

print("Characters:", len(text))
print("Tokens:", len(tokens))
