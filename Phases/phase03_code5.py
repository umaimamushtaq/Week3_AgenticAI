from transformers import AutoTokenizer

text = """Artificial intelligence is transforming software engineering."""

models = ["bert-base-uncased", "gpt2"]

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print("\nModel:", model_name)
    print("Tokens:", tokenizer.tokenize(text))
    print("Number of tokens:", len(token_ids))
