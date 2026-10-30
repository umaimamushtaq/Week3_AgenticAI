import sentencepiece as spm

training_text = """I love artificial intelligence.
Artificial intelligence is changing software engineering.
Large language models process text using tokens.
Tokenization converts text into smaller units.
Machine learning is an important field of computer science.
Natural language processing allows computers to understand text.
Python is widely used for artificial intelligence.
Students are learning about large language models."""

with open("training.txt", "w", encoding="utf-8") as file:
    file.write(training_text)

spm.SentencePieceTrainer.train(
    input="training.txt",
    model_prefix="my_tokenizer",
    vocab_size=50,
    model_type="unigram",
    character_coverage=1.0,
)

tokenizer = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
text = "Artificial intelligence is changing software engineering."

tokens = tokenizer.encode(text, out_type=str)
print("Original text:")
print(text)

print("\nTokens:")
print(tokens)

token_ids = tokenizer.encode(text, out_type=int)
print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))

decoded_text = tokenizer.decode(token_ids)
print("\nDecoded text:")
print(decoded_text)
