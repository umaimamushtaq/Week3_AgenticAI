from collections import Counter

text = "low lower lowest"

words = text.split()

vocab = []

for word in words:
    chars = list(word)
    vocab.append(chars)

print(vocab)

pairs = Counter()

for word in vocab:
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        pairs[pair] += 1

print(pairs)
