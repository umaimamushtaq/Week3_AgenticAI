import re

def clean_text(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Test the preprocessing function using multiple sentences
sentences = [
    "   I NEED HELP!!!   ",
    "Hello, World! Welcome to NLP (Week 03)...",
    "Can I reset my password??? Please email me!",
    "Too   many    spaces   and symbols: @#$%!",
]

for s in sentences:
    print("Original:", repr(s))
    print("Cleaned :", repr(clean_text(s)))
    print()