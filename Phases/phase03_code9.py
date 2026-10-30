def tokenize(text: str) -> list[str]:
    return text.lower().split()

if __name__ == "__main__":
    text = "Agentic AI systems can use tools"
    tokens = tokenize(text)
    print(tokens)
    print("Token count:", len(tokens))
