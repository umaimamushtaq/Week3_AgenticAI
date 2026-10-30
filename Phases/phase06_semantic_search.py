from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
 
 
documents = [
    "Students can register courses through the student portal.",
    "The examination office publishes examination schedules.",
    "The finance office handles tuition fee issues.",
    "The library provides digital research resources.",
]
 
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
document_embeddings = model.encode(documents)

# Test the search system using at least three different queries
queries = [
    "Where can I register my courses?",
    "When are my final exams?",
    "How can I pay my tuition fee?",
]

for query in queries:
    print(f"\nQuery: {query}")
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, document_embeddings)[0]

    results = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    for document, score in results:
        print(f"  {score:.4f} -> {document}")