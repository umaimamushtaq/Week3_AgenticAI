from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
 
 
documents = [
    "Students can register courses through the student portal.",
    "The examination office publishes examination schedules.",
    "The finance office handles tuition fee issues.",
    "The library provides digital research resources.",
]
 
 
query = "Where can I register my courses?"
 
 
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
 
 
document_embeddings = model.encode(
    documents
)
 
query_embedding = model.encode(
    [query]
)
 
 
scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]
 
 
results = sorted(
    zip(documents, scores),
    key=lambda item: item[1],
    reverse=True,
)
 
 
for document, score in results:
    print(
        f"{score:.4f} -> {document}"
    )