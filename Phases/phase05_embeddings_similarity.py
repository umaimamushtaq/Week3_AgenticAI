from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
 
 
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
 
 
sentences = [
    "How do I reset my password?",
    "I forgot my password.",
    "The weather is very hot today.",
]
 
 
embeddings = model.encode(sentences)
 
 
password_similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]],
)[0][0]
 
 
weather_similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]],
)[0][0]
 
 
print(
    "Password similarity:",
    password_similarity
)
 
print(
    "Weather similarity:",
    weather_similarity
)