from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

texts = [
    "I love this product",
    "This product is excellent",
    "Amazing experience",
    "I really enjoyed this",
    "This is fantastic",
    "I hate this product",
    "This product is terrible",
    "Very bad experience",
    "I did not enjoy this",
    "This is horrible",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vectors, y_train)

predictions = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

new_text = ["I really love this"]
new_vectors = vectorizer.transform(new_text)
prediction = model.predict(new_vectors)

print("\nNew text:", new_text[0])
print("Prediction:", prediction[0])
