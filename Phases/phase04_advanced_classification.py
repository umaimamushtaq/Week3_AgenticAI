import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("dataset.csv")
texts = df["message"]
labels = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))

new_text = ["I forgot my password and cannot login"]
new_vec = vectorizer.transform(new_text)
prediction = model.predict(new_vec)
print("\nNew text:", new_text[0])
print("Prediction:", prediction[0])
