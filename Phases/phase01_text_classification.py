from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

texts = [
    "I love this course",
    "This course is excellent",
    "Amazing course, really enjoyed it",
    "This course is fantastic",
    "I hate this course",
    "This course is terrible",
    "Very bad course and experience",
    "This course is horrible",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
]

# 1. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# 2. Convert text into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Train the text classification model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 4. Evaluate the model using accuracy and a classification report
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))

# 5. Generate predictions for new text
new_text = ["I really love this course"]
new_X = vectorizer.transform(new_text)
prediction = model.predict(new_X)
print("New Text:", new_text[0])
print("Prediction:", prediction[0])