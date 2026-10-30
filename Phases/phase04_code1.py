import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_dataset(path: str):
    df = pd.read_csv(path)
    return df["message"], df["label"]

def train_classifier(path: str):
    texts, labels = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))
    return pipeline

train_classifier("dataset.csv")
