import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv("deceptive-opinion.csv")

X = data["text"]
y = data["deceptive"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load vectorizer and model
with open("fake_review_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)

# Predict
y_pred = model.predict(X_test_tfidf)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Deceptive", "Truthful"],
    yticklabels=["Deceptive", "Truthful"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Fake Review Detection - Confusion Matrix")

plt.tight_layout()
plt.show()