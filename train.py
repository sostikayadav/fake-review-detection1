import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Load the real dataset
# ==========================================

data = pd.read_csv("deceptive-opinion.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nLabel distribution:")
print(data["deceptive"].value_counts())

# ==========================================
# 2. Select review text and labels
# ==========================================

X = data["text"]
y = data["deceptive"]

# ==========================================
# 3. Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 4. TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF transformation completed!")

# ==========================================
# 5. Train Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# ==========================================
# 6. Make predictions
# ==========================================

y_pred = model.predict(X_test_tfidf)

# ==========================================
# 7. Evaluate model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 8. Save model
# ==========================================

with open("fake_review_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\n==========================================")
print("Model and vectorizer saved successfully!")
print("==========================================")