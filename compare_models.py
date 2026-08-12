import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("deceptive-opinion.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Reviews
X = data["text"]

# Labels
y = data["deceptive"]


# ==========================================
# 2. SPLIT DATA
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
# 3. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF transformation completed!")


# ==========================================
# 4. CREATE MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "SVM":
        LinearSVC()
}


# ==========================================
# 5. TRAIN AND COMPARE
# ==========================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Train
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# ==========================================
# 6. DISPLAY COMPARISON
# ==========================================

print("\n\nMODEL COMPARISON")
print("=" * 50)

for name, accuracy in results.items():

    print(
        f"{name}: {accuracy * 100:.2f}%"
    )


# ==========================================
# 7. BEST MODEL
# ==========================================

best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print(
    f"Best Accuracy: {results[best_model] * 100:.2f}%"
)