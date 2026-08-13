import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("deceptive-opinion.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ============================================================
# INPUT AND TARGET
# ============================================================

X = data["text"]
y = data["deceptive"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF transformation completed!")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "SVM":
        LinearSVC()
}


# ============================================================
# MODEL EVALUATION
# ============================================================

results = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="deceptive"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="deceptive"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="deceptive"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["deceptive", "truthful"]
    )

    results[name] = {

        "Accuracy": accuracy * 100,

        "Precision": precision * 100,

        "Recall": recall * 100,

        "F1 Score": f1 * 100,

        "Confusion Matrix": cm
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)


for name, result in results.items():

    print("\nModel:", name)

    print(
        "Accuracy:",
        round(result["Accuracy"], 2),
        "%"
    )

    print(
        "Precision:",
        round(result["Precision"], 2),
        "%"
    )

    print(
        "Recall:",
        round(result["Recall"], 2),
        "%"
    )

    print(
        "F1 Score:",
        round(result["F1 Score"], 2),
        "%"
    )

    print("Confusion Matrix:")

    print(result["Confusion Matrix"])


# ============================================================
# FIND BEST MODEL
# ============================================================

best_model = max(
    results,
    key=lambda x: results[x]["Accuracy"]
)

print("\n")
print("=" * 60)

print(
    "BEST MODEL:",
    best_model
)

print(
    "BEST ACCURACY:",
    round(
        results[best_model]["Accuracy"],
        2
    ),
    "%"
)

print("=" * 60)
for name, result in results.items():

    print("\nModel:", name)

    print(
        "Accuracy:",
        round(result["Accuracy"], 2),
        "%"
    )

    print(
        "Precision:",
        round(result["Precision"], 2),
        "%"
    )

    print(
        "Recall:",
        round(result["Recall"], 2),
        "%"
    )

    print(
        "F1 Score:",
        round(result["F1 Score"], 2),
        "%"
    )

    print("Confusion Matrix:")

    print(result["Confusion Matrix"])


# ============================================================
# FIND BEST MODEL
# ============================================================

best_model = max(
    results,
    key=lambda x: results[x]["Accuracy"]
)

print("\n")
print("=" * 60)

print(
    "BEST MODEL:",
    best_model
)

print(
    "BEST ACCURACY:",
    round(
        results[best_model]["Accuracy"],
        2
    ),
    "%"
)

print("=" * 60)
for name, result in results.items():

    print("\nModel:", name)

    print(
        "Accuracy:",
        round(result["Accuracy"], 2),
        "%"
    )

    print(
        "Precision:",
        round(result["Precision"], 2),
        "%"
    )

    print(
        "Recall:",
        round(result["Recall"], 2),
        "%"
    )

    print(
        "F1 Score:",
        round(result["F1 Score"], 2),
        "%"
    )

    print("Confusion Matrix:")

    print(result["Confusion Matrix"])


# ============================================================
# FIND BEST MODEL
# ============================================================

for name, result in results.items():

    print("\nModel:", name)

    print(
        "Accuracy:",
        round(result["Accuracy"], 2),
        "%"
    )

    print(
        "Precision:",
        round(result["Precision"], 2),
        "%"
    )

    print(
        "Recall:",
        round(result["Recall"], 2),
        "%"
    )

    print(
        "F1 Score:",
        round(result["F1 Score"], 2),
        "%"
    )

    print("Confusion Matrix:")

    print(result["Confusion Matrix"])


# ============================================================
# FIND BEST MODEL
# ============================================================

best_model = max(
    results,
    key=lambda x: results[x]["Accuracy"]
)

print("\n")
print("=" * 60)

print(
    "BEST MODEL:",
    best_model
)

print(
    "BEST ACCURACY:",
    round(
        results[best_model]["Accuracy"],
        2
    ),
    "%"
)

print("=" * 60)