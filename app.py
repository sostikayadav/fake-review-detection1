import os
import streamlit as st
import pandas as pd
import joblib

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
    confusion_matrix,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "deceptive-opinion.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fake_review_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "fake_review_model.pkl was not found."
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

@st.cache_resource
def load_vectorizer():

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "tfidf_vectorizer.pkl was not found."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)

    return vectorizer


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "deceptive-opinion.csv was not found."
        )

    df = pd.read_csv(DATASET_PATH)

    return df


# ============================================================
# EVALUATE MODELS
# ============================================================

@st.cache_data
def evaluate_models():

    df = load_dataset()

    # Check required columns
    if "text" not in df.columns:
        raise ValueError(
            "Dataset does not contain a 'text' column."
        )

    if "deceptive" not in df.columns:
        raise ValueError(
            "Dataset does not contain a 'deceptive' column."
        )

    df = df.dropna(
        subset=["text", "deceptive"]
    ).copy()

    df["text"] = df["text"].astype(str)

    df["deceptive"] = (
        df["deceptive"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    X = df["text"]
    y = df["deceptive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Naive Bayes": MultinomialNB(),

        "SVM": LinearSVC(
            random_state=42
        )
    }

    results = []
    matrices = {}

    for name, current_model in models.items():

        current_model.fit(
            X_train_tfidf,
            y_train
        )

        predictions = current_model.predict(
            X_test_tfidf
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            pos_label="deceptive",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label="deceptive",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label="deceptive",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        matrices[name] = confusion_matrix(
            y_test,
            predictions,
            labels=["deceptive", "truthful"]
        )

    results_df = pd.DataFrame(results)

    best_index = results_df["Accuracy"].idxmax()

    best_model = results_df.loc[
        best_index,
        "Model"
    ]

    best_accuracy = results_df.loc[
        best_index,
        "Accuracy"
    ]

    return (
        results_df,
        matrices,
        len(df),
        len(X_train),
        len(X_test),
        best_model,
        best_accuracy
    )


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:

    model = load_model()
    vectorizer = load_vectorizer()
    dataset = load_dataset()

except Exception as error:

    st.error(
        "Unable to load project files."
    )

    st.code(str(error))

    st.info(
        "Make sure these files are in the same folder as app.py:\n\n"
        "fake_review_model.pkl\n"
        "tfidf_vectorizer.pkl\n"
        "deceptive-opinion.csv"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "📝 Review Prediction",
        "📊 Analytics Dashboard",
        "ℹ️ About Project"
    ]
)


# ============================================================
# COMMON HEADER
# ============================================================

st.title(
    "🔎 Fake Review Detection System"
)

st.caption(
    "NLP and Machine Learning Based Deceptive Review Detection"
)

st.divider()


# ============================================================
# PAGE 1 - REVIEW PREDICTION
# ============================================================

if page == "📝 Review Prediction":

    st.header(
        "📝 Review Prediction"
    )

    st.write(
        "Enter a hotel or product review below. "
        "The machine learning model will predict "
        "whether the review is deceptive or truthful."
    )

    st.divider()

    review = st.text_area(
        "Enter your review:",
        height=180,
        placeholder=(
            "Example: The hotel room was clean and comfortable. "
            "The staff were friendly and helpful."
        )
    )

    if st.button(
        "🔎 Check Review",
        type="primary",
        use_container_width=True
    ):

        if not review.strip():

            st.warning(
                "Please enter a review first."
            )

        else:

            try:

                review_vector = vectorizer.transform(
                    [review]
                )

                prediction = model.predict(
                    review_vector
                )[0]

                prediction_text = (
                    str(prediction)
                    .strip()
                    .lower()
                )

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                if prediction_text == "deceptive":

                    st.error(
                        "⚠️ DECEPTIVE REVIEW"
                    )

                    st.write(
                        "The model classified this review "
                        "as deceptive."
                    )

                elif prediction_text == "truthful":

                    st.success(
                        "✅ TRUTHFUL REVIEW"
                    )

                    st.write(
                        "The model classified this review "
                        "as truthful."
                    )

                else:

                    st.info(
                        f"Model prediction: {prediction}"
                    )

                # --------------------------------------------
                # PREDICTION PROBABILITY
                # --------------------------------------------

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        review_vector
                    )[0]

                    classes = list(
                        model.classes_
                    )

                    probability_df = pd.DataFrame({
                        "Class": classes,
                        "Probability": probabilities
                    })

                    probability_df[
                        "Probability"
                    ] = (
                        probability_df["Probability"] * 100
                    )

                    st.subheader(
                        "📊 Prediction Confidence"
                    )

                    st.bar_chart(
                        probability_df.set_index(
                            "Class"
                        )["Probability"]
                    )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(error)
                )


# ============================================================
# PAGE 2 - ANALYTICS DASHBOARD
# ============================================================

elif page == "📊 Analytics Dashboard":

    st.header(
        "📊 Model Analytics Dashboard"
    )

    st.write(
        "Performance analysis of the machine learning "
        "models used in the project."
    )

    try:

        (
            results_df,
            matrices,
            total_samples,
            training_samples,
            testing_samples,
            best_model_name,
            best_accuracy
        ) = evaluate_models()

    except Exception as error:

        st.error(
            "Unable to evaluate the models."
        )

        st.code(
            str(error)
        )

        st.stop()

    st.divider()

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.subheader(
        "📌 Dataset & Model Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Dataset",
            f"{total_samples:,}"
        )

    with col2:

        st.metric(
            "Training Samples",
            f"{training_samples:,}"
        )

    with col3:

        st.metric(
            "Testing Samples",
            f"{testing_samples:,}"
        )

    with col4:

        st.metric(
            "Best Accuracy",
            f"{best_accuracy:.2%}"
        )

    st.divider()

    # ========================================================
    # MODEL ACCURACY
    # ========================================================

    st.subheader(
        "📈 Model Accuracy Comparison"
    )

    chart_df = results_df[
        ["Model", "Accuracy"]
    ].copy()

    chart_df["Accuracy"] = (
        chart_df["Accuracy"] * 100
    )

    chart_df = chart_df.set_index(
        "Model"
    )

    st.bar_chart(
        chart_df
    )

    st.divider()

    # ========================================================
    # PERFORMANCE TABLE
    # ========================================================

    st.subheader(
        "📋 Model Performance"
    )

    display_df = results_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader(
        "🏆 Best Model"
    )

    st.success(
        f"{best_model_name} achieved the highest "
        f"accuracy of {best_accuracy:.2%}."
    )

    st.divider()

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )

    selected_model = st.selectbox(
        "Select Model",
        list(matrices.keys())
    )

    matrix = matrices[
        selected_model
    ]

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual Deceptive",
            "Actual Truthful"
        ],
        columns=[
            "Predicted Deceptive",
            "Predicted Truthful"
        ]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )


# ============================================================
# PAGE 3 - ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.header(
        "ℹ️ About Project"
    )

    st.write(
        "The Fake Review Detection System is an "
        "NLP and machine learning based application "
        "designed to classify reviews as deceptive "
        "or truthful."
    )

    st.divider()

    # ========================================================
    # OBJECTIVE
    # ========================================================

    st.subheader(
        "🎯 Project Objective"
    )

    st.write(
        "The main objective is to identify potentially "
        "deceptive reviews using Natural Language Processing "
        "and supervised machine learning."
    )

    st.divider()

    # ========================================================
    # WORKFLOW
    # ========================================================

    st.subheader(
        "🔄 Project Workflow"
    )

    workflow = [
        "1. Collect review dataset",
        "2. Clean and prepare review text",
        "3. Convert text into TF-IDF features",
        "4. Train machine learning models",
        "5. Evaluate model performance",
        "6. Select the best-performing model",
        "7. Predict new reviews",
        "8. Deploy the application using Streamlit"
    ]

    for step in workflow:

        st.write(step)

    st.divider()

    # ========================================================
    # TECHNOLOGIES
    # ========================================================

    st.subheader(
        "🛠️ Technologies Used"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### 🐍 Python"
        )

        st.write(
            "Programming language used for "
            "the project."
        )

    with col2:

        st.markdown(
            "### 🤖 Scikit-learn"
        )

        st.write(
            "Used for TF-IDF, machine learning "
            "and model evaluation."
        )

    with col3:

        st.markdown(
            "### 🌐 Streamlit"
        )

        st.write(
            "Used to create the interactive "
            "web application."
        )

    st.divider()

    # ========================================================
    # DATASET
    # ========================================================

    st.subheader(
        "📚 Dataset"
    )

    st.write(
        f"The dataset contains "
        f"{len(dataset):,} reviews."
    )

    st.write(
        "Dataset columns:"
    )

    st.code(
        ", ".join(
            dataset.columns.tolist()
        )
    )

    st.divider()

    # ========================================================
    # FINAL MODEL
    # ========================================================

    st.subheader(
        "🏆 Final Model"
    )

    st.success(
        "Logistic Regression + TF-IDF"
    )

    st.write(
        "Logistic Regression with TF-IDF "
        "text feature extraction is used "
        "as the final model."
    )

    st.metric(
        "Reported Best Accuracy",
        "88.44%"
    )

    st.info(
        "The accuracy shown here is the reported "
        "project result. The Analytics Dashboard "
        "calculates the current metrics directly "
        "from the dataset."
    )

    st.divider()

    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.subheader(
        "⚠️ Project Limitation"
    )

    st.write(
        "The prediction is based on patterns learned "
        "from the training dataset. A machine learning "
        "prediction is not a guarantee that a review "
        "is actually fake or truthful."
    )

    st.divider()

    st.caption(
        "Fake Review Detection System | AIML Project"
    )