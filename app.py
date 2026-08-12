import streamlit as st
import pickle
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("fake_review_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# AUTOMATIC MODEL EVALUATION
# ============================================================

@st.cache_data
def calculate_evaluation():

    # Load dataset
    data = pd.read_csv("deceptive-opinion.csv")

    X = data["text"]
    y = data["deceptive"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # TF-IDF
    evaluation_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X_train_tfidf = evaluation_vectorizer.fit_transform(X_train)
    X_test_tfidf = evaluation_vectorizer.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC()
    }

    results = {}

    # Train and evaluate each model
    for name, current_model in models.items():

        current_model.fit(
            X_train_tfidf,
            y_train
        )

        predictions = current_model.predict(
            X_test_tfidf
        )

        results[name] = {

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ) * 100,

            "Precision": precision_score(
                y_test,
                predictions,
                pos_label="deceptive"
            ) * 100,

            "Recall": recall_score(
                y_test,
                predictions,
                pos_label="deceptive"
            ) * 100,

            "F1 Score": f1_score(
                y_test,
                predictions,
                pos_label="deceptive"
            ) * 100,

            "Confusion Matrix": confusion_matrix(
                y_test,
                predictions,
                labels=[
                    "deceptive",
                    "truthful"
                ]
            )
        }

    return results


# Calculate evaluation
evaluation_results = calculate_evaluation()


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
# MAIN TITLE
# ============================================================

st.markdown(
    "<h1 style='text-align:center;'>"
    "🔍 Fake Review Detection System"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "NLP and Machine Learning Based Deceptive Review Detection"
    "</p>",
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 - REVIEW PREDICTION
# ============================================================

if page == "📝 Review Prediction":

    st.header("📝 Review Prediction")

    st.write(
        "Enter a hotel or product review below. "
        "The machine learning model will predict whether "
        "the review is deceptive or truthful."
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
        type="primary"
    ):

        if not review.strip():

            st.warning(
                "⚠️ Please enter a review first."
            )

        else:

            # Convert review to TF-IDF
            review_tfidf = vectorizer.transform(
                [review]
            )

            # Predict
            prediction = model.predict(
                review_tfidf
            )[0]

            st.divider()

            # Display result
            if prediction == "deceptive":

                st.error(
                    "🚨 DECEPTIVE REVIEW"
                )

                st.write(
                    "The model predicts that this review "
                    "has patterns associated with deceptive reviews."
                )

            else:

                st.success(
                    "✅ TRUTHFUL REVIEW"
                )

                st.write(
                    "The model predicts that this review "
                    "has patterns associated with truthful reviews."
                )

    st.divider()

    st.caption(
        "Model: Logistic Regression + TF-IDF"
    )


# ============================================================
# PAGE 2 - ANALYTICS DASHBOARD
# ============================================================

elif page == "📊 Analytics Dashboard":

    st.header("📊 Model Analytics Dashboard")

    st.write(
        "All metrics on this page are automatically calculated "
        "from the test dataset."
    )

    st.divider()


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    data = pd.read_csv(
        "deceptive-opinion.csv"
    )

    total_samples = len(data)

    training_samples = int(
        total_samples * 0.80
    )

    testing_samples = total_samples - training_samples


    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    best_model = max(
        evaluation_results,
        key=lambda name:
        evaluation_results[name]["Accuracy"]
    )

    best_accuracy = evaluation_results[
        best_model
    ]["Accuracy"]


    # ========================================================
    # TOP METRICS
    # ========================================================

    st.subheader(
        "📌 Dataset & Model Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Dataset",
            total_samples
        )

    with col2:

        st.metric(
            "Training Samples",
            training_samples
        )

    with col3:

        st.metric(
            "Testing Samples",
            testing_samples
        )

    with col4:

        st.metric(
            "Best Accuracy",
            f"{best_accuracy:.2f}%"
        )


    st.divider()


    # ========================================================
    # MODEL ACCURACY
    # ========================================================

    st.subheader(
        "🤖 Model Accuracy Comparison"
    )

    accuracy_data = pd.DataFrame({

        "Model": list(
            evaluation_results.keys()
        ),

        "Accuracy": [
            evaluation_results[name]["Accuracy"]
            for name in evaluation_results
        ]
    })

    st.bar_chart(
        accuracy_data.set_index(
            "Model"
        )
    )

    st.dataframe(
        accuracy_data.round(2),
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # ========================================================
    # DETAILED EVALUATION
    # ========================================================

    st.subheader(
        "📋 Detailed Model Evaluation"
    )

    detailed_data = pd.DataFrame({

        "Model": list(
            evaluation_results.keys()
        ),

        "Accuracy": [
            evaluation_results[name]["Accuracy"]
            for name in evaluation_results
        ],

        "Precision": [
            evaluation_results[name]["Precision"]
            for name in evaluation_results
        ],

        "Recall": [
            evaluation_results[name]["Recall"]
            for name in evaluation_results
        ],

        "F1 Score": [
            evaluation_results[name]["F1 Score"]
            for name in evaluation_results
        ]
    })

    st.dataframe(
        detailed_data.round(2),
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # ========================================================
    # PRECISION RECALL F1 CHART
    # ========================================================

    st.subheader(
        "📈 Precision, Recall and F1-Score"
    )

    metric_data = detailed_data.set_index(
        "Model"
    )[
        [
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]

    st.bar_chart(
        metric_data
    )


    st.divider()


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔢 Confusion Matrix"
    )

    selected_model = st.selectbox(
        "Select a model:",
        list(
            evaluation_results.keys()
        )
    )

    selected_matrix = evaluation_results[
        selected_model
    ]["Confusion Matrix"]

    matrix_data = pd.DataFrame(

        selected_matrix,

        columns=[
            "Predicted Deceptive",
            "Predicted Truthful"
        ],

        index=[
            "Actual Deceptive",
            "Actual Truthful"
        ]
    )

    st.dataframe(
        matrix_data,
        use_container_width=True
    )


    st.write(
        "The confusion matrix shows the number of "
        "correct and incorrect predictions."
    )


    st.divider()


    # ========================================================
    # DATASET DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Dataset Distribution"
    )

    distribution = (
        data["deceptive"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "Review Type",
        "Count"
    ]

    st.bar_chart(
        distribution.set_index(
            "Review Type"
        )
    )

    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader(
        "🏆 Best Performing Model"
    )

    st.success(
        f"🏆 {best_model} is the best-performing model "
        f"with an accuracy of {best_accuracy:.2f}%."
    )


# ============================================================
# PAGE 3 - ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.header(
        "ℹ️ About Project"
    )

    st.write(
        "Fake Review Detection is a Natural Language Processing "
        "(NLP) and Machine Learning project designed to identify "
        "whether a review is deceptive or truthful."
    )

    st.divider()


    # ========================================================
    # OBJECTIVE
    # ========================================================

    st.subheader(
        "🎯 Project Objective"
    )

    st.write(
        "The objective of this project is to develop a machine "
        "learning system that analyzes review text and predicts "
        "whether the review is deceptive or truthful."
    )


    st.divider()


    # ========================================================
    # METHODOLOGY
    # ========================================================

    st.subheader(
        "⚙️ Project Methodology"
    )

    st.markdown(
        """
        **1. Dataset Collection**

        A dataset containing 1,600 hotel reviews is used.

        **2. Text Preprocessing**

        Review text is prepared for machine learning.

        **3. TF-IDF Vectorization**

        Text is converted into numerical features using TF-IDF.

        **4. Model Training**

        Three machine learning algorithms are evaluated:

        - Logistic Regression
        - Naive Bayes
        - Support Vector Machine

        **5. Model Evaluation**

        Models are evaluated using:

        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Confusion Matrix

        **6. Web Application**

        Streamlit is used to create the interactive web interface.
        """
    )


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
            """
            ### 🐍 Python

            Programming language used
            for the project.
            """
        )

    with col2:

        st.markdown(
            """
            ### 🤖 Scikit-learn

            Machine learning and
            model evaluation.
            """
        )

    with col3:

        st.markdown(
            """
            ### 🌐 Streamlit

            Interactive web interface.
            """
        )


    st.divider()


    # ========================================================
    # FINAL MODEL
    # ========================================================

    st.subheader(
        "🏆 Final Model"
    )

    st.success(
       f"Logistic Regression + TF-IDF"
    )

    st.metric(
        "Best Accuracy",
        f"{best_accuracy:.2f}%"
    )


    st.divider()

    st.caption(
        "Fake Review Detection System | AIML Project"
    )