# 🔍 Fake Review Detection System

A machine learning based web application that predicts whether a hotel or product review is likely to be deceptive or truthful.

## 📌 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify reviews.

## 🧠 Machine Learning

The project compares three models:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

## 📊 Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 88.44% |
| Naive Bayes | 86.56% |
| SVM | 85.94% |

### Best Model

**Logistic Regression — 88.44%**

## 🔤 Text Processing

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert review text into numerical features.

## 🖥️ Web Interface

The application is developed using Streamlit.

### Features

- Review prediction
- Model analytics dashboard
- Model comparison
- Project information
- Interactive user interface

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## 📁 Project Structure

```text
fake-review-detection/
│
├── app.py
├── fake_review_model.pkl
├── tfidf_vectorizer.pkl
├── evaluation.py
├── compare_models.py
├── requirements.txt
└── README.md