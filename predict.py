import pickle


# ==========================================
# LOAD MODEL AND VECTORIZER
# ==========================================

with open("fake_review_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==========================================
# HEADER
# ==========================================

print("=" * 50)
print("        FAKE REVIEW DETECTION SYSTEM")
print("=" * 50)

print("\nEnter a review below.")
print("Type 'exit' to stop the program.\n")


# ==========================================
# PREDICTION LOOP
# ==========================================

while True:

    review = input("Enter review: ")

    # Stop program
    if review.lower() == "exit":
        print("\nProgram stopped.")
        break

    # Check empty input
    if not review.strip():
        print("Please enter a review.\n")
        continue

    # Convert review to TF-IDF
    review_tfidf = vectorizer.transform([review])

    # Prediction
    prediction = model.predict(review_tfidf)[0]

    # Display result
    print("\nPrediction:", prediction.upper())

    print("-" * 50)