import joblib

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load("models/decision_tree.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    print("Model, Vectorizer, and Label Encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model components: {e}")
    exit()

# Test input: User's skills
test_skills = ["Python, Machine Learning"]

# Transform input using the trained vectorizer
try:
    user_vector = vectorizer.transform(test_skills)
    print("Vectorization successful!")
except Exception as e:
    print(f"Error in vectorization: {e}")
    exit()

# Make a prediction
try:
    predicted_label = model.predict(user_vector)[0]  # Get numeric prediction
    predicted_career = label_encoder.inverse_transform([predicted_label])[0]  # Convert to career name
    print(f"Predicted Career: {predicted_career}")
except Exception as e:
    print(f"Error during prediction: {e}")
