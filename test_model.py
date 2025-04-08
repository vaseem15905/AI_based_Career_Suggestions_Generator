import joblib

# Load the trained model
try:
    model = joblib.load("models/decision_tree.pkl")  # Use joblib instead of pickle
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
