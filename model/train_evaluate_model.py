import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Enhanced dataset with more questions
data = [
    # Beginner Level
    ("What are pointers in C?", "Pointers store memory addresses.", "Beginner"),
    ("What is the difference between HTML and HTML5?", "HTML5 supports modern features like audio, video.", "Beginner"),
    ("Explain what a variable is in programming.", "A variable is a named storage location that holds data.", "Beginner"),
    ("What does 'HTTP' stand for?", "HyperText Transfer Protocol.", "Beginner"),
    ("What is a for loop?", "A for loop repeats code a specific number of times.", "Beginner"),
    ("Define an array in programming.", "An array is a collection of items stored at contiguous memory locations.", "Beginner"),

    # Intermediate Level
    ("Explain memory management in C.", "Memory is allocated dynamically using malloc, free.", "Intermediate"),
    ("What is polymorphism in C++?", "Polymorphism allows functions to take multiple forms.", "Intermediate"),
    ("What are Python decorators?", "Decorators modify function behavior without changing code.", "Intermediate"),
    ("Explain REST API principles.", "REST APIs use HTTP methods to perform CRUD operations on resources.", "Intermediate"),
    ("What is recursion in programming?", "Recursion is when a function calls itself to solve smaller instances.", "Intermediate"),
    ("Explain the concept of Big O notation.", "Big O describes algorithm performance in terms of input size growth.", "Intermediate"),

    # Expert Level
    ("Explain the difference between C and C++.", "C++ supports OOP while C is procedural.", "Expert"),
    ("Explain list comprehensions in Python.", "List comprehensions allow concise list creation.", "Expert"),
    ("What is the difference between an interface and an abstract class in Java?", "Abstract classes allow method implementation, interfaces do not.", "Expert"),
    ("Explain how garbage collection works in Java.", "Garbage collection automatically reclaims memory from unused objects.", "Expert"),
    ("What are design patterns in software engineering?", "Design patterns are reusable solutions to common problems in software design.", "Expert"),
    ("Explain the CAP theorem in distributed systems.", "CAP states a distributed system can't simultaneously guarantee Consistency, Availability, and Partition tolerance.", "Expert"),
    ("What is the difference between symmetric and asymmetric encryption?", "Symmetric uses one key, asymmetric uses public/private key pairs.", "Expert"),
    ("Explain the MapReduce programming model.", "MapReduce processes large datasets by mapping and reducing functions across clusters.", "Expert"),

    # Newly Added (classified based on estimated difficulty)
    ("What is the difference between an interface and an abstract class in Java?", "Abstract classes allow method implementation, interfaces do not.", "Expert"),
    ("What is polymorphism in C++?", "Polymorphism allows functions to take multiple forms.", "Intermediate"),
    ("What are Python decorators?", "Decorators modify function behavior without changing code.", "Intermediate"),
    ("Explain list comprehensions in Python.", "List comprehensions allow concise list creation.", "Expert"),
    ("What does 'HTTP' stand for?", "HyperText Transfer Protocol.", "Beginner"),
]


# Convert data to DataFrame
df = pd.DataFrame(data, columns=["Question", "Answer", "Skill_Level"])

# Encode skill levels
label_encoder = LabelEncoder()
df["Skill_Level"] = label_encoder.fit_transform(df["Skill_Level"])

# Vectorize Answers using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Answer"])  # Convert text to numerical vectors
y = df["Skill_Level"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Save trained model and vectorizer
joblib.dump(model, "../models/decision_tree.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")


def evaluate_answer(model, vectorizer, label_encoder, user_answer, expected_answer):
    """Evaluate if user answer matches expected answer"""
    # Vectorize both answers
    vectors = vectorizer.transform([user_answer, expected_answer])

    # Predict skill levels
    user_pred = model.predict(vectors[0])
    expected_pred = model.predict(vectors[1])

    # Decode skill levels
    user_level = label_encoder.inverse_transform(user_pred)[0]
    expected_level = label_encoder.inverse_transform(expected_pred)[0]

    # Calculate similarity score
    similarity = (vectors[0] @ vectors[1].T).toarray()[0][0]

    return {
        "user_level": user_level,
        "expected_level": expected_level,
        "similarity": similarity,
        "is_correct": similarity > 0.7  # Threshold can be adjusted
    }

print("Model training complete. Model saved in 'models/' folder.")