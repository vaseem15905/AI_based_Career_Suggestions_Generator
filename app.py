from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import joblib
import os
import logging
from logging.handlers import RotatingFileHandler
import random

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Model paths
MODEL_PATH = "models/decision_tree.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# In-memory storage
users = {"admin": "password123"}
student_data = {}

QUESTIONS = {
    "C": [
        {"question": "What are pointers in C?", "answer": "Pointers store memory addresses."},
        {"question": "Explain memory management in C.", "answer": "Memory is allocated dynamically using malloc, free."},
        {"question": "Explain the difference between C and C++.", "answer": "C++ supports OOP while C is procedural."},
        {"question": "What is a segmentation fault in C?", "answer": "It occurs when a program tries to access restricted memory."},
        {"question": "What is a NULL pointer?", "answer": "A pointer that does not point to any memory location."},
        {"question": "What is the use of 'sizeof' operator in C?", "answer": "It returns the size of a data type or variable."}
    ],
    "C++": [
        {"question": "What is polymorphism in C++?", "answer": "Polymorphism allows functions to take multiple forms."},
        {"question": "What is a constructor in C++?", "answer": "A constructor initializes objects when they are created."},
        {"question": "What is operator overloading in C++?", "answer": "It allows you to redefine the meaning of operators."},
        {"question": "What is a virtual function in C++?", "answer": "A function that can be overridden in a derived class."}
    ],
    "Python": [
        {"question": "What are Python decorators?", "answer": "Decorators modify function behavior without changing code."},
        {"question": "Explain list comprehensions in Python.", "answer": "List comprehensions allow concise list creation."},
        {"question": "What is a lambda function?", "answer": "An anonymous function defined using the lambda keyword."},
        {"question": "What is the difference between 'is' and '==' in Python?", "answer": "'is' checks identity, '==' checks value equality."},
        {"question": "What are Python's data types?", "answer": "Common types include int, float, str, list, tuple, dict."}
    ],
    "Java": [
        {"question": "What is the difference between an interface and an abstract class in Java?", "answer": "Abstract classes allow method implementation, interfaces do not."},
        {"question": "What is the JVM?", "answer": "Java Virtual Machine runs Java bytecode on any platform."},
        {"question": "What is method overloading?", "answer": "Defining multiple methods with the same name but different parameters."},
        {"question": "What is the purpose of the 'final' keyword in Java?", "answer": "It prevents further modification of variables, methods, or classes."}
    ],
    "Web": [
        {"question": "What is the difference between HTML and HTML5?", "answer": "HTML5 supports modern features like audio, video."},
        {"question": "What does 'HTTP' stand for?", "answer": "HyperText Transfer Protocol."},
        {"question": "What is CSS used for?", "answer": "CSS styles the HTML elements visually."},
        {"question": "What is JavaScript primarily used for?", "answer": "It adds interactivity and dynamic content to websites."},
        {"question": "What is the purpose of a meta tag in HTML?", "answer": "Meta tags provide metadata like character set or page description."}
    ],
    "General": [
        {"question": "Explain what a variable is in programming.", "answer": "A variable is a named storage location that holds data."},
        {"question": "Explain REST API principles.", "answer": "REST APIs use HTTP methods to perform CRUD operations on resources."},
        {"question": "Explain the CAP theorem in distributed systems.", "answer": "CAP states a distributed system can't simultaneously guarantee Consistency, Availability, and Partition tolerance."},
        {"question": "What is a compiler?", "answer": "A compiler translates code from high-level to machine language."},
        {"question": "What is version control?", "answer": "It's a system to track changes in code over time, e.g., Git."},
        {"question": "What is an algorithm?", "answer": "A step-by-step procedure to solve a problem or perform a task."}
    ]
}


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return model, vectorizer, label_encoder
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None


model, vectorizer, label_encoder = load_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Both fields are required", "danger")
            return render_template("login.html")

        if users.get(username) == password:
            session["user"] = username
            return redirect(url_for("data_demo"))

        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/data_demo")
def data_demo():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("data_demo.html")


@app.route("/get_questions", methods=["POST"])
def get_questions():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    selected_skills = data.get("skills", [])

    student_data[session["user"]] = {
        "info": {
            "name": data.get("fullName"),
            "id": data.get("studentId"),
            "email": data.get("email"),
            "class": data.get("studentClass")
        },
        "skills": selected_skills
    }

    gathered_questions = []
    for skill in selected_skills:
        if skill in QUESTIONS:
            gathered_questions.extend(QUESTIONS[skill])

    if not gathered_questions:
        gathered_questions = QUESTIONS.get("General", [])

    questions = random.sample(gathered_questions, min(4, len(gathered_questions)))
    return jsonify({"questions": questions, "skills": selected_skills})


@app.route("/evaluate_answers", methods=["POST"])
def evaluate_answers():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not all([model, vectorizer, label_encoder]):
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    evaluations = []
    correct = 0
    total_similarity = 0

    for answer in data["answers"]:
        vectors = vectorizer.transform([answer["userAnswer"], answer["expectedAnswer"]])
        similarity = (vectors[0] @ vectors[1].T).toarray()[0][0]
        is_correct = similarity > 0.7
        evaluations.append({
            "user_answer": answer["userAnswer"],
            "expected_answer": answer["expectedAnswer"],
            "similarity": float(similarity),
            "is_correct": bool(is_correct)
        })
        if is_correct:
            correct += 1
        total_similarity += similarity

    avg_score = (total_similarity / len(evaluations)) * 100 if evaluations else 0
    skill_level = "Beginner"
    if avg_score > 70: skill_level = "Intermediate"
    if avg_score > 85: skill_level = "Expert"

    if session["user"] not in student_data:
        student_data[session["user"]] = {}

    student_data[session["user"]]["results"] = {
        "score": avg_score,
        "skill_level": skill_level,
        "correct": correct,
        "total": len(evaluations)
    }

    return jsonify({
        "evaluations": evaluations,
        "correct_count": correct,
        "total_questions": len(evaluations),
        "average_score": avg_score,
        "skill_level": skill_level
    })


@app.route("/career_suggestion", methods=["GET", "POST"])
def career_suggestion():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        return redirect(url_for("career_results"))

    user_data = student_data.get(session["user"], {})
    return render_template("career_suggestion.html",
                           skills=user_data.get("skills", []),
                           results=user_data.get("results"))


@app.route("/career_results", methods=["POST"])
def career_results():
    if "user" not in session:
        return redirect(url_for("login"))

    skills_input = request.form.get("skills", "")
    skills = [skill.strip().lower() for skill in skills_input.split(",") if skill.strip()]

    suggestions = []
    if any(skill in ["python", "data", "machine learning"] for skill in skills):
        suggestions.append("Data Scientist")
    if any(skill in ["web", "html", "css", "javascript"] for skill in skills):
        suggestions.append("Web Developer")
    if any(skill in ["java", "c++", "oop"] for skill in skills):
        suggestions.append("Software Engineer")

    if not suggestions:
        suggestions.append("Explore more to find your interests!")

    return render_template("career_suggestion.html",
                           suggestions=suggestions,
                           skills=skills,
                           results=student_data.get(session["user"], {}).get("results"))


if __name__ == "__main__":
    app.run(debug=True)
