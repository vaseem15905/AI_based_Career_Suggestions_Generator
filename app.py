# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# import joblib
# import logging
# from flask_cors import CORS

# app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)

# # Secret key for session management
# app.secret_key = "your_secret_key"

# logging.basicConfig(level=logging.INFO)

# # Dummy user database (Replace with a real database later)
# users = {"admin": "1234", "naveen": "password"}

# # Load Model and Vectorizer
# MODEL_PATH = r"C:\skill.AI\model\decision_tree.pkl"
# VECTORIZER_PATH = r"C:\skill.AI\model\tfidf_vectorizer.pkl"

# try:
#     model = joblib.load(MODEL_PATH)
#     vectorizer = joblib.load(VECTORIZER_PATH)
#     logging.info("✅ Model and vectorizer loaded successfully!")
# except FileNotFoundError as e:
#     logging.error(f"❌ File not found: {e}")
# except Exception as e:
#     logging.error(f"❌ Error loading model or vectorizer: {e}")

# # ======== Routes =========

# @app.route('/')
# def home():
#     if 'user' in session:
#         return redirect(url_for('data_demo'))  # Redirect if already logged in
#     return render_template('login.html')

# @app.route('/login', methods=['POST'])
# def login():
#     username = request.form.get('username')
#     password = request.form.get('password')

#     logging.info(f"Login attempt with username: {username}")

#     if username in users and users[username] == password:
#         session['user'] = username  # Store user in session
#         logging.info(f"User {username} logged in successfully.")
#         return redirect(url_for('data_demo'))  # Redirect to data_demo.html
#     else:
#         logging.warning("Invalid login attempt.")
#         return render_template('login.html', error="Invalid username or password")

# @app.route('/data_demo')
# def data_demo():
#     if 'user' not in session:
#         logging.warning("Unauthorized access attempt to data_demo.")
#         return redirect(url_for('home'))  # Redirect to login if not logged in
#     logging.info(f"Rendering data_demo.html for user: {session['user']}")
#     return render_template('data_demo.html')  # Render the data_demo.html page

# @app.route('/logout')
# def logout():
#     session.pop('user', None)  # Remove user from session
#     logging.info("User logged out.")
#     return redirect(url_for('home'))  # Redirect to login page

# @app.route('/career_suggestion')
# def career_suggestion():
#     if 'user' not in session:
#         logging.warning("Unauthorized access attempt to career_suggestion.")
#         return redirect(url_for('home'))  # Redirect to login if not logged in
#     return render_template('career_suggestion.html')  # Render the career_suggestion.html page

# @app.route('/career_results')
# def career_results():
#     return render_template('career_results.html')

# # ======== API Endpoint for Model Prediction =========
# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.get_json()
#         answers = data.get('answers', [])

#         if not answers or len(answers) != 5:
#             logging.warning("Invalid number of answers received.")
#             return jsonify({'error': 'Invalid number of answers. Expected 5 answers.'}), 400

#         input_vector = vectorizer.transform(answers)
#         prediction = model.predict(input_vector)[0]

#         skill_mapping = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
#         skill_level = skill_mapping.get(prediction, 'Unknown')

#         logging.info(f"Prediction: {skill_level}")
#         return jsonify({'skill_level': skill_level})
    
#     except Exception as e:
#         logging.error(f"Error in evaluation: {e}")
#         return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

# # ======== Global Error Handler =========
# @app.errorhandler(Exception)
# def handle_exception(e):
#     logging.error(f"Unhandled Error: {e}")
#     return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

# # ======== Run the Application =========
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# 🔹 Load the trained model and vectorizer
MODEL_PATH = "models/decision_tree.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("❌ Model files are missing! Run train_evaluate_model.py first.")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# 🔹 Dummy user credentials for login
USER_CREDENTIALS = {"admin": "password123"}  # Modify as needed

# ✅ Index Route: Redirects to login
@app.route("/")
def home():
    return redirect(url_for("login"))

# ✅ Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session["user"] = username
            return redirect(url_for("career_suggestion"))
        else:
            flash("Invalid credentials! Try again.", "danger")

    return render_template("login.html")

# ✅ Career Suggestion Route (Requires Login)
@app.route("/career_suggestion", methods=["GET", "POST"])
def career_suggestion():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        user_input = request.form["skills"]
        user_vector = vectorizer.transform([user_input])
        predicted_career = model.predict(user_vector)[0]

        return render_template("career_suggestion.html", career=predicted_career)

    return render_template("career_suggestion.html", career=None)

# ✅ Logout Route
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
