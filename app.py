from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import os
import logging
from logging.handlers import RotatingFileHandler

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change this to a strong secret key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# # Constants
# MODEL_PATH = "models/decision_tree.pkl"
# VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
# LABEL_ENCODER_PATH= "models/label_encoder.pkl"  # <-- Load the label encoder

# # Dummy user credentials (replace with proper authentication in production)
# USER_CREDENTIALS = {"admin": "password123"}

# def load_model_assets():
#     """Load model and vectorizer using joblib with detailed error logging"""
#     try:
#         if not all(map(os.path.exists, [MODEL_PATH, VECTORIZER_PATH])):
#             missing = [f for f in [MODEL_PATH, VECTORIZER_PATH] if not os.path.exists(f)]
#             logger.error(f"Model files missing: {missing}")
#             return None, None

#         if any(map(lambda f: os.path.getsize(f) == 0, [MODEL_PATH, VECTORIZER_PATH])):
#             empty = [f for f in [MODEL_PATH, VECTORIZER_PATH] if os.path.getsize(f) == 0]
#             logger.error(f"Empty model files: {empty}")
#             return None, None

#         model = joblib.load(MODEL_PATH)
#         vectorizer = joblib.load(VECTORIZER_PATH)

#         logger.info("Model and vectorizer loaded successfully")
#         return model, vectorizer

#     except Exception as e:
#         logger.error(f"Error loading model: {str(e)}")
#         return None, None

# # Load model and vectorizer at startup
# model, vectorizer = load_model_assets()
# Constants
MODEL_PATH = "models/decision_tree.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"  # <-- Include label encoder

# Dummy user credentials (replace with proper authentication in production)
USER_CREDENTIALS = {"admin": "password123"}
def load_model_assets():
    """Load model, vectorizer, and label encoder"""
    try:
        if not all(map(os.path.exists, [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH])):
            missing = [f for f in [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH] if not os.path.exists(f)]
            logger.error(f"Model files missing: {missing}")
            return None, None, None

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)  # <-- Load Label Encoder

        logger.info("Model, vectorizer, and label encoder loaded successfully")
        return model, vectorizer, label_encoder

    except Exception as e:
        logger.error(f"Error loading model assets: {str(e)}", exc_info=True)
        return None, None, None

# Load assets
model, vectorizer, label_encoder = load_model_assets()

@app.route("/")
def home():
    """Home page route"""
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Both username and password are required", "danger")
            return render_template("login.html")

        if USER_CREDENTIALS.get(username) == password:
            session["user"] = username
            logger.info(f"User {username} logged in successfully")
            # Changed default redirect from career_suggestion to data_demo
            next_page = request.args.get("next", url_for("data_demo"))
            return redirect(next_page)
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            flash("Invalid username or password", "danger")

    return render_template("login.html")

@app.route("/data_demo")
def data_demo():
    """Show dataset preview"""
    if "user" not in session:
        logger.warning("Unauthorized access attempt to data_demo")
        flash("Please login first", "warning")
        return redirect(url_for("login", next=url_for("data_demo")))
    
    # Sample data - replace with your actual dataset
    sample_data = [
        {"skills": "Python, Data Analysis, Machine Learning", "career": "Data Scientist"},
        {"skills": "Java, Spring, SQL", "career": "Backend Developer"},
        {"skills": "JavaScript, HTML, CSS", "career": "Frontend Developer"},
    ]
    return render_template("data_demo.html", data=sample_data)


@app.route("/career_suggestion", methods=["GET", "POST"])
def career_suggestion():
    """Handle career suggestions based on user's skills."""
    if "user" not in session:
        flash("Please login first", "warning")
        return redirect(url_for("login", next=url_for("career_suggestion")))

    # Check if model components are loaded
    if not all([model, vectorizer, label_encoder]):
        flash("Career prediction service is currently unavailable. Please try again later.", "danger")
        return render_template("career_suggestion.html", career=None)

    if request.method == "POST":
        # Get the user's skills input
        skills = request.form.get("skills", "").strip()

        if skills:
            try:
                # Example logic: map skills to career paths
                career_mapping = {
                    "Python": "Data Scientist",
                    "JavaScript": "Web Developer",
                    "Machine Learning": "AI Engineer",
                    "Software": "Software Engineer",
                    "Data": "Data Analyst"
                }

                # For simplicity, we match keywords from the skills to career paths
                suggested_career = None
                for skill in skills.split(","):
                    skill = skill.strip().lower()
                    if skill in career_mapping:
                        suggested_career = career_mapping[skill]
                        break

                # If no match is found, show a generic suggestion
                if not suggested_career:
                    suggested_career = "Software Engineer"

                return render_template("career_suggestion.html", career=suggested_career)

            except Exception as e:
                flash(f"An error occurred during prediction: {str(e)}", "danger")
                return render_template("career_suggestion.html", career=None)

    # Default rendering without career suggestion
    return render_template("career_suggestion.html", career=None)














# @app.route("/career_suggestion", methods=["GET", "POST"])
# def career_suggestion():
#     """Handle career suggestions"""
#     if "user" not in session:
#         flash("Please login first", "warning")
#         return redirect(url_for("login", next=url_for("career_suggestion")))

#     # Check if model components are loaded
#     if not all([model, vectorizer, label_encoder]):
#         flash("Career prediction service is currently unavailable. Please try again later.", "danger")
#         return render_template("career_suggestion.html", career=None)

#     if request.method == "POST":
#         skills = request.form.get("skills", "").strip()
#         if not skills:
#             flash("Please enter your skills", "warning")
#             return render_template("career_suggestion.html", career=None)

#         try:
#             logger.info(f"User Input: {skills}")

#             # Transform input
#             user_vector = vectorizer.transform([skills])
#             logger.info(f"Vectorized Input Shape: {user_vector.shape}")

#             # Predict career
#             predicted_label = model.predict(user_vector)[0]
#             logger.info(f"Predicted Label: {predicted_label}")

#             # Convert label to career name
#             suggested_career = label_encoder.inverse_transform([predicted_label])[0]
#             logger.info(f"Suggested Career: {suggested_career}")

#             # Mock skill-based questions for now
#             skill_questions = {
#                 "AI": ["What is supervised learning?", "Explain the difference between AI and ML."],
#                 "Web Development": ["What is the difference between HTML and React?", "Explain REST APIs."],
#                 "Data Science": ["What is data preprocessing?", "Explain feature engineering."],
#             }

#             questions = skill_questions.get(suggested_career, ["No questions available."])
#             print(f"DEBUG - Passing to career_results.html: {suggested_career}, {questions}")  # Debugging

#             return render_template(
#                 "career_results.html",
#                 careers=[suggested_career],  # Pass as a list
#                 questions={suggested_career: questions},  # Format questions properly
#             )

#         except Exception as e:
#             import traceback
#             error_details = traceback.format_exc()
#             print(f"Error during prediction:\n{error_details}")  # Debug print
#             flash(f"An error occurred during prediction: {str(e)}", "danger")

#     return render_template("career_suggestion.html", career=None)







# @app.route("/career_suggestion", methods=["GET", "POST"])
# def career_suggestion():
#     """Handle career suggestions"""
#     if "user" not in session:
#         flash("Please login first", "warning")
#         return redirect(url_for("login", next=url_for("career_suggestion")))

#     # Check if model components are loaded
#     if not all([model, vectorizer, label_encoder]):
#         flash("Career prediction service is currently unavailable. Please try again later.", "danger")
#         return render_template("career_suggestion.html", career=None)

#     if request.method == "POST":
#         skills = request.form.get("skills", "").strip()
#         if not skills:
#             flash("Please enter your skills", "warning")
#             return render_template("career_suggestion.html", career=None)

#         try:
#             logger.info(f"User Input: {skills}")

#             # Transform input
#             user_vector = vectorizer.transform([skills])
#             logger.info(f"Vectorized Input Shape: {user_vector.shape}")

#             # Predict career
#             predicted_label = model.predict(user_vector)[0]
#             logger.info(f"Predicted Label: {predicted_label}")

#             # Convert label to career name
#             suggested_career = label_encoder.inverse_transform([predicted_label])[0]
#             logger.info(f"Suggested Career: {suggested_career}")

#             print(f"DEBUG - Passing to career_results.html: {suggested_career}")  # Debugging

#             return render_template(
#                 "career_results.html",
#                 question=skills,
#                 suggested_career=suggested_career
#             )

#         except Exception as e:
#             import traceback
#             error_details = traceback.format_exc()
#             print(f"Error during prediction:\n{error_details}")  # Debug print
#             flash(f"An error occurred during prediction: {str(e)}", "danger")

#     return render_template("career_suggestion.html", career=None)




# @app.route("/career_suggestion", methods=["GET", "POST"])
# def career_suggestion():
#     """Handle career suggestions"""
#     if "user" not in session:
#         flash("Please login first", "warning")
#         return redirect(url_for("login", next=url_for("career_suggestion")))

#     # Check if model components are loaded
#     if not all([model, vectorizer, label_encoder]):
#         flash("Career prediction service is currently unavailable. Please try again later.", "danger")
#         return render_template("career_suggestion.html", career=None)

#     if request.method == "POST":
#         skills = request.form.get("skills", "").strip()
#         if not skills:
#             flash("Please enter your skills", "warning")
#             return render_template("career_suggestion.html", career=None)

#         try:
#             logger.info(f"User Input: {skills}")

#             # Transform input
#             user_vector = vectorizer.transform([skills])
#             logger.info(f"Vectorized Input Shape: {user_vector.shape}")

#             # Predict career
#             predicted_label = model.predict(user_vector)[0]
#             logger.info(f"Predicted Label: {predicted_label}")

#             # Convert label to career name
#             suggested_career = label_encoder.inverse_transform([predicted_label])[0]
#             logger.info(f"Suggested Career: {suggested_career}")

#             return render_template("career_results.html", question=skills, suggested_career=suggested_career)

#         except Exception as e:
#             import traceback
#             error_details = traceback.format_exc()
#             print(f"Error during prediction:\n{error_details}")  # Debug print
#             flash(f"An error occurred during prediction: {str(e)}", "danger") 
#             # Show specific error
#     return render_template("career_suggestion.html", career=None)



@app.route("/logout")
def logout():
    """Handle user logout"""
    username = session.pop("user", None)
    if username:
        logger.info(f"User {username} logged out")
        flash("You have been logged out successfully", "success")
    return redirect(url_for("home"))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template("500.html"), 500

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Run the app
    app.run(host="0.0.0.0", port=5000, debug=True)