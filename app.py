from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
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

# Constants
MODEL_PATH = "models/decision_tree.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Dummy user credentials (replace with proper authentication in production)
USER_CREDENTIALS = {"admin": "password123"}

def load_model_assets():
    """Load model and vectorizer with proper error handling"""
    try:
        # Verify files exist and are not empty
        if not all(map(os.path.exists, [MODEL_PATH, VECTORIZER_PATH])):
            missing = [f for f in [MODEL_PATH, VECTORIZER_PATH] if not os.path.exists(f)]
            raise FileNotFoundError(f"Model files missing: {missing}")
            
        if any(map(lambda f: os.path.getsize(f) == 0, [MODEL_PATH, VECTORIZER_PATH])):
            empty = [f for f in [MODEL_PATH, VECTORIZER_PATH] if os.path.getsize(f) == 0]
            raise ValueError(f"Empty model files: {empty}")

        # Try different protocols and encodings to load pickle files
        for encoding in ['latin1', 'utf-8', 'bytes']:
            try:
                with open(MODEL_PATH, "rb") as model_file:
                    model = pickle.load(model_file, encoding=encoding)
                
                with open(VECTORIZER_PATH, "rb") as vectorizer_file:
                    vectorizer = pickle.load(vectorizer_file, encoding=encoding)
                
                logger.info("Model and vectorizer loaded successfully")
                return model, vectorizer
                
            except Exception as e:
                logger.warning(f"Attempt with encoding {encoding} failed: {str(e)}")
                continue

        # If all attempts failed
        raise RuntimeError("All attempts to load model files failed with different encodings")
        
    except Exception as e:
        logger.error(f"Critical error loading model assets: {str(e)}")
        # Instead of exiting, we'll set model and vectorizer to None
        # and check them before making predictions
        return None, None

# Load model and vectorizer at startup
model, vectorizer = load_model_assets()

@app.route("/")
def home():
    """Redirect to login page"""
    return redirect(url_for("login"))

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
            next_page = request.args.get("next", url_for("career_suggestion"))
            return redirect(next_page)
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            flash("Invalid username or password", "danger")
    
    return render_template("login.html")

@app.route("/career_suggestion", methods=["GET", "POST"])
def career_suggestion():
    """Handle career suggestions"""
    if "user" not in session:
        logger.warning("Unauthorized access attempt to career_suggestion")
        flash("Please login first", "warning")
        return redirect(url_for("login", next=request.url))
    
    # Check if model and vectorizer are loaded
    if model is None or vectorizer is None:
        logger.error("Model or vectorizer not loaded - prediction unavailable")
        flash("Career prediction service is currently unavailable. Please try again later.", "danger")
        return render_template("career_suggestion.html", career=None)
    
    if request.method == "POST":
        skills = request.form.get("skills", "").strip()
        if not skills:
            flash("Please enter your skills", "warning")
            return render_template("career_suggestion.html", career=None)
        
        try:
            user_vector = vectorizer.transform([skills])
            predicted_career = model.predict(user_vector)[0]
            logger.info(f"Career prediction made for user {session['user']}")
            return render_template("career_suggestion.html", career=predicted_career)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash("An error occurred during prediction. Please try again.", "danger")
    
    return render_template("career_suggestion.html", career=None)

@app.route("/logout")
def logout():
    """Handle user logout"""
    username = session.pop("user", None)
    if username:
        logger.info(f"User {username} logged out")
        flash("You have been logged out successfully", "success")
    return redirect(url_for("login"))

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
