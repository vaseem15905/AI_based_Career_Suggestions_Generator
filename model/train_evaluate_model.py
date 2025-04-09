# # import pandas as pd
# # import joblib
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.model_selection import train_test_split

# # # Sample dataset: Questions, Correct Answers, and Corresponding Skill Levels
# # data = [
# #     ("What are pointers in C?", "Pointers store memory addresses.", "Beginner"),
# #     ("Explain memory management in C.", "Memory is allocated dynamically using malloc, free.", "Intermediate"),
# #     ("What is polymorphism in C++?", "Polymorphism allows functions to take multiple forms.", "Intermediate"),
# #     ("Explain the difference between C and C++.", "C++ supports OOP while C is procedural.", "Expert"),
# #     ("What are Python decorators?", "Decorators modify function behavior without changing code.", "Intermediate"),
# #     ("Explain list comprehensions in Python.", "List comprehensions allow concise list creation.", "Expert"),
# #     ("What is the difference between an interface and an abstract class in Java?", "Abstract classes allow method implementation, interfaces do not.", "Expert"),
# #     ("What is the difference between HTML and HTML5?", "HTML5 supports modern features like audio, video.", "Beginner")
# # ]

# # # Convert data to a DataFrame
# # df = pd.DataFrame(data, columns=["Question", "Answer", "Skill_Level"])

# # # Encode Skill Levels
# # skill_mapping = {"Beginner": 0, "Intermediate": 1, "Expert": 2}
# # df["Skill_Level"] = df["Skill_Level"].map(skill_mapping)

# # # Vectorize Answers using TF-IDF
# # vectorizer = TfidfVectorizer()
# # X = vectorizer.fit_transform(df["Answer"])  # Convert text to numerical vectors
# # y = df["Skill_Level"]

# # # Split the dataset (80% training, 20% testing)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Train Decision Tree Classifier
# # model = DecisionTreeClassifier()
# # model.fit(X_train, y_train)

# # # Save the trained model & vectorizer
# # joblib.dump(model, "model/decision_tree.pkl")
# # joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

# # print("✅ Model training complete. Model saved as 'decision_tree.pkl'.")
# # import pandas as pd
# # import joblib
# # import os
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.model_selection import train_test_split

# # # Sample dataset: Questions, Answers, and Skill Levels
# # data = [
# #     ("What are pointers in C?", "Pointers store memory addresses.", "Beginner"),
# #     ("Explain memory management in C.", "Memory is allocated dynamically using malloc, free.", "Intermediate"),
# #     ("What is polymorphism in C++?", "Polymorphism allows functions to take multiple forms.", "Intermediate"),
# #     ("Explain the difference between C and C++.", "C++ supports OOP while C is procedural.", "Expert"),
# #     ("What are Python decorators?", "Decorators modify function behavior without changing code.", "Intermediate"),
# #     ("Explain list comprehensions in Python.", "List comprehensions allow concise list creation.", "Expert"),
# #     ("What is the difference between an interface and an abstract class in Java?", "Abstract classes allow method implementation, interfaces do not.", "Expert"),
# #     ("What is the difference between HTML and HTML5?", "HTML5 supports modern features like audio, video.", "Beginner")
# # ]

# # # Convert data to DataFrame
# # df = pd.DataFrame(data, columns=["Question", "Answer", "Skill_Level"])

# # # Encode skill levels
# # skill_mapping = {"Beginner": 0, "Intermediate": 1, "Expert": 2}
# # df["Skill_Level"] = df["Skill_Level"].map(skill_mapping)

# # # Vectorize Answers using TF-IDF
# # vectorizer = TfidfVectorizer()
# # X = vectorizer.fit_transform(df["Answer"])  # Convert text to numerical vectors
# # y = df["Skill_Level"]

# # # Split data (80% training, 20% testing)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Train Decision Tree Classifier
# # model = DecisionTreeClassifier()
# # model.fit(X_train, y_train)

# # # Ensure model directory exists
# # os.makedirs("models", exist_ok=True)

# # # Save trained model and vectorizer
# # joblib.dump(model, "models/decision_tree.pkl")
# # joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# # print("Model training complete. Model saved in 'models/' folder.")



# import pandas as pd
# import joblib
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split

# # Sample dataset: Questions, Answers, and Skill Levels
# data = [
#     ("What are pointers in C?", "Pointers store memory addresses.", "Beginner"),
#     ("Explain memory management in C.", "Memory is allocated dynamically using malloc, free.", "Intermediate"),
#     ("What is polymorphism in C++?", "Polymorphism allows functions to take multiple forms.", "Intermediate"),
#     ("Explain the difference between C and C++.", "C++ supports OOP while C is procedural.", "Expert"),
#     ("What are Python decorators?", "Decorators modify function behavior without changing code.", "Intermediate"),
#     ("Explain list comprehensions in Python.", "List comprehensions allow concise list creation.", "Expert"),
#     ("What is the difference between an interface and an abstract class in Java?", 
#      "Abstract classes allow method implementation, interfaces do not.", "Expert"),
#     ("What is the difference between HTML and HTML5?", "HTML5 supports modern features like audio, video.", "Beginner")
# ]

# # Convert data to DataFrame
# df = pd.DataFrame(data, columns=["Question", "Answer", "Skill_Level"])

# # Encode skill levels
# skill_mapping = {"Beginner": 0, "Intermediate": 1, "Expert": 2}
# df["Skill_Level"] = df["Skill_Level"].map(skill_mapping)

# # Vectorize Answers using TF-IDF
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df["Answer"])  # Convert text to numerical vectors
# y = df["Skill_Level"]

# # Split data (80% training, 20% testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Decision Tree Classifier
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Ensure model directory exists
# os.makedirs("models", exist_ok=True)

# # Save trained model and vectorizer
# joblib.dump(model, "models/decision_tree.pkl")
# joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# print("Model training complete. Model saved in 'models/' folder.")








import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample dataset: Answers mapped to careers
data = [
    ("Python, Machine Learning, Data Analysis", "Data Scientist"),
    ("HTML, CSS, JavaScript, React", "Frontend Developer"),
    ("C, C++, Algorithms, Data Structures", "Software Engineer"),
    ("Python, Flask, Django, APIs", "Backend Developer"),
    ("Cloud Computing, Docker, Kubernetes", "DevOps Engineer"),
    ("Cybersecurity, Network Security, Penetration Testing", "Cybersecurity Specialist"),
    ("Java, Spring Boot, Microservices", "Backend Developer"),
    ("SQL, Database Management, Data Warehousing", "Database Administrator"),
    ("UI/UX Design, Figma, Photoshop", "UI/UX Designer"),
    ("Embedded Systems, IoT, Microcontrollers", "Embedded Systems Engineer"),
]

# Convert data to DataFrame
df = pd.DataFrame(data, columns=["Skills", "Career"])

# Encode career labels
label_encoder = LabelEncoder()
df["Career"] = label_encoder.fit_transform(df["Career"])

# Vectorize skills using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Skills"])  # Convert text to numerical vectors
y = df["Career"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Save trained model, vectorizer, and label encoder
joblib.dump(model, "models/decision_tree.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Model training complete. Model, vectorizer, and label encoder saved in 'models/' folder.")
