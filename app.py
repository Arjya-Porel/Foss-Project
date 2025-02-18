from flask import Flask, render_template, request, redirect, url_for, flash
import os
import fitz  # PyMuPDF
import spacy
import sqlite3
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize database
conn = sqlite3.connect("quiz.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, topic TEXT, score INTEGER)")
conn.commit()

# Machine Learning Model for Weak Area Prediction
X = np.array([[80], [50], [60], [30], [90]])  # Sample quiz scores
y = ["Strong", "Weak", "Weak", "Very Weak", "Strong"]
model = RandomForestClassifier()
model.fit(X, y)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = " ".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to generate quiz questions
def generate_questions(text):
    questions = []
    for sentence in text.split("."):
        sentence = sentence.strip()
        if not sentence:
            continue

        fill_blank = generate_fill_in_the_blank(sentence)
        if fill_blank:
            questions.append(fill_blank)

        mcq = generate_mcq(sentence)
        if mcq:
            questions.append(mcq)

        tf = generate_true_false(sentence)
        if tf:
            questions.append(tf)

        short_ans = generate_short_answer(sentence)
        if short_ans:
            questions.append(short_ans)

    return questions[:10]  # Limit to 10 questions

# Question Generation Functions
def generate_fill_in_the_blank(sentence):
    doc = nlp(sentence)
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    if not nouns:
        return None
    blank = random.choice(nouns)
    question = sentence.replace(blank, "____", 1)
    return {"type": "fill_blank", "question": question, "answer": blank}

def generate_mcq(sentence):
    doc = nlp(sentence)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]
    if not entities:
        return None
    correct_answer = random.choice(entities)
    question = sentence.replace(correct_answer, "____")
    choices = list(set(entities) - {correct_answer})[:3]
    choices.append(correct_answer)
    while len(choices) < 4:  # Ensure four choices
        choices.append("RandomOption" + str(random.randint(1, 10)))
    random.shuffle(choices)
    return {"type": "mcq", "question": question, "choices": choices, "answer": correct_answer}

def generate_true_false(sentence):
    doc = nlp(sentence)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]
    if not entities:
        return None
    original_entity = random.choice(entities)
    fake_entity = "Google" if original_entity != "Google" else "Microsoft"
    modified_sentence = sentence.replace(original_entity, fake_entity)
    return {"type": "true_false", "question": f"True/False: {modified_sentence}", "answer": "False"}

def generate_short_answer(sentence):
    doc = nlp(sentence)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]
    if not entities:
        return None
    return {"type": "short_answer", "question": f"What is being discussed here: '{sentence}'?", "answer": random.choice(entities)}

# Function to store quiz results
def save_result(topic, score):
    with sqlite3.connect("quiz.db") as conn:
        c = conn.cursor()
        c.execute("INSERT INTO results (topic, score) VALUES (?, ?)", (topic, score))
        conn.commit()

# Function to fetch results
def get_results():
    with sqlite3.connect("quiz.db") as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM results")
        return c.fetchall()

# Function to predict weak area
def predict_weak_area(score):
    return model.predict([[score]])[0]

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Upload route
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file selected")
        return redirect(url_for("home"))
    
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("home"))

    if not file.filename.endswith(".pdf"):
        flash("Only PDF files are allowed")
        return redirect(url_for("home"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    if text.startswith("Error"):
        flash(text)
        return redirect(url_for("home"))

    questions = generate_questions(text)
    return render_template("index.html", questions=questions)

# Quiz submission route
@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    topic = request.form.get("topic", "Uploaded PDF")
    score = request.form.get("score", "0")
    if not score.isdigit():
        return "Invalid score input", 400

    score = int(score)
    save_result(topic, score)
    prediction = predict_weak_area(score)
    flash(f"Quiz submitted! Your weak area prediction: {prediction}")
    return redirect(url_for("home"))

# Results route
@app.route("/results")
def results():
    results = get_results()
    return render_template("index.html", results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)