from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and train model (same as before)
data = pd.read_csv("tickets.csv")
X = data["text"]
y = data["category"]

vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# Create Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        user_input = request.form["text"]
        input_vec = vectorizer.transform([user_input])
        result = model.predict(input_vec)
        prediction = result[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)