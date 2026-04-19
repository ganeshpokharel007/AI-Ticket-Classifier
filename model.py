import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression


# Load data
data = pd.read_csv("tickets.csv")

# Features and labels
X = data["text"]
y = data["category"]

# Convert text to numbers
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Create model
model = LogisticRegression(max_iter=200)

# Calculate accuracy (better method)
scores = cross_val_score(model, X_vec, y, cv=3)
print("Average Accuracy:", scores.mean())

# Train model on full data
model.fit(X_vec, y)

# Test with new samples
samples = [
    "Internet not connecting",
    "Laptop screen broken",
    "App keeps crashing"
]

samples_vec = vectorizer.transform(samples)
predictions = model.predict(samples_vec)

for s, p in zip(samples, predictions):
    print(s, "->", p)
