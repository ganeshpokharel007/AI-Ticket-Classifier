import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Load data
data = pd.read_csv("tickets.csv")

# Features and labels
X = data["text"]
y = data["category"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)



# After training
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
# Test prediction
samples = [
    "Internet not connecting",
    "Laptop screen broken",
    "App keeps crashing"
]

samples_vec = vectorizer.transform(samples)
predictions = model.predict(samples_vec)

for s, p in zip(samples, predictions):
    print(s, "->", p)
