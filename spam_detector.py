import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "message": [
        "Win money now",
        "Hello how are you",
        "Free entry in contest",
        "Let's meet tomorrow",
        "Claim your prize now",
        "Are you coming today"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
msg = ["Win a free ticket now"]
msg_vec = vectorizer.transform(msg)
prediction = model.predict(msg_vec)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")
