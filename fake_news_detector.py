import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "news": [
        "Government announces new policy",
        "Celebrity caught in fake scandal",
        "Scientists discover new planet",
        "Fake news spreading on social media",
        "New technology launched today",
        "Shocking fake rumor about actor"
    ],
    "label": [0, 1, 0, 1, 0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["news"], df["label"], test_size=0.2)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom news
test = ["Breaking news: new scam detected"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)

if prediction[0] == 1:
    print("Fake News")
else:
    print("Real News")
