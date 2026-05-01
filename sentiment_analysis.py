import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    "review": [
        "I love this product",
        "This is amazing",
        "Very bad experience",
        "I hate this",
        "Best purchase ever",
        "Worst product"
    ],
    "sentiment": [1, 1, 0, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test custom input
test = ["This product is awesome"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)

if prediction[0] == 1:
    print("Positive Review 😊")
else:
    print("Negative Review 😡")
