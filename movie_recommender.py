import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
data = {
    "title": [
        "Nilave",
        "Saiyaara",
        "Dude",
        "Band melam",
        "Rowdy Boys",
        "Vaazha 2"
    ],
    "genre": [
        "action hero",
        "action hero tech",
        "action hero soldier",
        "action hero god",
        "action dark hero",
        "action  hero"
    ]
}

df = pd.DataFrame(data)

# Convert text to numbers
cv = CountVectorizer()
matrix = cv.fit_transform(df["genre"])

# Similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie):
    index = df[df["title"] == movie].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("Recommended movies:")
    for i in scores[1:4]:
        print(df.iloc[i[0]]["title"])

# Test
recommend("Nilave")
