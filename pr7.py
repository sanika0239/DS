import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
documents = [

    "Prachi likes data science and machine learning",
    "Machine learning is used in artificial intelligence",
    "Python is a popular language for data analytics",
    "Students use Python for data visualization and analysis",
    "Text analytics is an important part of data science",
    "Data science helps students understand machine learning",
    "Artificial intelligence and data analytics are related",
    "Python is widely used in text analytics",
    "Students learn data visualization using Python",
    "Machine learning improves prediction systems"
]

print("\n================ SAMPLE DOCUMENTS ================\n")
for i, doc in enumerate(documents):
    print("Document", i+1, ":", doc)
# ============================================================
# STEP 4 : TOKENIZATION
# ============================================================
print("\n================ TOKENIZATION ================\n")
tokens = []
for doc in documents:
    word_tokens = doc.split()
    tokens.append(word_tokens)
    print(word_tokens)
# ============================================================
# STEP 5 : CUSTOM STOPWORDS
# ============================================================
stop_words = {
    'is', 'a', 'the', 'and', 'for',
    'in', 'of', 'to', 'an', 'use',
    'used', 'are'
}
# ============================================================
# STEP 6 : STOPWORD REMOVAL
# ============================================================
print("\n================ STOPWORD REMOVAL ================\n")
filtered_tokens = []
for word_tokens in tokens:
    filtered = [
        word for word in word_tokens
        if word.lower() not in stop_words
    ]
    filtered_tokens.append(filtered)
    print(filtered)
# ============================================================
# STEP 7 : SIMPLE STEMMING
# ============================================================
print("\n================ STEMMING ================\n")
stemmed_words = []
for filtered in filtered_tokens:
    stemmed = []
    for word in filtered:
        word = word.lower()
        if word.endswith("ing"):
            word = word[:-3]
        elif word.endswith("ed"):
            word = word[:-2]
        stemmed.append(word)
    stemmed_words.append(stemmed)
    print(stemmed)
# ============================================================
# STEP 8 : SIMPLE LEMMATIZATION
# ============================================================
print("\n================ LEMMATIZATION ================\n")
lemmatized_words = []
for filtered in filtered_tokens:
    lemmatized = [word.lower() for word in filtered]
    lemmatized_words.append(lemmatized)
    print(lemmatized)
# ============================================================
# STEP 9 : TERM FREQUENCY (TF)
# ============================================================
print("\n================ TERM FREQUENCY ================\n")
word_counts = {}
for doc in documents:
    words = doc.lower().split()
    for word in words:
        if word not in stop_words:
            word_counts[word] = word_counts.get(word, 0) + 1
tf_df = pd.DataFrame({

    'Word': list(word_counts.keys()),
    'Frequency': list(word_counts.values())
})
print(tf_df)
# ============================================================
# STEP 10 : TF-IDF CALCULATION
# ===========================================================
print("\n================ TF-IDF MATRIX ================\n")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)
print(tfidf_df)
# ============================================================
# STEP 13 : VISUALIZATION - PIE CHART
# ============================================================
top_words = tf_df.head(5)
plt.figure(figsize=(7,7))
plt.pie(
    top_words['Frequency'],
    labels=top_words['Word'],
    autopct='%1.1f%%'
)
plt.title("Top 5 Most Frequent Words")
plt.show()