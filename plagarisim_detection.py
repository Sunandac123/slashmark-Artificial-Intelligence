import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

nltk.download('punkt')
nltk.download('stopwords')

DATASET_PATH = "data"

documents = []
file_names = []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".txt"):

        path = os.path.join(DATASET_PATH, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

            documents.append(text)
            file_names.append(file)

def preprocess(text):

    text = text.lower()

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    filtered_words = [
        word for word in words
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(filtered_words)

processed_docs = [preprocess(doc) for doc in documents]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(processed_docs)

cosine_sim = cosine_similarity(tfidf_matrix)

results = []

threshold = 0.60

for i in range(len(file_names)):
    for j in range(i + 1, len(file_names)):

        cosine_score = cosine_sim[i][j]

        fuzzy_score = fuzz.ratio(
            processed_docs[i],
            processed_docs[j]
        ) / 100

        avg_score = (cosine_score + fuzzy_score) / 2

        plagiarism = "Yes" if avg_score >= threshold else "No"

        results.append({
            "File 1": file_names[i],
            "File 2": file_names[j],
            "Cosine Similarity": round(cosine_score, 2),
            "Fuzzy Similarity": round(fuzzy_score, 2),
            "Average Score": round(avg_score, 2),
            "Plagiarism": plagiarism
        })

df = pd.DataFrame(results)

df.to_csv("report.csv", index=False)

print("\nPlagiarism Detection Report\n")
print(df)

print("\nReport saved as report.csv")