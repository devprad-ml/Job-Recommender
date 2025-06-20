from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_recommender(df):
    df = df.dropna(subset=['Title', 'JobDescription'])
    df['combined_text'] = df['Title'].fillna('') + " " + df['JobDescription'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return df, vectorizer, tfidf_matrix

def recommend_jobs(df, vectorizer, tfidf_matrix, job_title, job_description, top_k=10):
    user_input = job_title.strip() + " " + job_description.strip()
    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices][['Title', 'JobDescription']].copy()
    results['Score'] = similarity_scores[top_indices]
    return results
