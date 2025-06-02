from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def get_match_score(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(score * 10, 2)

def analyze_keywords(resume_text, job_desc_text):
    jd_keywords = set(re.findall(r'\b\w+\b', job_desc_text.lower()))
    resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
    missing_keywords = jd_keywords - resume_keywords
    matched_keywords = jd_keywords & resume_keywords
    return {"missing_keywords": list(missing_keywords), "matched_keywords": list(matched_keywords)}