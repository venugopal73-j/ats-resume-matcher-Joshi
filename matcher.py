from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# List of irrelevant or generic terms to exclude
IRRELEVANT_KEYWORDS = {
    'also', 'and', 'are', 'because', 'been', 'being', 'between', 'but', 'every', 'had', 'has', 'have',
    'how', 'however', 'if', 'into', 'is', 'it', 'its', 'know', 'like', 'many', 'may', 'more', 'most',
    'not', 'now', 'on', 'one', 'other', 'our', 'out', 'over', 'own', 'said', 'same', 'she', 'should',
    'since', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these',
    'they', 'this', 'those', 'through', 'too', 'under', 'up', 'upon', 'use', 'using', 'very', 'was',
    'way', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'will', 'with', 'would', 'you'
}

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize into words
    tokens = nltk.word_tokenize(text)

    # Remove stopwords, short words, and custom irrelevant list
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2 and word not in IRRELEVANT_KEYWORDS
    ]

    return ' '.join(filtered_tokens)

def get_match_score(resume_text, job_desc_text):
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_desc_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(score * 10, 2)  # Scale to 10

def analyze_keywords(resume_text, job_desc_text):
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_desc_text)

    # Get unique important keywords from cleaned JD and resume
    jd_keywords = set(cleaned_jd.split())
    resume_keywords = set(cleaned_resume.split())

    missing_keywords = list(jd_keywords - resume_keywords)
    matched_keywords = list(jd_keywords & resume_keywords)

    return {
        "missing_keywords": sorted(missing_keywords),
        "matched_keywords": sorted(matched_keywords)
    }