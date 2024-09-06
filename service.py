import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# Initialize the stemmer
stemmer = PorterStemmer()

def load_slang_dict(file_path="kamus_singkatan.csv"):
    try:
        slang_df = pd.read_csv(file_path)
        # Create a dictionary from the 'singkatan' and 'asli' columns
        return dict(zip(slang_df['singkatan'], slang_df['asli']))
    except FileNotFoundError:
        return {}

def replace_slang(text, slang_dict):
    words = text.split()
    return " ".join([slang_dict.get(word, word) for word in words])

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)

def remove_stopwords(tokens, language="indonesian"):
    stop_words = set(stopwords.words(language))
    return [word for word in tokens if word not in stop_words]

def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

def clean_text(text, slang_dict):
    # Remove URLs
    text = remove_urls(text)
    # Convert text to lowercase
    text = to_lowercase(text)
    # slang remove
    text = replace_slang(text, slang_dict)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation from tokens
    tokens = remove_punctuation(" ".join(tokens)).split()
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    # Apply stemming
    tokens = stem_tokens(tokens)
    # Join tokens back into a string
    return " ".join(tokens)

def preprocess_text(text, slang_dict):
    # Create a dictionary of cleaned steps
    url_removed = remove_urls(text)
    lowercased = to_lowercase(url_removed)
    punctuation_removed = remove_punctuation(lowercased)
    tokens = word_tokenize(punctuation_removed)
    stopwords_removed = remove_stopwords(tokens)
    slang = replace_slang(" ".join(tokens), slang_dict)
    stemmed = stem_tokens(stopwords_removed)
    return {
        "Original": text,
        "URL Removed": url_removed,
        "Lowercased": lowercased,
        "Tokenized": tokens,
        "Punctuation Removed": remove_punctuation(lowercased),
        "Stopwords Removed": " ".join(stopwords_removed),
        "Stemmed": " ".join(stemmed),
        "Slang Normalisasi": slang,
    }
