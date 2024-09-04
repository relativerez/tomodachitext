import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Initialize the stemmer
stemmer = PorterStemmer()

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(tokens, language='indonesian'):
    stop_words = set(stopwords.words(language))
    return [word for word in tokens if word not in stop_words]

def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

def clean_text(text):
    # Convert text to lowercase
    text = to_lowercase(text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation from tokens
    tokens = remove_punctuation(' '.join(tokens)).split()
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    # Apply stemming
    tokens = stem_tokens(tokens)
    # Join tokens back into a string
    return ' '.join(tokens)

def preprocess_text(text):
    # Create a dictionary of cleaned steps
    lowercased = to_lowercase(text)
    punctuation_removed = remove_punctuation(lowercased)
    tokens = word_tokenize(punctuation_removed)
    stopwords_removed = remove_stopwords(tokens)
    stemmed = stem_tokens(stopwords_removed)
    return {
        'Original': text,
        'Lowercased': lowercased,
        'Tokenized': tokens,
        'Punctuation Removed': remove_punctuation(lowercased),
        'Stopwords Removed': ' '.join(stopwords_removed),
        'Stemmed': ' '.join(stemmed)
    }
