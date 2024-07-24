import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Ensure that NLTK stopwords are downloaded
nltk.download('stopwords')

def preprocess_and_split(prompts):
    specifiers = []
    for prompt in prompts:
        parts = prompt.split(',')
        for part in parts:
            # Remove special characters, normalize text, and exclude stopwords
            clean_part = re.sub(r'[^a-zA-Z0-9\s]', '', part).strip().lower()
            clean_part = ' '.join([word for word in clean_part.split() if word not in stop_words])
            if clean_part:
                specifiers.append(clean_part)
    return specifiers
