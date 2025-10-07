"""
Text preprocessing module for ATS Resume Classifier
This module must be separate to ensure pickle compatibility with Gunicorn
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)


class ImprovedTextPreprocessor:
    """
    Enhanced text preprocessing for resume and job description text.
    This class must be importable for pickle deserialization.
    """
    def __init__(self):
        try:
            # Use a smaller set of stopwords to preserve technical context
            self.stop_words = set(stopwords.words('english')) - {
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
                'over', 'under', 'again', 'further', 'then', 'once'
            }
        except Exception as e:
            logger.warning(f"Could not load stopwords: {e}. Using empty set.")
            self.stop_words = set()
    
    def clean_text(self, text):
        """Clean and preprocess text while preserving technical terms"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep alphanumeric, spaces, and important symbols like +, #, .
        text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text):
        """Tokenize and remove only stopwords, NO stemming"""
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove only stopwords, keep all meaningful terms
            tokens = [token for token in tokens 
                      if token not in self.stop_words and len(token) > 1]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}. Returning cleaned text.")
            return text
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_filter(text)
        return text