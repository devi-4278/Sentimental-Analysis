import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import shutil

# Function to ensure NLTK data is downloaded
def download_nltk_data():
    nltk_data_path = os.path.join(nltk.data.path[0], 'corpora', 'vader_lexicon')
    
    # Remove existing data if there's an issue
    if os.path.exists(nltk_data_path):
        shutil.rmtree(nltk_data_path)
    
    # Attempt to download the VADER lexicon
    try:
        nltk.download('vader_lexicon')
    except Exception as e:
        print(f"An error occurred while downloading NLTK data: {e}")
        return False
    return True

# Ensure VADER lexicon is downloaded
if not download_nltk_data():
    print("Failed to download VADER lexicon. Please check your internet connection or NLTK setup.")
    exit()

# Create a sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to determine sentiment
def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Test the sentiment analysis function
text = input("Give a sentiment statement: ")
sentiment = analyze_sentiment(text)
print(f"Sentiment: {sentiment}")
