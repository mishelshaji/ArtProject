from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
# Example reviews
reviews = [
    'i love this product',
]

# Initialize VADER sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis
for review in reviews:
    scores = analyzer.polarity_scores(review)
    print(f"Review: {review} --> Sentiment Scores: {scores}")

    # Interpret the scores
    if scores['compound'] >= 0.05:
        print("Sentiment: Positive")
    elif scores['compound'] <= -0.05:
        print("Sentiment: Negative")
    else:
        print("Sentiment: Neutral")