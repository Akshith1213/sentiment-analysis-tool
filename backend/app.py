from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
import nltk
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

app = Flask(__name__)
# Configure CORS to allow requests from any origin in development
# and from specific origins in production
if os.environ.get('FLASK_ENV') == 'production':
    # In production, specify your Netlify URL
    CORS(app, resources={r"/*": {"origins": os.environ.get('FRONTEND_URL', '*')}})
else:
    # In development, allow all origins
    CORS(app)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')  # Download VADER lexicon for sentiment analysis

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Custom mixed sentiment indicators
MIXED_INDICATORS = [
    'but', 'however', 'although', 'though', 'nevertheless', 
    'nonetheless', 'yet', 'still', 'while', 'despite', 'in spite of',
    'on the other hand', 'even though', 'that said', 'conversely'
]

def enhanced_sentiment_analysis(text):
    """Perform enhanced sentiment analysis using VADER and custom rules"""
    # Get VADER sentiment scores
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    # Check for mixed sentiment indicators
    has_mixed_indicator = any(indicator in text.lower().split() for indicator in MIXED_INDICATORS)
    
    # Determine sentiment based on VADER compound score
    if compound >= 0.05:
        base_sentiment = 'positive'
    elif compound <= -0.05:
        base_sentiment = 'negative'
    else:
        base_sentiment = 'neutral'
    
    # Check for mixed sentiment patterns
    if has_mixed_indicator and -0.5 < compound < 0.5:
        # If there's a mixed indicator and the sentiment is not strongly positive or negative
        sentiment = 'mixed'
    else:
        sentiment = base_sentiment
    
    return {
        'sentiment': sentiment,
        'compound': compound,
        'pos': scores['pos'],
        'neg': scores['neg'],
        'neu': scores['neu'],
        'has_mixed_indicator': has_mixed_indicator
    }

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Overall sentiment analysis with enhanced method
        overall_analysis = enhanced_sentiment_analysis(text)
        overall_sentiment = overall_analysis['sentiment']
        overall_compound = overall_analysis['compound']
        
        # Use NLTK's sentence tokenizer which handles various punctuation properly
        sentences = nltk.sent_tokenize(text)
        
        sentence_analysis = []
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        mixed_count = 0
        
        for sentence in sentences:
            sentence_result = enhanced_sentiment_analysis(sentence)
            sentiment = sentence_result['sentiment']
            compound = sentence_result['compound']
            
            if sentiment == 'positive':
                positive_count += 1
            elif sentiment == 'negative':
                negative_count += 1
            elif sentiment == 'neutral':
                neutral_count += 1
            elif sentiment == 'mixed':
                mixed_count += 1
                
            sentence_analysis.append({
                'text': sentence,
                'sentiment': sentiment,
                'polarity': compound  # Use compound score for consistency
            })
        
        # Check if there's mixed sentiment in the overall text
        has_mixed_sentiment = (
            overall_analysis['has_mixed_indicator'] or 
            mixed_count > 0 or
            (positive_count > 0 and negative_count > 0)
        )
        
        # If the text contains mixed indicators and has both positive and negative elements,
        # override the overall sentiment to "mixed"
        if has_mixed_sentiment and (positive_count > 0 and negative_count > 0):
            overall_sentiment = 'mixed'
        
        return jsonify({
            'overall_sentiment': overall_sentiment,
            'polarity': overall_compound,
            'has_mixed_sentiment': has_mixed_sentiment,
            'sentence_analysis': sentence_analysis,
            'sentiment_counts': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'mixed': mixed_count
            }
        })
    except Exception as e:
        print(f"Error in analyze_sentiment: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Add a root endpoint
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "Sentiment Analysis API",
        "endpoints": {
            "/analyze": "POST - Analyze text sentiment",
            "/health": "GET - Health check"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', debug=False, port=port)