# sentiment_analysis.py
import joblib

import joblib

# Load the Decision Tree model from the saved file
try:
    DecisionTreeClassifier = joblib.load('C:/Users/Marsel/Desktop/hdy/Sentiment_Analysis/decisiontree.sav')
    vectorizer = joblib.load('C:/Users/Marsel/Desktop/hdy/Sentiment_Analysis/vectorizer.sav')
except FileNotFoundError:
    print("The file 'decisiontree.sav' was not found. Please provide the correct path.")
    #DecisionTreeClassifier = joblib.load('decisiontree.sav')


def analyze_sentiment(review):
    review_features = vectorizer.transform([review])
    prediction = DecisionTreeClassifier.predict(review_features)
    return prediction[0]