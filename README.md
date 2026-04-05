# Google Play Store Reviews - NLP Analysis

End-to-end NLP pipeline built on 12,495 Google Play Store reviews.

## Overview
This project applies core NLP techniques to analyse customer sentiment and discover hidden topics in app reviews. It compares a rule-based approach (VADER) against a trained machine learning model (Logistic Regression).

## What's covered
- Text cleaning and preprocessing
- Tokenization, stopword removal and lemmatization (NLTK)
- VADER sentiment analysis (58% accuracy, no training required)
- Logistic Regression with TF-IDF + bigrams (72% accuracy)
- Class imbalance handling with `class_weight='balanced'`
- LDA topic modelling — 7 discovered topics
- Visualisations: WordCloud, topic bar charts, rating distribution

## Dataset
[Google Play Store Reviews](https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews) via Kaggle — 12,495 reviews across multiple apps.

## Key findings
- VADER underperforms on short technical app reviews vs longer expressive reviews
- Bigrams significantly improved classification by capturing phrases like "keeps crashing" and "not good"
- Pricing emerged as a major complaint topic among users

## Libraries
- pandas, numpy
- nltk
- scikit-learn
- matplotlib
- wordcloud

## Related project
[Sephora Reviews NLP Analysis](https://github.com/GreenBlueberryKate/bluberrybush) — same NLP pipeline applied to beauty product reviews for comparison.

## Author
GreenBlueberryKate
