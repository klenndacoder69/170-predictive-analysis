'''
Sentiment Analyzer

@Author: CMSC 170 (Borja, Chambal, Wangli)

'''

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import os
# Function to determine sentiment
def get_sentiment(text):
    if pd.isna(text):  # Handle NaN text
        return 0
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 1
    elif scores['compound'] <= -0.05:
        return 0
    else:
        return 0

# Download VADER lexicon if not already downloaded
download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Iterate over all the files in the directory
for file_path in os.listdir('../data'):
    # The filename will be used later for outputting where to put the file
    filename = file_path.split('_')[-1].split('.')[0]
    # Load the Excel file
    df = pd.read_excel('../data/' + file_path)

    # Apply sentiment analysis to Title and Body columns
    df['Sentiment'] = df.apply(lambda row: get_sentiment(f"{row['Title']} {row['Body']}"), axis=1)

    # Save the updated file
    output_file_path = f'../labelled_data/sentiment_{filename}.xlsx'  # Output file name
    df.to_excel(output_file_path, index=False)

    print(f"Sentiment analysis completed. File saved to {output_file_path}.")

