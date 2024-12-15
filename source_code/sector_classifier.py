import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define Categories
categories = ['Agriculture', 'Commerce', 'Finance']

# Function for building the Multinomial Naive Bayes Model
def initialize_MNB_model():
    # Load datasets
    df_agriculture = pd.read_excel(os.path.abspath("../1_Data_Collection/outputs/data_agriculture.xlsx")).dropna(subset=['Body'])
    df_commerce = pd.read_excel(os.path.abspath("../1_Data_Collection/outputs/data_commerce.xlsx"))
    df_finance = pd.read_excel(os.path.abspath("../1_Data_Collection/outputs/data_finance.xlsx")).dropna(subset=['Body'])

    # Combine datasets
    df = pd.concat([df_agriculture, df_commerce, df_finance], ignore_index=True)

    # Text preprocessing
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Body'])
    y = pd.Categorical(df['Category']).codes

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=170)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=categories)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_matrix)

    return model, vectorizer




# Uncomment the whole code below this line to initialize the model and use the prediction functionality
##############################################################################################################

# # Predicting a new article

# # Function to classify a new article
# def classify_new_article(model, vectorizer, new_article_text):
#     article_vector = vectorizer.transform([new_article_text])
#     predicted_category = model.predict(article_vector)[0]
#     return predicted_category


# # main

# # initialize the model
# model, vectorizer = initialize_MNB_model()

# # Example prediction
# new_article_text = '''
# !!! PASTE ARTICLE BODY HERE !!!
# '''

# predicted_category = classify_new_article(model, vectorizer, new_article_text)
# print(f"The predicted category for the new article is: {categories[predicted_category]}")