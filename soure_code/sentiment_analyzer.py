'''
Sentiment Analyzer

@Author: CMSC 170 (Borja, Chambal, Wangli)

'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Directory paths
LABELLED_DATA_DIR = "../labelled_data"

# 1. Load the labeled data
def load_labelled_data(directory):
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith(".xlsx"):
            file_path = os.path.join(directory, file)
            df = pd.read_excel(file_path)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# 2. Preprocess data for BERT
def preprocess_data(df):
    # Ensure we have the necessary columns
    if 'Body' not in df or 'Sentiment' not in df:
        raise ValueError("Data must have 'Body' and 'Sentiment' columns")
    df = df[['Body', 'Sentiment']].dropna()
    df['Sentiment'] = df['Sentiment'].astype(int)  # Ensure sentiment is int
    return df

# 3. Tokenize data for BERT
def tokenize_data(df, tokenizer):
    return tokenizer(list(df['Body']), truncation=True, padding=True, max_length=512)

# 4. Train sentiment analysis model
def train_sentiment_analyzer(data):
    # Split into training and testing
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['Body'], data['Sentiment'], test_size=0.2, random_state=42
    )
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize
    train_encodings = tokenize_data(pd.DataFrame({'Body': train_texts}), tokenizer)
    test_encodings = tokenize_data(pd.DataFrame({'Body': test_texts}), tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': list(train_labels)
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': list(test_labels)
    })

    # Define model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Define Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
    acc = accuracy_score(test_labels, preds)

    print(f"Accuracy: {acc}")
    print(classification_report(test_labels, preds))

    # Save the model
    model.save_pretrained("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")

    return model, tokenizer

# Main function
def main():
    # Load and preprocess data
    labelled_data = load_labelled_data(LABELLED_DATA_DIR)
    processed_data = preprocess_data(labelled_data)

    # Train sentiment analyzer
    train_sentiment_analyzer(processed_data)

if __name__ == "__main__":
    main()