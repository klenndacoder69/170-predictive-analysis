import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os

# Try to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
model_path = "sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# Define batch size
BATCH_SIZE = 16  # Adjust this value based on your GPU capacity

# Load a dataset (change the file path if analyzing a different dataset)
for file_path in os.listdir("../train_and_test/test"):
    print(file_path)
    category = file_path.split('_')[-1].split('.')[0]
    
    df = pd.read_excel("../train_and_test/test/" + file_path)

    # Check if the necessary column exists
    if 'Body' not in df.columns:
        raise ValueError("The dataset must have a 'Body' column for text data.")

    # Prepare text for prediction
    texts = df['Body'].astype(str).tolist()  # Ensure text data is in string format

    # Store probabilities
    positive_probs = []

    # Process in batches
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]  # Get a batch of texts

        # Tokenize the text batch
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512  # Truncate long articles
        ).to(device)

        # Get predictions for the batch
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Extract probabilities for positive sentiment
        positive_probs.extend(probabilities[:, 1].tolist())

        # Clear CUDA cache to free memory
        torch.cuda.empty_cache()

    # Add predictions to the DataFrame
    df['Positive Sentiment Probability'] = positive_probs

    # Save results to a new Excel file
    output_path = f"./results/predicted_sentiment_{category}.xlsx"
    df.to_excel(output_path, index=False)

    print(f"Predicted probabilities have been saved to: {output_path}")
