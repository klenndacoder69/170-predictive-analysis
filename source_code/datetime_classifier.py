import pandas as pd
from datetime import datetime
import os
# Load the Excel file into a DataFrame
# Replace 'file.xlsx' with the actual path to your Excel file

for file_path in os.listdir('../labelled_data'):
    category = file_path.split('_')[-1].split('.')[0]

    df = pd.read_excel('../labelled_data/' + file_path)

    # Ensure the 'date' column is in datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Define the cutoff dates
    train_end_date = datetime(2024, 5, 31)  # May 31, 2024

    # Create the training dataset (Nov 2022 to May 2024)
    train_df = df[(df['Datetime'] >= datetime(2022, 11, 1)) & (df['Datetime'] <= train_end_date)]

    # Create the testing dataset (June 2024 and beyond)
    test_df = df[df['Datetime'] > train_end_date]

    # Output the resulting DataFrames
    print("Training Dataset:")
    print(train_df)

    print("\nTesting Dataset:")
    print(test_df)

    # Optionally, save the results to new Excel files
    train_df.to_excel(f'../train_and_test/train/train_dataset_{category}.xlsx', index=False)
    test_df.to_excel(f'../train_and_test/test/test_dataset_{category}.xlsx', index=False)