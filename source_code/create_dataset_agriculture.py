import requests
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd

# Extract URLs from file storage
article_file = open("Data Collection/articles_agriculture.txt", 'r')
article_file_lines = article_file.readlines()
article_file.close()

# Output file path
output_file = "data_agriculture.xlsx"

# Headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Data to write into Excel
all_data = []

urls = [line.strip() for line in article_file_lines]
for url in urls:
    # Send the GET request
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        # Force UTF-8 encoding
        response.encoding = 'utf-8'

        # Parse the content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract headline
        headline = soup.find('h1', {'class': 'title content-title'}).text.strip()

        # Extract and convert date from meta tag
        date_meta = soup.find('meta', {'property': 'article:published_time'})
        if date_meta:
            date_str = date_meta['content']
            publish_datetime = datetime.fromisoformat(date_str)  # Convert to datetime object
            date = publish_datetime.strftime('%Y-%m-%d %H:%M:%S')
        else:
            date = 'Date not available'

        # Extract news article content
        article_div = soup.find('article')
        article_content = article_div.get_text(separator="\n").replace('\n\n', '\n')
        article_content = article_content[article_content.find('202') + 4 : article_content.rfind('.') + 1].strip()

        # Store the extracted information
        data = {
            'Datetime': date,
            'Category': 'Agriculture',
            'Title': headline,
            'Body': article_content,
            'Outlet': 'Philippine Department of Agriculture',
            'Source': url
        }

        all_data.append(data)

    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")

# Create DataFrame from all extracted data
df = pd.DataFrame(all_data)

# Write DataFrame to Excel
df.to_excel(output_file, index=False)
