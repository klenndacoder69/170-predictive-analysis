import requests
import time
from bs4 import BeautifulSoup


agriculture = {'start_page': 6, 
               'end_page': 67, 
               'url': 'https://www.da.gov.ph/category/news/page/',
               'filter_link': 'https://www.da.gov.ph/',
               'clean_links': lambda link_list: list(dict.fromkeys(link_list[57:57+23]))}

finance = {'start_page': 1, 
           'end_page': 202, 
           'url': 'https://www.bworldonline.com/banking-finance/page/',
           'filter_link': 'https://www.bworldonline.com/banking-finance/202',
           'clean_links': lambda link_list: list(dict.fromkeys(link_list[11:]))[:12]}

production = {'start_page': 1, 
              'end_page': 66, 
              'url': 'https://www.dti.gov.ph/category/archives/news-archives/page/',
              'filter_link': 'https://www.dti.gov.ph/archives/news-archives/',
              'clean_links': lambda link_list: list(dict.fromkeys(link_list))}


sectors = {'agriculture' : agriculture, 'finance' : finance, 'production' : production}


headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }


for s in sectors:
    # print(f'Sector: {s}')
    output_file = open(f'articles_{s}.txt', 'w')
    for i in range(sectors[s]['start_page'], sectors[s]['end_page'] + 1):
        print(f'Page {i}')
        # Send a GET request to the page with a User-Agent header
        url = f"{sectors[s]['url']}{i}/"
        response = requests.get(url, headers=headers, timeout=10)   
        print(f"Response: {response}")

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            print(f"Soup: {soup}")

            # Find all links
            links = soup.find_all('a', href=True)

            # Filter links starting with the filter link
            sector_links = [link['href'] for link in links if link['href'].startswith(sectors[s]['filter_link'])]

            # Store the filtered links
            for link in sectors[s]['clean_links'](sector_links):
                output_file.write(f"{link}\n")
                # print(link)
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
        
        time.sleep(1 if s != 'finance' else 2)
        # print()
    output_file.close()