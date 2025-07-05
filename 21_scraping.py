# %%
!pip install BeautifulSoup4

# %%
import os
import json
import time
import random
import zipfile
import requests
import pandas as pd
from bs4 import BeautifulSoup

# %% [markdown]
# # Class Explanation: `NewsScraper`
# 
# ## Overview
# The `NewsScraper` class is designed for scraping news articles from three different Urdu news websites: Geo, Jang, and Express. The class has methods that cater to each site's unique structure and requirements. Below, we will go through the class and its methods, detailing what each function does, the input it takes, and the output it returns.
# 
# ## Class Definition
# 
# ```python
# class NewsScraper:
#     def __init__(self, id_=0):
#         self.id = id_
# ```
# 
# 
# ## Method 1: `get_express_articles`
# 
# ### Description
# Scrapes news articles from the Express website across categories like saqafat (entertainment), business, sports, science-technology, and world. The method navigates through multiple pages for each category to gather a more extensive dataset.
# 
# ### Input
# - **`max_pages`**: The number of pages to scrape for each category (default is 7).
# 
# ### Process
# - Iterates over each category and page.
# - Requests each category page and finds article cards within `<ul class='tedit-shortnews listing-page'>`.
# - Extracts the article's headline, link, and content by navigating through `<div class='horiz-news3-caption'>` and `<span class='story-text'>`.
# 
# ### Output
# - **Returns**: A tuple of:
#   - A Pandas DataFrame containing columns: `id`, `title`, and `link`).
#   - A dictionary `express_contents` where the key is the article ID and the value is the article content.
# 
# ### Data Structure
# - Article cards are identified by `<li>` tags.
# - Content is structured within `<span class='story-text'>` and `<p>` tags.
# 
# 

# %%
class NewsScraper:
    def __init__(self,id_=0):
        self.id = id_


  # write functions to scrape from other websites
    def get_Geo_articles(self, max_pages=7):
        geo_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
        }
        base_url = 'https://urdu.geo.tv/'
        categories = ['entertainment', 'business', 'sports', 'science-technology', 'world']   # saqafat is entertainment category
        
        # Iterating over the specified number of pages
        for category in categories:
            # for page in range(1, max_pages + 1):
                
                print(f"Scraping of category '{category}'...")
                
                url = f"{base_url}/category/{category}"                
                response = requests.get(url)
                if response.url != url:  # Check if redirection occurred
                    print(f"Redirected from {url} to {response.url}")
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Finding article cards
                cards = soup.find_all('li', class_='border-box')
                print(f"\t--> Found {len(cards)} articles.")

                success_count = 0

                for card in cards:
                    try:
                        
                        div = card.find('a', class_='open-section')  # You are directly searching for the <a> tag

                        if not div:
                            continue
                        
                        # Extract Article Title
                        headline = card.find('a')['title'].strip().replace('\xa0', ' ')
                        
                        # Extract Article Link (href)
                        link = div['href']
                        
                        if not link:
                            continue

                        # Requesting the content from each article's link
                        article_response = requests.get(link)
                        article_response.raise_for_status()
                        content_soup = BeautifulSoup(article_response.text, "html.parser")


                        # Content arranged in paras inside <span> tags
                        paras = content_soup.find('div', class_='content-area').find_all('p')

                        combined_text = " ".join(
                        p.get_text(strip=True).replace('\xa0', ' ').replace('\u200b', '')
                        for p in paras if p.get_text(strip=True)
                        )

                        # Storing data
                        geo_df['id'].append(self.id)
                        geo_df['title'].append(headline)
                        geo_df['link'].append(link)
                        geo_df['gold_label'].append(category.replace('science','science-technology'))
                        geo_df['content'].append(combined_text)

                        # Increment ID and success count
                        self.id += 1
                        success_count += 1

                    except Exception as e:
                        print(f"\t--> Failed to scrape an article on {category}': {e}")

                print(f"\t--> Successfully scraped {success_count} articles of '{category}'.")
                time.sleep(1)

        return pd.DataFrame(geo_df)
        
    def get_Jang_articles(self, max_pages=7):
        
        Jang_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
        }
        base_url = 'https://jang.com.pk'
        categories = ['entertainment', 'business', 'sports', 'health-science',
                      'world']        
         # Iterating over the specified number of pages
        for category in categories:
            
            url = f"{base_url}/category/latest-news/{category}"
                        
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Finding article cards
            cards = soup.find_all('div', class_='main-pic')
            print(f"\t--> Found {len(cards)} articles on  of '{category}'.")

            success_count = 0

            for card in cards:
                try:
                    
                    div = card.find('div', class_='main-pic')

                    # For the headline, get text from the 'a' tag
                    headline = card.find('a')['title'].strip().replace('\xa0', ' ')

                    # For the article link
                    link = card.find('a')['href']
                    
                    # Requesting the content from each article's link
                    article_response = requests.get(link)
                    article_response.raise_for_status()
                    content_soup = BeautifulSoup(article_response.text, "html.parser")


                    # Content arranged in paras inside <span> tags
                    paras = content_soup.find('div',class_='detail_view_content').find_all('p')

                    combined_text = " ".join(
                    p.get_text(strip=True).replace('\xa0', ' ').replace('\u200b', '')
                    for p in paras if p.get_text(strip=True)
                    )

                    # Storing data
                    Jang_df['id'].append(self.id)
                    Jang_df['title'].append(headline)
                    Jang_df['link'].append(link)
                    Jang_df['gold_label'].append(category.replace('health-science','science-technology'))
                    Jang_df['content'].append(combined_text)

                    # Increment ID and success count
                    self.id += 1
                    success_count += 1

                except Exception as e:
                    print(f"\t--> Failed to scrape an article on  of '{category}': {e}")
            time.sleep(1)

            print(f"\t--> Successfully scraped {success_count} articles from of '{category}'.")

        return pd.DataFrame(Jang_df)
        
    def get_ARY_articles(self, max_pages=7):
        
        ARY_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
        }
        
        
        base_url = 'https://urdu.arynews.tv' 
        categories = ['fun-o-sakafat', 'کاروباری-خبریں', 'sports-2', 'سائنس-اور-ٹیکنالوجی', 'international-2']   # saqafat is entertainment category
        
        for category in categories:
            
                count = 0

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0 Edg/118.0",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://urdu.arynews.tv"
                }

                url = f"{base_url}/category/{category}"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Finding article cards
                print("Getting the cards")
                
                cards = soup.find_all('div', class_='td-module-meta-info')
                success_count = 0

                for card in cards:
                    
                    
                    try:
                        
                        h3 = card.find('h3', class_='entry-title td-module-title')

                        # Article Title
                        headline = h3.find('a').get_text(strip=True).replace('\xa0', ' ')

                        # Article link
                        link = h3.find('a')['href']

                        # Requesting the content from each article's link
                        
                        article_response = requests.get(link, headers=headers)
                        article_response.raise_for_status()
                        content_soup = BeautifulSoup(article_response.text, "html.parser")

                        # Content arranged in paras inside <span> tags
                        paras = content_soup.find_all('p')
                                                
                        combined_text = " ".join(
                        p.get_text(strip=True).replace('\xa0', ' ').replace('\u200b', '')
                        for p in paras if p.get_text(strip=True)
                        )
                        
                        print(combined_text)

                        # Storing data
                        ARY_df['id'].append(self.id)
                        ARY_df['title'].append(headline)
                        ARY_df['link'].append(link)
                        ARY_df['gold_label'].append(category.replace('fun-o-sakafat','entertainment').replace('سائنس-اور-ٹیکنالوجی','science-technology').replace('international-2','world').replace('کاروباری-خبریں','business').replace('sports-2','sports'))
                        ARY_df['content'].append(combined_text)

                        # Increment ID and success count
                        self.id += 1
                        success_count += 1

                    except Exception as e:
                        print(f"\t--> Failed to scrape an article on of '{category}': {e}")

                print(f"\t--> Successfully scraped {success_count} articles from '{category}'.")
          

        return pd.DataFrame(ARY_df)   
        
    def get_Duniya_articles(self, max_pages=7):
        dawn_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
        }
        base_url = 'https://www.dawn.com'
        categories = ['saqafat', 'business', 'sports', 'science', 'world']   # saqafat is entertainment category

    def get_express_articles(self, max_pages=8):
        express_df = {
            "id": [],
            "title": [],
            "link": [],
            "content": [],
            "gold_label": [],
        }
        base_url = 'https://www.express.pk'
        categories = ['saqafat', 'business', 'sports', 'science', 'world']   # saqafat is entertainment category

        # Iterating over the specified number of pages
        for category in categories:
            for page in range(1, max_pages + 1):
                print(f"Scraping page {page} of category '{category}'...")
                url = f"{base_url}/{category}/archives?page={page}"
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Finding article cards
                cards = soup.find('ul', class_='tedit-shortnews listing-page').find_all('li')  # Adjust class as per actual site structure
                print(f"\t--> Found {len(cards)} articles on page {page} of '{category}'.")

                success_count = 0

                for card in cards:
                    try:
                        div = card.find('div',class_='horiz-news3-caption')

                        # Article Title
                        headline = div.find('a').get_text(strip=True).replace('\xa0', ' ')

                        # Article link
                        link = div.find('a')['href']

                        # Requesting the content from each article's link
                        article_response = requests.get(link)
                        article_response.raise_for_status()
                        content_soup = BeautifulSoup(article_response.text, "html.parser")


                        # Content arranged in paras inside <span> tags
                        paras = content_soup.find('span',class_='story-text').find_all('p')

                        combined_text = " ".join(
                        p.get_text(strip=True).replace('\xa0', ' ').replace('\u200b', '')
                        for p in paras if p.get_text(strip=True)
                        )

                        # Storing data
                        express_df['id'].append(self.id)
                        express_df['title'].append(headline)
                        express_df['link'].append(link)
                        express_df['gold_label'].append(category.replace('saqafat','entertainment').replace('science','science-technology'))
                        express_df['content'].append(combined_text)

                        # Increment ID and success count
                        self.id += 1
                        success_count += 1

                    except Exception as e:
                        print(f"\t--> Failed to scrape an article on page {page} of '{category}': {e}")

                print(f"\t--> Successfully scraped {success_count} articles from page {page} of '{category}'.")
            print('')

        return pd.DataFrame(express_df)

# %%
scraper = NewsScraper()

# %%
express_df = scraper.get_express_articles() # Confirm Correct
geo_df = scraper.get_Geo_articles() # Looks Correct
Jang_df = scraper.get_Jang_articles() # Looks correct

# %% [markdown]
# # Output
# - Save a combined csv of all 3 sites.

# %%
print(express_df.shape) # 300 something
print(geo_df.shape) # 160
# print(ARY_df.shape) # 50
print(Jang_df.shape)   # 297

combined_df = pd.concat([express_df, geo_df, Jang_df], ignore_index=True)

combined_df.drop_duplicates(subset='link', keep='first', inplace=True)

combined_df.reset_index(drop=True, inplace=True)

csv_filename = 'news_articles_combined.csv'
combined_df.to_csv(csv_filename, index=False)


