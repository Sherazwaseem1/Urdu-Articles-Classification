# %%
!pip3 install pandas transformers

# %%
import pandas as pd
import re
from transformers import AutoTokenizer

# Install dependencies
!pip3 install openpyxl urduhack

from urduhack.normalization import normalize
from urduhack.stop_words import STOP_WORDS


# %%

# Load data
file_path = 'news_article_combined.xlsx'  
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"File {file_path} not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Data cleaning
data = data.drop_duplicates()
data = data.dropna(subset=['title', 'content', 'gold_label'])
data['gold_label'] = data['gold_label'].replace({'science-technology-technology': 'science-technology'})

def clean_text(text):
    text = normalize(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return ' '.join(filtered_words)

data['content'] = data['content'].apply(clean_text).apply(remove_stopwords)
data['title'] = data['title'].apply(clean_text).apply(remove_stopwords)

# Save cleaned data
cleaned_file_path = 'cleaned_news_articles.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained("hadidev/gpt2-urdu-tokenizer-withgpt2")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

data['tokenized_content'] = data['content'].apply(
    lambda x: tokenizer.encode(x, truncation=True, max_length=512)
)
data['token_length'] = data['tokenized_content'].apply(len)

tokenized_file_path = 'tokenized_news_articles.csv'
data.to_csv(tokenized_file_path, index=False)
print(f"Tokenized data saved to {tokenized_file_path}")

print(data[['content', 'tokenized_content', 'token_length']].head())



