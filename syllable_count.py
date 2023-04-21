import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import cmudict as cmd
from nltk.corpus import stopwords
import string
import re

def flatten_2d_list(my_2d_list):
    return [item for sublist in my_2d_list for item in sublist]

def tokenize_texts(texts):
    token_list = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        token_list.append(tokens)
    return token_list

def count_syllables(word):
    vowels = "aeiouy"
    count = 0
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    # Exceptions for words ending with "es" and "ed"
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if word.endswith("es") or word.endswith("ed"):
        if word[-3:-1] in vowels:
            count -= 1
    # Handling single-letter words and words with no vowels
    if count == 0 and len(word) > 0:
        count = 1
    return count

syllable_df = pd.DataFrame(columns=['SYLLABLE PER WORD'])

df = pd.read_csv(r"C:\Users\Harshil\Desktop\Blackcoffers\inputdata.csv")
df = df.dropna()
urls = df['URL'].tolist()

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    paragraphs = [word.lower() for  word in paragraphs]
    import string
    paragraphs = [''.join(char for char in sentence if char not in string.punctuation) for sentence in paragraphs]
    token_list = tokenize_texts(paragraphs)
    token_list = flatten_2d_list(token_list)
    syllable_counts = [count_syllables(word) for word in token_list]
    total_syllable = sum(syllable_counts)
    avg_syllable_per_word = total_syllable / len(token_list)

    new_row = {'SYLLABLE PER WORD': avg_syllable_per_word}

    syllable_df = syllable_df.append(new_row, ignore_index = True)
    print("Count addded")

import openpyxl
syllable_df.to_excel('syllable.xlsx')