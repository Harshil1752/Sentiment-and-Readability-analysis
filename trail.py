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
    word = word.lower()
    if word.endswith(('es', 'ed')) and not word.endswith(('aes', 'ees', 'oes', 'les', 'mes')):
        word = word[:-2]
    elif word.endswith('e') and not word.endswith(('le', 'me')):
        word = word[:-1]
    syllables = nltk.corpus.cmudict.dict().get(word)
    return len([s for s in syllables[0] if s[-1].isdigit()]) if syllables else 0

def count_syll(word):
    try:
        phones = nltk.corpus.cmudict.dict()[word.lower()]
        return len([phone for phone in phones[0] if phone[-1].isdigit()])
    except KeyError:
        pass
    vowels = re.findall(r'[aeiou]+', word.lower())

    if word.lower().endswith(('es', 'ed')) and len(vowels) > 1 and vowels[-1] == 'e':
        vowels.pop()
    return len(vowels)

def count_syllables1(word):
    vowels = "aeiouy"
    count = 0
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if count == 0:
        count += 1
    return count

url = r"https://insights.blackcoffer.com/what-if-the-creation-is-taking-over-the-creator/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
paragraphs = [p.get_text() for p in soup.find_all('p')]
paragraphs = [word.lower() for  word in paragraphs]
import string
paragraphs = [''.join(char for char in sentence if char not in string.punctuation) for sentence in paragraphs]
token_list = tokenize_texts(paragraphs)
token_list = flatten_2d_list(token_list)
syllable_counts = [count_syllables1(word) for word in token_list]
total_syllable = sum(syllable_counts)

avg_syllable_per_word = total_syllable / len(token_list)
print(avg_syllable_per_word)
new_row = {'SYLLABLE PER WORD': avg_syllable_per_word}
syllable_df = syllable_df.append(new_row, ignore_index = True)
print("Count addded")