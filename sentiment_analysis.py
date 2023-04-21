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

#Converting 2D list to 1D list
def flatten_2d_list(my_2d_list):
    return [item for sublist in my_2d_list for item in sublist]

#Importing Stopwords from given folder files
audi_sw = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\StopWords\StopWords_Auditor.txt', 'r') as adu:
    words = adu.read().split('\n')
    audi_sw.append(words)
audi_sw = flatten_2d_list(audi_sw)
audi_sw = [word.lower() for  word in audi_sw]

curr_sw = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\StopWords\StopWords_Currencies.txt', 'r') as cur:
    words = cur.read().split('\n')
    curr_sw.append(words)
curr_sw = flatten_2d_list(curr_sw)
curr_new_sw = []
for string in curr_sw:
    words = string.split('|')
    curr_new_sw.append(words)
curr_sw = flatten_2d_list(curr_new_sw)
curr_sw = [string.replace(" ", "") for string in curr_sw]
curr_sw = [word.lower() for  word in curr_sw]

dateandno_sw = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\StopWords\StopWords_DatesandNumbers.txt', 'r') as dno:
    words = dno.read().split('\n')
    dateandno_sw.append(words)
dateandno_sw = flatten_2d_list(dateandno_sw)
dateandno_sw = [word.lower() for  word in dateandno_sw]

gen_sw = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\StopWords\StopWords_Generic.txt', 'r') as gen:
    words = gen.read().split('\n')
    gen_sw.append(words)
gen_sw = flatten_2d_list(gen_sw)
gen_sw = [word.lower() for  word in gen_sw]

genlo_sw = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\StopWords\StopWords_GenericLong.txt', 'r') as genlo:
    words = genlo.read().split('\n')
    genlo_sw.append(words)
genlo_sw = flatten_2d_list(genlo_sw)
genlo_sw = [word.lower() for  word in genlo_sw]

geo_sw = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\StopWords\StopWords_Geographic.txt', 'r') as geo:
    words = geo.read().split('\n')
    geo_sw.append(words)
geo_sw = flatten_2d_list(geo_sw)
geo_sw = [word.lower() for  word in geo_sw]

name_sw = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\StopWords\StopWords_Names.txt', 'r') as name:
    words = name.read().split('\n')
    name_sw.append(words)
name_sw = flatten_2d_list(name_sw)
name_sw = [word.lower() for  word in name_sw]

#Importing Positive and Negative words dictionary
pos_words = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\MasterDictionary\positive-words.txt', 'r') as file1:
    words = file1.read().split('\n')
    pos_words.append(words)
pos_words = flatten_2d_list(pos_words)
pos_words = [word.lower() for  word in pos_words]

neg_words = []
with open(r'C:\Users\Harshil\Desktop\Blackcoffers\MasterDictionary\negative-words.txt', 'r') as file2:
    words = file2.read().split('\n')
    neg_words.append(words)
neg_words = flatten_2d_list(neg_words)
neg_words = [word.lower() for  word in neg_words]

#Function to remove words from string
def remove_words_from_strings(string_list, words_to_remove):
    clean_strings = []
    for s in string_list:
        words = s.split()
        clean_words = [word for word in words if word not in words_to_remove]
        clean_strings.append(' '.join(clean_words))
    return clean_strings

#Function for text tokenization
def tokenize_texts(texts):
    token_list = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        token_list.append(tokens)
    return token_list

#Function for syllabel count
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

#Function for personal Word count
def personal_pro(text):
    pattern = r'\b(I|we|my|ours|us)\b'
    ex_pattern = r'\b(US)\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    matches = [m for m in matches if not re.match(ex_pattern, m, re.IGNORECASE)]
    count = len(matches)
    return count

#Function for average word Length
def average_word_length(words):
    total_length = sum(len(word) for word in words)
    total_words = len(words)
    if total_words > 0:
        average_length = total_length / total_words
    else:
        average_length = 0

    return average_length

#Importing urls in Dataframe
df = pd.read_csv(r"C:\Users\Harshil\Desktop\Blackcoffers\inputdata.csv")
df = df.dropna()
urls = df['URL'].tolist()

#Creating Output Dataframe to store  results
output_df = pd.DataFrame(columns=['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                                   'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
                                     'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                                       'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])

#Running each individual url through the process
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.find('title').text
    #paragraphs = soup.find_all('p')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    paragraphs2 = [word.lower() for  word in paragraphs]

    #Cleaning using Stop Words Lists
    filter_para = remove_words_from_strings(paragraphs, gen_sw)
    filter_para = remove_words_from_strings(filter_para, audi_sw)
    filter_para = remove_words_from_strings(filter_para, curr_sw)
    filter_para = remove_words_from_strings(filter_para, dateandno_sw)
    filter_para = remove_words_from_strings(filter_para, genlo_sw)
    filter_para = remove_words_from_strings(filter_para, geo_sw)
    filter_para = remove_words_from_strings(filter_para, name_sw)
    import string
    filter_para = [''.join(char for char in sentence if char not in string.punctuation) for sentence in filter_para]
    token_list = tokenize_texts(filter_para)
    #Extracting Derived variables
    positive_score = 0
    negative_score = 0

    for tokens in token_list:
        for token in tokens:
            if token.lower() in pos_words:
                positive_score += 1
            elif token.lower() in neg_words:
                negative_score += 1

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    total_words = sum(len(tokens) for tokens in token_list)
    subjectivity_score = (positive_score + negative_score)/ (total_words + 0.000001)

    #Analysis of Readability
    word_dict = cmd.dict()
    no_words = 0
    no_com_words = 0
    no_sen = 0

    token_list2 = tokenize_texts(paragraphs2)
    for tokens in token_list2:
        no_words += len(tokens)
        no_sen += 1
        for token in tokens:
            token = token.translate(str.maketrans('', '', string.punctuation))
            if token.lower() in word_dict:
                no_syllabes = [len(list(y for y  in x if y[-1].isdigit())) for x in word_dict[token.lower()]][0]
                if no_syllabes >= 2:
                    no_com_words += 1

    avg_sen_len = no_words / no_sen
    per_com_words = no_com_words / no_words
    fog_index = 0.4*(avg_sen_len + per_com_words)

    #Average Number of Words Per Sentence
    total_no_words = 0
    total_no_sentences = 0
    for text in paragraphs2:
        sentences = nltk.sent_tokenize(text)
        num_sentences = len(sentences)

        tok = nltk.word_tokenize(text)
        num_words = len(tok)

        total_no_words += num_words
        total_no_sentences += num_sentences    
    avg_no_word_per_sen = total_no_words / total_no_sentences

    #Word Count
    paragraphs3 = [''.join(char for char in sentence if char not in string.punctuation) for sentence in paragraphs2]
    stop_words = set(stopwords.words('english'))
    paragraphs3 = [nltk.word_tokenize(sentences) for sentences in paragraphs3]
    paragraphs3 = [[word for word in sentences if word.lower() not in stop_words] for  sentences in paragraphs3]
    paragraphs3 = flatten_2d_list(paragraphs3)
    word_count = len(paragraphs3)

    #Average Syllable count per word
    token_list = flatten_2d_list(token_list)
    syllable_counts = [count_syllables(word) for word in token_list]
    total_syllable = sum(syllable_counts)
    avg_syllable_per_word = total_syllable / len(token_list)

    #Personal Pronouns
    personal_pro_list = []
    for text in paragraphs:
        count_pro = personal_pro(text)
        personal_pro_list.append(count_pro)
    total_pronoun = sum(personal_pro_list)
    avg_word_len = average_word_length(paragraphs3)
    
    #Adding all the output to the output Dataframe 
    new_row = {'POSITIVE SCORE': positive_score, 'NEGATIVE SCORE': negative_score, 'POLARITY SCORE': polarity_score,
                'SUBJECTIVITY SCORE': subjectivity_score, 'AVG SENTENCE LENGTH': avg_sen_len,
                  'PERCENTAGE OF COMPLEX WORDS': per_com_words, 'fog_index': fog_index,
                    'AVG NUMBER OF WORDS PER SENTENCE': avg_no_word_per_sen, 'COMPLEX WORD COUNT': no_com_words,
                      'WORD COUNT': word_count,'SYLLABLE PER WORD':avg_syllable_per_word, 'PERSONAL PRONOUNS': total_pronoun,
                       'AVG WORD LENGTH': avg_word_len}
    
    output_df = output_df.append(new_row, ignore_index=True)


import openpyxl
output_df.to_excel('Output_structure.xlsx')