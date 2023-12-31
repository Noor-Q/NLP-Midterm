import os
import json
import codecs
import pandas as pd
import re
import string
from rake_nltk import Rake
import nltk
#Reading the training set#
directory = 'TestSet'
data = []
# Iterate over each JSON file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        # Read the contents of the JSON file
        with codecs.open(file_path, 'r', encoding='utf-8-sig') as file:
            try:
                json_data = json.load(file)
                # Store the parsed data in the list
                data.append(json_data)
            #We also need to identify any files that have codec issues
            except json.decoder.JSONDecodeError as e:
                continue

df = pd.DataFrame(data)
df.rename(columns = {'Anahtar Kelimeler':'Anahtar_Kelimeler'}, inplace = True)
df.drop(['Tip', 'Dergi İsmi', 'Üniversite İsmi', 'Gün', 'Ay', 'Yıl', 'Cilt', 'Sayı',
        'Title', 'Yazar İsmi', 'Danışman İsmi', 'Keywords','Abstract','Kaynakça'],axis='columns',inplace=True)
#function to remove special character, numbers and Upper case
def clean_Text(words):
    if isinstance(words, str):
        cleaned_word = words.lower().replace('_', ' ')
        cleaned_word = re.sub(r'[^\w ]', '', cleaned_word)
        cleaned_word = re.sub(r'\d', '', cleaned_word)
        return cleaned_word
    else:
        raise ValueError("Invalid input. Expected string or list.")
    
#Use clean Text fucntion to clean the data of special characters
df['Başlık'] = df['Başlık'].astype(str).apply(clean_Text)
df['Özet'] = df['Özet'].astype(str).apply(clean_Text)
df['Metin'] = df['Metin'].astype(str).apply(clean_Text)
df['Anahtar_Kelimeler'] = df['Anahtar_Kelimeler'].apply(lambda x: [word.lower() for word in x])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def remove_turkish_stopwords(text):
  # Load NLTK stopwords
  stop_words = set(stopwords.words('turkish'))
  # Load custom stop words from a text file
  with open('stopturkish.txt', 'r') as file:
    custom_stop_words = [line.strip() for line in file]
  # Combine NLTK stopwords and custom stop words
  stop_words = stop_words.union(custom_stop_words)
  tokens = word_tokenize(text)
  filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
  return ' '.join(filtered_tokens)
# Apply the function to the desired column in the DataFrame
df['Başlık'] = df['Başlık'].astype(str).apply(remove_turkish_stopwords)
df['Özet'] = df['Özet'].astype(str).apply(remove_turkish_stopwords)
df['Metin'] = df['Metin'].astype(str).apply(remove_turkish_stopwords)
df['combined_text'] = df['Başlık'] + ' ' +df['Özet'] + ' ' +df['Metin'] 
# Define a function to apply keyword extraction to each text
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import word_tokenize, ngrams, FreqDist

def extract_keywords(text):
  keywords = []
  tokens = word_tokenize(text)
  fdist = FreqDist(tokens)
  most_common = fdist.most_common(4)
  for word, count in most_common:
    keywords.append(word)
  return keywords
def extract_ngrams(text, n):
  ngrams_list = []
  tokens = word_tokenize(text)
  ngram_tokens = list(ngrams(tokens, n))
  fdist = FreqDist(ngram_tokens)
  most_common = fdist.most_common(4)
  for ngram, count in most_common:
    ngrams_list.append(' '.join(ngram))
  return ngrams_list
# Apply the function to the 'text' column and store the results in a new column 'keywords'
df['keywords'] = df['combined_text'].astype(str).apply(lambda x: extract_keywords(x) + extract_ngrams(x, n=2)+extract_ngrams(x, n=3))
def count_matching_elements(list1, list2):
    matching_count = 0
    for element1 in list1:
        for element2 in list2:
            if element1 == element2:
                matching_count += 1
    return matching_count
df['matching_count'] = [count_matching_elements(list1, list2) for list1, list2 in zip(df['keywords'], df['Anahtar_Kelimeler'])]

TP = df['matching_count'].sum()
elemnet_count = df['Anahtar_Kelimeler'].apply(len)
total=elemnet_count.sum()
accuracy =TP/total*100
print("Accuracy:", accuracy, "%")




"""
total_records = len(df)
keywords_to_match_per_row = 6
total_expected_matches = total_records * keywords_to_match_per_row
matching_records = df['matching_count'].sum()
df['accuracy'] = df['matching_count'] / total_expected_matches
accuracy = df['accuracy'].mean()

def calculate_precision(matches, tp, fp):
  precision = tp / (tp + fp)*100
  return precision

def calculate_recall(matches, tp, fn):
  recall = tp / (tp + fn)*100
  return recall

def calculate_f_score(precision, recall):
  f_score = 2 * (precision * recall) / (precision + recall) *100
  return f_score



tp = matching_records
fp = total_expected_matches - tp
fn = total_expected_matches - tp
precision = calculate_precision(matching_records, tp, fp)
recall = calculate_recall(matching_records, tp, fn)
f_score = calculate_f_score(precision, recall)
print("Accuracy:", accuracy, "%")
print("Precision:", precision)
print("Recall:", recall)
print("F-score:", f_score)"""
