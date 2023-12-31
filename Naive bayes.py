import os
import json
import codecs
import pandas as pd
import numpy as np
import re
import string
import requests
import warnings
warnings.filterwarnings("ignore")
#We use directory to specify the source file and Data array to hold the data 
directory = 'TrainingSet'
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
directory = 'TrainingSet'
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
            
test_df = pd.DataFrame(data)
#We should rename the columns wirh 2 word names to easily access them
df.rename(columns = {'Anahtar Kelimeler':'Anahtar_Kelimeler'}, inplace = True)
#we assign the keywords as out target dataset and then drop allunnessesary columns
df.drop(['Tip', 'Dergi İsmi', 'Üniversite İsmi', 'Gün', 'Ay', 'Yıl', 'Cilt', 'Sayı',
        'Title', 'Yazar İsmi', 'Danışman İsmi', 'Keywords','Abstract','Kaynakça'],axis='columns',inplace=True)
#We should rename the columns wirh 2 word names to easily access them
test_df.rename(columns = {'Anahtar Kelimeler':'Anahtar_Kelimeler'}, inplace = True)
#we assign the keywords as out target dataset and then drop allunnessesary columns
test_df.drop(['Tip', 'Dergi İsmi', 'Üniversite İsmi', 'Gün', 'Ay', 'Yıl', 'Cilt', 'Sayı',
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

test_df['Başlık'] = test_df['Başlık'].astype(str).apply(clean_Text)
test_df['Özet'] = test_df['Özet'].astype(str).apply(clean_Text)
test_df['Metin'] = test_df['Metin'].astype(str).apply(clean_Text)
test_df['Anahtar_Kelimeler'] = test_df['Anahtar_Kelimeler'].apply(lambda x: [word.lower() for word in x])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#Use clean Text fucntion to remove stop words
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
test_df['Başlık'] = test_df['Başlık'].astype(str).apply(remove_turkish_stopwords)
test_df['Özet'] = test_df['Özet'].astype(str).apply(remove_turkish_stopwords)
test_df['Metin'] = test_df['Metin'].astype(str).apply(remove_turkish_stopwords)
test_df['combined_text'] = test_df['Başlık'] + ' ' +test_df['Özet'] + ' ' +test_df['Metin']

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import nltk
from nltk.corpus import stopwords


# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
label_vectors = mlb.fit_transform(df['Anahtar_Kelimeler'])
label_df = pd.DataFrame(label_vectors, columns=mlb.classes_)

test_label_vectors = mlb.fit_transform(test_df['Anahtar_Kelimeler'])
test_label_df = pd.DataFrame(test_label_vectors, columns=mlb.classes_)

v = CountVectorizer(ngram_range=(1, 3), max_features=1000)
x_count = v.fit_transform(df['combined_text'].values)

clf = MultiOutputClassifier(MultinomialNB())

scores = cross_val_score(clf, x_count, test_label_df, cv=5)

# Define the scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall': make_scorer(recall_score, average='weighted', zero_division=1),
    'f1_score': make_scorer(f1_score, average='weighted', zero_division=1)
}

# Perform cross-validation
results = cross_val_score(clf, x_count, test_label_df, scoring='f1_micro', cv=10)
# Print the Micro average scores
average_accuracy = results.mean() * 100
average_precision = results[0].mean() * 100
average_recall = results[1].mean() * 100
average_f1_score = results[2].mean() * 100
print(" Micro Scores")
print(f"Average Accuracy: {average_accuracy:.2f} %")
print(f"Average Precision: {average_precision:.2f} %")
print(f"Average Recall: {average_recall:.2f} %")
print(f"Average F1-score: {average_f1_score:.2f} %")
# Print the Macro average scores
results = cross_val_score(clf, x_count, test_label_df, scoring='f1_macro', cv=10)
average_accuracy = results.mean() * 100
average_precision = results[0].mean() * 100
average_recall = results[1].mean() * 100
average_f1_score = results[2].mean() * 100
print(" Macro Scores")
print(f"Average Accuracy: {average_accuracy:.2f} %")
print(f"Average Precision: {average_precision:.2f} %")
print(f"Average Recall: {average_recall:.2f} %")
print(f"Average F1-score: {average_f1_score:.2f} %")










