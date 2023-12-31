import os
import json
import codecs
import pandas as pd
import re
import string
from rake_nltk import Rake
import nltk
import warnings
warnings.filterwarnings("ignore")

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
# Now we can create a dataframe from the data
df = pd.DataFrame(data)
#We should rename the columns wirh 2 word names to easily access them
df.rename(columns = {'Anahtar Kelimeler':'Anahtar_Kelimeler'}, inplace = True)
#we assign the keywords as out target dataset and then drop allunnessesary columns
df.drop(['Tip', 'Dergi İsmi', 'Üniversite İsmi', 'Gün', 'Ay', 'Yıl', 'Cilt', 'Sayı',
        'Title', 'Yazar İsmi', 'Danışman İsmi', 'Keywords','Abstract','Kaynakça'],axis='columns',inplace=True)
#function to remove special character, numbers and Upper case
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def clean_Text(words):
    if isinstance(words, str):
        # If it's a single word
        cleaned_word = words.lower().replace('_', ' ')
        cleaned_word = re.sub(r'[^\w ]', '', cleaned_word)
        cleaned_word = re.sub(r'\d', '', cleaned_word)
        return cleaned_word
    else:
        raise ValueError("Invalid input. Expected string or list.")
    
with open('stopturkish.txt', 'r') as file:
    stopwords = [line.strip() for line in file]    
#Use clean Text fucntion to clean the data of special characters
df['Başlık'] = df['Başlık'].astype(str).apply(clean_Text)
df['Özet'] = df['Özet'].astype(str).apply(clean_Text)
df['Metin'] = df['Metin'].astype(str).apply(clean_Text)
df['Anahtar_Kelimeler'] = df['Anahtar_Kelimeler'].apply(lambda x: [word.lower() for word in x])
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

df['combined_text'] = df['Başlık'] + ' ' +df['Özet']

from transformers import AutoModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT

#tr_stop_words = set(stopwords.words('turkish'))
model_name = 'ytu-ce-cosmos/turkish-mini-bert-uncased'
model = AutoModel.from_pretrained(model_name)
kw_model = KeyBERT(model=model)

# Apply the KeyBERT model to the 'combined_text' column of the dataframe
keywords_1 = kw_model.extract_keywords(df['Özet'].tolist(), keyphrase_ngram_range=(1, 1), use_mmr=True, diversity=0.6)
keywords_2 = kw_model.extract_keywords(df['Özet'].tolist(), keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.6)
keywords_3 = kw_model.extract_keywords(df['Özet'].tolist(), keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.6)
# Extract top 4 keywords for each row
keywords_1 = [[word for word, _ in row[:4]] for row in keywords_1]
keywords_2 = [[word for word, _ in row[:4]] for row in keywords_2]
keywords_3 = [[word for word, _ in row[:4]] for row in keywords_3]
# Joining the lists from keywords_1, keywords_2, and keywords_3 into a single column 'Keywords'
df['Keywords'] = [k1 + k2 + k3 for k1, k2, k3 in zip(keywords_1, keywords_2, keywords_3)]

def count_matching_elements(list1, list2):
    matching_count = 0
    for element1 in list1:
        for element2 in list2:
            if element1 == element2:
                matching_count += 1
    return matching_count
df['matching_count'] = [count_matching_elements(list1, list2) for list1, list2 in zip(df['Keywords'], df['Anahtar_Kelimeler'])]

TP = df['matching_count'].sum()
elemnet_count = df['Anahtar_Kelimeler'].apply(len)
total=elemnet_count.sum()
accuracy =TP/total*100
print("Accuracy:", accuracy, "%")

