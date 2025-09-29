# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 20:50:35 2025

@author: schud
"""

import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import word_tokenize
# import spacy
from tqdm import tqdm

# Example topic seeds
seed_keywords = [
    "sexual violence",
    "rape",
    "assault",
    "harassment",
    "molestation",
    "abuse",
    "intimate partner violence",
    "domestic violence",
]

noninfor = ['united',
           'nations',
           'un',
           'also',
           'would',
           'like']

sessions = np.arange(1, 80)

def preprocess(words):
    sw = stopwords.words("english")
    no_sw = []
    for w in words:
        if (w not in sw):
            if (w not in noninfor):
                if w.isalpha():
                    no_sw.append(w)
    return no_sw


data=[]

for session in sessions:
    directory = f"./UN_speeches_data/Data/TXT/Session {session:02} - {1945 + session}"
    # directory = f"./TXT/Session {session} - {1945+session}"
    for filename in os.listdir(directory):
        # f = open(os.path.join(directory, filename))
        with open(os.path.join(directory, filename), encoding="utf-8") as f:
            if filename[0]==".": #ignore hidden files
                continue
            splt = filename.split("_")
            data.append([session, 1945+session, splt[0], f.read()])

df_speech = pd.DataFrame(data, columns=['Session','Year','ISO-alpha3 Code','Speech'])

df_codes = pd.read_csv("UNSD — Methodology.csv", sep=";")

merged_df = pd.merge(df_codes, df_speech, left_on="ISO-alpha3 Code", right_on="ISO-alpha3 Code", how="inner")

columns = [
    "Country or Area",
    "Region Name",
    "Sub-region Name",
    "ISO-alpha3 Code",
    "Least Developed Countries (LDC)",
    "Session",
    "Year",
    "Speech"
]

df_un_merged = merged_df[columns]
df_un_merged = df_un_merged.set_index(["Year", "ISO-alpha3 Code"])

# #Semantically select our library to look for
# #Get semantics model
# # Load medium spaCy model for embeddings
# import en_core_web_lg
# nlp = en_core_web_lg.load() # or "en_core_web_lg" for larger embeddings

# # Flatten all candidate words from corpus
# candidate_words = set()
# corpus_texts = df_un_merged['Speech']
# for text in tqdm(corpus_texts):
#     candidate_words.update([token.text for token in nlp(text.lower()) if token.is_alpha])

# # Discover related keywords
# discovered_keywords = set(seed_keywords)
# seed_vecs = [nlp(seed) for seed in tqdm(seed_keywords)]

# for word in tqdm(candidate_words):
#     word_vec = nlp(word)
#     for seed_vec in seed_vecs:
#         if seed_vec.similarity(word_vec) > 0.8:  # medium threshold for discovery
#             discovered_keywords.add(word.lower())


