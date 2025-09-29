# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 20:50:35 2025

@author: schud
"""

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import word_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt

# Example topic seeds
violence_keywords_themed = {
    "general violence abuse": [
        "gender-based violence", "violence against women", "physical violence",
        "emotional abuse", "psychological abuse", "verbal abuse", "controlling behavior", 
        # "coercion", "intimidation"
    ],
    "sexual violence exploitation": [
        "sexual assault", "sexual harassment", "sexual coercion", 
        "sexual exploitation", "sex trafficking",
        "grooming", "indecent assault", "unwanted sexual contact",
        "forced sex", "statutory rape"
    ],
    "domestic intimate partner violence": [
        "intimate partner violence", "IPV", "battering",
        "spousal abuse", "marital rape", "coercive control",
        "partner abuse", "relationship violence", "family violence",
        "domestic violence"
    ],
    "stalking harassment": [
        "stalking", "cyberstalking", "online harassment",
        "catcalling", "unwanted advances"
    ],
    "workplace violence or harrassment": [
        "workplace harassment", "quid pro quo harassment",
        "sexual misconduct", "exploitation of authority"
    ],
    # "indicators_consequences": [
    #     "trauma", "PTSD", "survivor", "victim blaming",
    #     "crisis hotline", "protective order", "restraining order",
    #     "shelter", "safety planning"
    # ],
    "hashtags abbreviations movements": [
        "#MeToo", "#TimesUp", "#NiUnaMenos", "#SayHerName", "#NotOneMore",
        "#WhyIDidntReport", "#BelieveWomen", "#YesAllWomen",
        # "SA", 
        "DV", "GBV", "VAW",
        "rape culture", "toxic masculinity", "misogyny",
        "silence breakers", "consent culture"
    ]
}

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
                elif w.startswith('#'):
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

keywords = [term.lower() for terms in violence_keywords_themed.values() for term in terms]

#wordcounter
vectorizer = CountVectorizer(vocabulary=keywords, 
                             ngram_range=(1, 3), 
                             lowercase=True,
                             token_pattern=r"(?u)\b\w[\w#@'-]+\b")
count_mat = vectorizer.fit_transform(df_un_merged["Speech"])

#tfidf vectorizer
tfidfvectorizer = TfidfVectorizer(
    vocabulary=keywords,
    lowercase=True,
    ngram_range=(1, 3),  
    token_pattern=r"(?u)\b\w[\w#@'-]+\b"
)
tfidf_mat = vectorizer.fit_transform(df_un_merged["Speech"])


# Create a DataFrame of word counts
word_counts = pd.DataFrame(
    count_mat.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=df_un_merged.index
)

# create a dataframe of tf_idf
tfidf_scores = pd.DataFrame(
    tfidf_mat.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=df_un_merged.index
)

theme_counts = pd.DataFrame(index=df_un_merged.index)
word_counts = word_counts.loc[:, (word_counts.sum(axis=0) > 0)]

#group themes
for theme, keywords in violence_keywords_themed.items():
    # sum across all matching columns (if the word exists in vocab)
    matching_cols = [w for w in keywords if w in word_counts.columns]
    if matching_cols:
        theme_counts[theme] = word_counts[matching_cols].sum(axis=1)
    else:
        theme_counts[theme] = 0
        
theme_tfidf = pd.DataFrame(index=df_un_merged.index)

for theme, keywords in violence_keywords_themed.items():
    matching_cols = [term.lower() for term in keywords if term.lower() in tfidf_scores.columns]
    theme_tfidf[theme] = tfidf_scores[matching_cols].sum(axis=1)


df_un_merged = pd.merge(df_un_merged, theme_counts, left_on= ["Year", "ISO-alpha3 Code"], right_on= ["Year", "ISO-alpha3 Code"], how="inner")
df_un_merged = pd.merge(df_un_merged, word_counts, left_on= ["Year", "ISO-alpha3 Code"], right_on= ["Year", "ISO-alpha3 Code"], how="inner")
df_un_merged = pd.merge(df_un_merged, theme_tfidf, left_on= ["Year", "ISO-alpha3 Code"], right_on= ["Year", "ISO-alpha3 Code"], how="inner")
df_un_merged = pd.merge(df_un_merged, word_counts, left_on= ["Year", "ISO-alpha3 Code"], right_on= ["Year", "ISO-alpha3 Code"], how="inner")

# Sum and sort values for wordcount and theme count
totalcounts = word_counts.sum(axis=0)
totalcounts_sorted = totalcounts.sort_values(ascending=False)
themetotalcounts = theme_counts.sum(axis=0)
themetotalcounts_sorted = themetotalcounts.sort_values(ascending=False)

# Plot wordcounts and theme counts
plt.figure(0, figsize=(10, 6))
totalcounts_sorted.plot(kind="bar")
plt.title("Keyword Frequencies")
plt.ylabel("Count")
plt.xlabel("Keyword")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot wordcounts and theme counts
plt.figure(1, figsize=(10, 6))
themetotalcounts_sorted.plot(kind="bar")
plt.title("Keytheme Frequencies")
plt.ylabel("Count")
plt.xlabel("Theme")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()








