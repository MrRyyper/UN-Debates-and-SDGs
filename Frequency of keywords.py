# -*- coding: utf-8 -*-
"""
Analyze UN General Assembly speech texts for the presence of gender-based
violence-related keywords and themes.

Workflow overview:
- Define theme dictionaries (keywords grouped by topic) and flatten to a keyword list.
- Load and join UN speech texts with UNSD country metadata.
- Vectorize speeches using CountVectorizer (counts) and TfidfVectorizer (tf-idf).
- Aggregate at the theme level by summing term columns per theme.
- Concatenate all derived features onto the main frame by aligned index.
- Produce overview bar charts and time-series plots (counts and tf-idf).
"""

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Define theme → keywords mapping and derive a lowercase keyword list
violence_keywords_themed = {
    "general violence abuse": [
        "gender based violence", "violence against women", "physical violence",
        "emotional abuse", "psychological abuse", "verbal abuse", "controlling behavior", 
        # "coercion", "intimidation"
    ],
    "sexual violence exploitation": [
        "sexual assault", "sexual harassment", "sexual coercion", 
        "sexual exploitation", "sex trafficking", "sexual violence",
        "grooming", "indecent assault", "unwanted sexual contact",
        "forced sex", "statutory rape", "rape"
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
    "workplace violence or harassment": [
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
keywords = [term.lower() for terms in violence_keywords_themed.values() for term in terms]

# Session numbering helper
sessions = np.arange(1, 80)

#%%
# Load speech texts for each session directory; build row list
# Each row: [Session, Year, ISO alpha-3 code derived from filename, full speech text]

data=[]

for session in sessions:
    directory = f"./UN_speeches_data/Data/TXT/Session {session:02} - {1945 + session}"
    if not os.path.isdir(directory):
        continue
    for filename in os.listdir(directory):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, encoding="utf-8") as f:
            splt = filename.split("_")
            data.append([session, 1945 + session, splt[0], f.read()])

# Convert rows to DataFrame and join to UNSD metadata

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

# Use multi-index to facilitate aligned concatenations later

df_un_merged = merged_df[columns]

df_un_merged = df_un_merged.set_index(["Year", "ISO-alpha3 Code"])

#%%
# Vectorize speeches to obtain (1) keyword counts, (2) tf-idf scores

# Counts restricted to our keyword vocabulary
vectorizer = CountVectorizer(vocabulary=keywords, 
                             ngram_range=(1, 3), 
                             lowercase=True,
                             token_pattern=r"(?u)\b\w[\w#@'-]+\b")
count_mat = vectorizer.fit_transform(df_un_merged["Speech"])

# Full tf-idf to later select only keyword columns

tfidfvectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),  
    token_pattern=r"(?u)\b\w[\w#@'-]+\b",
)

tfidf_mat = tfidfvectorizer.fit_transform(df_un_merged["Speech"])

# Build DataFrames for counts and keyword-filtered tf-idf

word_counts = pd.DataFrame(
    count_mat.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=df_un_merged.index
)

feature_names = tfidfvectorizer.get_feature_names_out()

# Map keywords to tf-idf column indices and extract

keywords_to_idx = {word: i for i, word in enumerate(feature_names) if word in keywords}

selected_tfidf_mat = tfidf_mat[:, list(keywords_to_idx.values())]

tfidf_scores = pd.DataFrame(
    selected_tfidf_mat.toarray(), 
    index=df_un_merged.index,
    columns=list(keywords_to_idx.keys())
)

# Retain only columns that occur at least once (avoid all-zero columns)

word_counts = word_counts.loc[:, (word_counts.sum(axis=0) > 0)]

tfidf_scores = tfidf_scores[[col for col in tfidf_scores.columns if col in keywords ]]

tfidf_scores = tfidf_scores.loc[:, (tfidf_scores.sum(axis=0) > 0)]

#%%
# Aggregate per-theme: sum keyword columns belonging to each theme

theme_counts = pd.DataFrame(index=df_un_merged.index)

for theme, kw in violence_keywords_themed.items():
    matching_cols = [w.lower() for w in kw if w.lower() in word_counts.columns]
    theme_counts[theme] = word_counts[matching_cols].sum(axis=1) if matching_cols else 0
        

theme_tfidf = pd.DataFrame(index=df_un_merged.index)

for theme, kw in violence_keywords_themed.items():
    matching_cols = [term.lower() for term in kw if term.lower() in tfidf_scores.columns]
    theme_tfidf[theme] = tfidf_scores[matching_cols].sum(axis=1) if matching_cols else 0

# Concatenate all derived features with explicit suffixes

theme_counts = theme_counts.add_suffix('_counts')
word_counts = word_counts.add_suffix('_counts')

theme_tfidf = theme_tfidf.add_suffix('_tfidf')
tfidf_scores = tfidf_scores.add_suffix('_tfidf')

df_un_merged = pd.concat([df_un_merged, theme_counts, word_counts, theme_tfidf, tfidf_scores], axis=1)

#%%
# Summaries used for bar charts

totalcounts_sorted = word_counts.sum(axis=0).sort_values(ascending=False)

themetotalcounts_sorted = theme_counts.sum(axis=0).sort_values(ascending=False)

# tf-idf summaries

totaltfidf_sorted = tfidf_scores.sum(axis=0).sort_values(ascending=False)

themetotaltfidf_sorted = theme_tfidf.sum(axis=0).sort_values(ascending=False)

#%%
# Overview bar charts

plt.figure(0, figsize=(10, 6))
totalcounts_sorted.plot(kind="bar")
plt.title("Keyword Frequencies")
plt.ylabel("Count")
plt.xlabel("Keyword")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(1, figsize=(10, 6))
themetotalcounts_sorted.plot(kind="bar")
plt.title("Keytheme Frequencies")
plt.ylabel("Count")
plt.xlabel("Theme")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot tf-idf scores and theme tf-idf scores
plt.figure(2, figsize=(10, 6))
totaltfidf_sorted.plot(kind="bar")
plt.title("Keyword tfidf")
plt.ylabel("tfidf score")
plt.xlabel("Keyword")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(3, figsize=(10, 6))
themetotaltfidf_sorted.plot(kind="bar")
plt.title("Keytheme tfidf")
plt.ylabel("tfidf score")
plt.xlabel("Theme")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#%%
# Time series of theme counts and theme tf-idf per year

# Build theme column lists

theme_cols = [theme+'_counts' for theme in list(violence_keywords_themed.keys())]

# Compare count over the years (sum across countries per year)

yearly_counts = df_un_merged.groupby('Year')[theme_cols].sum()

plt.figure(4, figsize=(12,6))

for theme in theme_cols:
    plt.plot(yearly_counts.index, yearly_counts[theme], label=theme)

plt.title("Development of Theme Mentions in UN Speeches Over the Years")
plt.xlabel("Year")
plt.ylabel("Total Mentions per theme")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Theme tf-idf columns

theme_cols = [theme+'_tfidf' for theme in list(violence_keywords_themed.keys())]

# Compare tf-idf over the years

yearly_counts = df_un_merged.groupby('Year')[theme_cols].sum()

plt.figure(5, figsize=(12,6))

for theme in theme_cols:
    plt.plot(yearly_counts.index, yearly_counts[theme], label=theme)

plt.title("Development of Theme tf idf in UN Speeches Over the Years")
plt.xlabel("Year")
plt.ylabel("Summed tf idf per theme")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Per-year corpus analysis: compute yearly tf-idf for keywords

# Concatenate all speeches per year into one text per year

yearly_speechcorpus = df_un_merged.groupby('Year')['Speech'].apply(lambda texts: " ".join(texts))

# Reuse keywords list; build yearly tf-idf matrix

yearlyvectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),   # allows multi-word terms like "intimate partner violence"
    token_pattern=r"(?u)\b\w[\w#@'-]+\b",
)

matrix_yearly = yearlyvectorizer.fit_transform(yearly_speechcorpus)

feature_names = yearlyvectorizer.get_feature_names_out()

# Map keywords to yearly tf-idf column indices and extract

yearlykeywords_to_idx = {word: i for i, word in enumerate(feature_names) if word in keywords}

yearly_tfidf_mat = matrix_yearly[:, list(yearlykeywords_to_idx.values())]

tfidf_yearly = pd.DataFrame(
    yearly_tfidf_mat.toarray(), 
    index=yearly_speechcorpus.index,
    columns=list(yearlykeywords_to_idx.keys())
)

# Build theme-level yearly tf-idf summaries

theme_tfidf_yearly = pd.DataFrame(index=tfidf_yearly.index)

for theme, kw in violence_keywords_themed.items():
    matching_cols = [k.lower() for k in kw if k.lower() in tfidf_yearly.columns]
    if matching_cols:
        theme_tfidf_yearly[theme] = tfidf_yearly[matching_cols].sum(axis=1)
    else:
        theme_tfidf_yearly[theme] = 0
        
#%%
# Plot yearly theme tf-idf lines

plt.figure(6, figsize=(12,6))
for theme in theme_tfidf_yearly.columns:
    plt.plot(theme_tfidf_yearly.index, theme_tfidf_yearly[theme], label=theme)

plt.title("Theme TF-IDF Scores Over Years in UN Speeches")
plt.xlabel("Year")
plt.ylabel("Sum of TF-IDF Scores")
plt.legend()
plt.grid(True)
plt.show()

# Plot yearly per-keyword tf-idf lines (for terms that exist in the yearly vocab)

plt.figure(7, figsize=(12,6))
for col in tfidf_yearly.columns:
    if col in keywords:
        plt.plot(tfidf_yearly.index, tfidf_yearly[col], label=col)

plt.title("TF-IDF Scores Over Years in UN Speeches")
plt.xlabel("Year")
plt.ylabel("TF-IDF Scores")
plt.legend()
plt.grid(True)
plt.show()

df_un_merged = df_un_merged.drop('Speech', axis = 1)

#%%
#Plot regional differences in total mentions and tf-idf scores
keyword_cols = [col for col in df_un_merged.columns if col.lower() in [keyword+'_counts' for keyword in keywords]]

#add a total_mentions column per speech
df_un_merged["total mentions"] = df_un_merged[keyword_cols].sum(axis=1)

#group by region and year
mentions_by_region_year = (
    df_un_merged.groupby(["Region Name", "Year"])["total mentions"]
      .sum()
      .reset_index()
)

#plot
plt.figure(8, figsize=(12,6))

for region, subdf in mentions_by_region_year.groupby("Region Name"):
    plt.plot(subdf["Year"], subdf["total mentions"], label=region)

plt.title("Total Mentions of Keywords by Region Over Time")
plt.xlabel("Year")
plt.ylabel("Total Mentions")
plt.legend(title="Region")
plt.grid(True)
plt.show()

#Plot regional differences in total mentions and tf-idf scores
keyword_cols = [col for col in df_un_merged.columns if col.lower() in [keyword+'_tfidf' for keyword in keywords]]
#add a total tf-idf column per speech
df_un_merged["total tfidf"] = df_un_merged[keyword_cols].sum(axis=1)

#group by region and year
tfidf_by_region_year = (
    df_un_merged.groupby(["Region Name", "Year"])["total tfidf"]
      .sum()
      .reset_index()
)

#plot
plt.figure(9, figsize=(12,6))

for region, subdf in tfidf_by_region_year.groupby("Region Name"):
    plt.plot(subdf["Year"], subdf["total tfidf"], label=region)

plt.title("Total tfidf score of Keywords by Region Over Time")
plt.xlabel("Year")
plt.ylabel("Total tfidf")
plt.legend(title="Region")
plt.grid(True)
plt.show()





