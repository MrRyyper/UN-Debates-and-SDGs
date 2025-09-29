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
import textwrap

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

#%%
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

#%%
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
plt.xlim((1990,2024))
plt.grid(True)
plt.show()

#%%
#Prettyplots
#plot total mentions of violance against women
total_mentions_by_year = (
    mentions_by_region_year.groupby("Year")["total mentions"]
    .sum()
    .reset_index()
)


fig = plt.figure(10, figsize=(6,6))

plt.plot(total_mentions_by_year["Year"], total_mentions_by_year["total mentions"], color="blue", label='Mentions of keywords')
plt.title("Total Mentions of Violence-Related Keywords Over Time (All Regions)")
plt.xlabel("Year")
plt.ylabel("Total # of Mentions")
plt.axvline(x=1990, color="grey", linestyle="--", linewidth=2, label="Legislation starts")

plt.annotate(
    "First laws implemented",         # the label text
    xy=(1990, 30),                   # point to (x=1990, y=0 baseline)
    xytext=(1950, total_mentions_by_year["total mentions"].max() * 0.7),  # text position
    arrowprops=dict(facecolor="grey", lw=1.5),
    fontsize=10, color="grey", fontweight="bold"
)
plt.legend(loc = 'upper left')
plt.grid(True)
caption = ("Fig. 1: This figure shows the rise of mentions of (sexual) violence keywords within the united nations general debates and how it coincides with the first legislations put into place.")

wrapped_caption = "\n".join(textwrap.wrap(caption, width=65))  
fig.text(0.12, 0.06, wrapped_caption,
         ha="left", va="bottom", fontsize=10)

plt.subplots_adjust(top=0.90, bottom=0.25)

plt.show()

#%%
#plot regions related to beijing convention
fig = plt.figure(11, figsize=(6,6))

# Track whether we've already added the "Other regions" label
other_label_added = False

for region, subdf in mentions_by_region_year.groupby("Region Name"):
    if region == 'Europe':
        plt.plot(subdf["Year"], subdf["total mentions"], label=region, color='blue')
    elif region == 'Asia':
        plt.plot(subdf["Year"], subdf["total mentions"], label=region, color='red')
    else:
        if not other_label_added:
            plt.plot(subdf["Year"], subdf["total mentions"], label='Other regions', color='grey')
            other_label_added = True
        else:
            plt.plot(subdf["Year"], subdf["total mentions"], color='grey')

plt.axvline(x=1995, color="green", linestyle="--", linewidth=2)

plt.annotate(
    "Beijing conference",         # the label text
    xy=(1995, 15),                   # point to (x=1990, y=0 baseline)
    xytext=(1982, 20),  # text position
    arrowprops=dict(facecolor="green", lw=1.5),
    fontsize=10, color="green", fontweight="bold"
)

plt.title("Total mentions of Violence related keywords per region 1980-2010")
plt.xlabel("Year")
plt.ylabel("Total # of Mentions")
plt.xlim((1980,2010))
plt.legend(title="Region")
plt.grid(True)

caption = ("Fig. 2: This figure shows the peaks of mentioned keywords related to (sexual) violence against women around the World Conference on Women in Beijing. It shows a peak in Asia before, which could already have been preparing for the conference, and a peak in Europe after, which could have been a reaction to the conference.")


wrapped_caption = "\n".join(textwrap.wrap(caption, width=65))  
fig.text(0.12, 0.06, wrapped_caption,
         ha="left", va="bottom", fontsize=10)

plt.subplots_adjust(top=0.90, bottom=0.30)


plt.show()

#%%
#Prettyplot of chosen keywords
chosen_words = ['sexual exploitation','gender based violence','sexual violence']
fig = plt.figure(12, figsize=(12,6))
for col in tfidf_yearly.columns:
    if col in chosen_words:
        plt.plot(tfidf_yearly.index, tfidf_yearly[col], label=col)

plt.title("TF-IDF Scores of sexual exploitation, gender based violence and sexual violence in UN Speeches")
plt.xlabel("Year")
plt.ylabel("TF-IDF Scores")
plt.xlim((1990,2025))

plt.grid(True)
plt.annotate(
    "Istanbul Convention\n(2013)", 
    xy=(2013, tfidf_yearly['sexual violence'].loc[2013]), 
    xytext=(2005, tfidf_yearly.max().max() * 0.75),
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
    fontsize=9, color="black", fontweight="bold"
)

# #MeToo Movement (2017 peak in sexual exploitation)
plt.annotate(
    "#MeToo movement (2017)",
    xy=(2017, tfidf_yearly['sexual exploitation'].loc[2017]),
    xytext=(2018, tfidf_yearly.max().max() * 0.80),
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
    fontsize=9, color="black", fontweight="bold"
)

# COVID-19 & GBV increase (2020)
plt.annotate(
    "COVID-19 lockdown (2020)",
    xy=(2020, tfidf_yearly['gender based violence'].loc[2020]),
    xytext=(2018, tfidf_yearly.max().max() * 0.55),
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
    fontsize=9, color="black", fontweight="bold"
)
plt.legend(loc='upper left')

caption = (
    "Fig. 3: The most notable spikes of keywords that seem related to global events. Mentions of "
    "sexual exploitation peak in 2017 during the #MeToo movement. Gender based "
    "violence was most prominent in 2020, linked to increased violence during "
    "COVID-19 lockdowns. Sexual violence spikes "
    "in 2013–2014 align with the Istanbul Convention's implementation."
)
wrapped_caption = "\n".join(textwrap.wrap(caption, width=120))
fig.text(0.12, 0.05, wrapped_caption, ha="left", va="bottom", fontsize=10)

plt.subplots_adjust(top=0.88, bottom=0.20)

plt.show()

#%%

#plot
fig = plt.figure(14, figsize=(12,6))

for region, subdf in tfidf_by_region_year.groupby("Region Name"):
    plt.plot(subdf["Year"], subdf["total tfidf"], label=region)

plt.title("Total tf idf score by Region 1990-2024")
plt.xlabel("Year")
plt.ylabel("Total tf-idf")
plt.legend(title="Region")
plt.xlim((1990,2024))
plt.grid(True)

caption = (
    "Figure 4. Regional trends in total TF-IDF scores for references to violence against women over the past three decades. Europe consistently shows the highest frequency since 2005, though declining in recent years. America, Asia, and Africa exhibit slightly lower levels, with a notable spike around the Istanbul Convention. Africa displays a distinct increase in 2020 during the COVID-19 pandemic. Oceania consistently records the lowest scores, particularly over the past five years."
)
wrapped_caption = "\n".join(textwrap.wrap(caption, width=120))
fig.text(0.12, 0.05, wrapped_caption, ha="left", va="bottom", fontsize=10)

plt.subplots_adjust(top=0.88, bottom=0.25)

plt.show()



