# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:07:32 2025

@author: schud
"""

import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Create an empty dataframe scaffold
main_df = pd.DataFrame(columns=["session", "filename", "relative_path"])

base_dir = os.path.dirname(__file__)

maindirpath = os.path.join(base_dir, "UN_speeches_data", "Data", "TXT")

maindirlist = os.listdir(maindirpath)
records = []
for session in maindirlist:
    sessionpath = os.path.join(maindirpath, session)
    filenames = os.listdir(sessionpath)
    for file in filenames:
        filepath = os.path.join(sessionpath, file)
        records.append({
            "session": session,
            "filename": file,
            "relative path": filepath
        })

# Convert to DataFrame
main_df = pd.DataFrame(records)

main_df['year'] = main_df['session'].str[-4:].astype(int)
main_df['country'] = main_df['filename'].str[:3].astype(str)

# for filepath in allfilepaths:
#     f = open(filepath, encoding="utf-8")
#     print(f.read())
#     break

## simple countvectorizer on some keywords
keywords = [
    "sexual violence",
    "rape",
    "assault",
    "harassment",
    "molestation",
    "abuse",
    "intimate partner violence",
    "domestic violence",
]

#open & read docs
docs = []
for path in main_df["relative path"]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        docs.append(f.read())

#use wordcount vectorizer
vectorizer = CountVectorizer(vocabulary=keywords, ngram_range=(1, 3), lowercase=True)
X = vectorizer.fit_transform(docs)

#merge keyword vectors with original df
keyword_df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=main_df.index  # align with main_df rows
)
main_df = pd.concat([main_df, keyword_df], axis=1)

main_df['totalcount'] = main_df[[
    "sexual violence", 
    "rape",
    "assault", 
    "harassment", 
    "molestation", 
    "abuse", 
    "intimate partner violence", 
    "domestic violence",]].sum(axis=1)

