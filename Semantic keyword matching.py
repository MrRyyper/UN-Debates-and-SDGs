# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 20:50:35 2025

@author: schud
"""

import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

sessions = np.arange(1, 80)
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