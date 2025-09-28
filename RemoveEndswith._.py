# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:07:32 2025

@author: schud
"""

import os

base_dir = os.path.dirname(__file__)

maindirpath = os.path.join(base_dir, "UN_speeches_data", "Data", "TXT")

maindirlist = os.listdir(maindirpath)
allfilepaths = []
for entry in maindirlist:
    sessionpath = os.path.join(maindirpath, entry)
    filenames = os.listdir(sessionpath)
    for file in filenames:
        filepath = os.path.join(sessionpath, file)
        allfilepaths.append(filepath)

for filepath in allfilepaths:
    f = open(filepath, encoding="utf-8")
    print(f.read())
    break
