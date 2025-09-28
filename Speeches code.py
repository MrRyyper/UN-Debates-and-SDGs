# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:07:32 2025

@author: schud
"""

import os

base_dir = os.path.dirname(__file__)

maindirpath = os.path.join(base_dir, "UN_speeches_data", "Data", "TXT")

maindirlist = os.listdir(maindirpath)
for entry in maindirlist:
    sessionpath = os.path.join(maindirpath, entry)
    if entry.startswith('._'):
        os.remove(os.path.join(maindirpath, entry))
    else:
        filenames = os.listdir(sessionpath)
        for file in filenames:
            filepath = os.path.join(sessionpath, file)
            print(file)
            if file.startswith('._'):
                print(file)
                os.remove(filepath)
