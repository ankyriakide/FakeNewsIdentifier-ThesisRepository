#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:26:28 2019

@author: ankyriakide
"""
import json
import pandas as pd

#df1 = pd.read_json('tweets1.json', encoding='utf-8')
#df2 = pd.read_json('tweets2.json', encoding='utf-8')
#df3 = pd.read_json('tweets3.json', encoding='utf-8')

#df1csv = df1.to_csv (r'df1.csv', index = None, header=True)
#df2csv = df2.to_csv (r'df2.csv', index = None, header=True)
#df3csv = df3.to_csv (r'df3.csv', index = None, header=True)


#Run over the terminal 
#April
twitterscraper "merger OR acquisition" --lang en --limit 1000 --begindate 2019-04-01 --enddate 2019-04-30 --output=tweets1.json --profiles

df1 = panda.read_json('tweets1.json', encoding='utf-8')

#May
twitterscraper "merger OR acquisition" --lang en --limit 1000 --begindate 2019-05-01 --enddate 2019-05-31 --output=tweets2.json --profiles

df2 = panda.read_json('tweets.json', encoding='utf-8')

#June
twitterscraper "merger OR acquisition" --lang en --limit 1000 --begindate 2019-06-01 --enddate 2019-06-30 --output=tweets3.json --profiles

df3 = panda.read_json('tweets3.json', encoding='utf-8')