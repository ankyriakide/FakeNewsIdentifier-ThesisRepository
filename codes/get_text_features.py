#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:37:16 2019

@author: ankyriakide
"""

import pandas as pd
import re   #https://github.com/python/cpython/blob/3.7/Lib/re.py, https://www.w3schools.com/python/python_regex.asp
from textblob import TextBlob 

df= pd.read_csv('my_tweets_data.csv')

#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

first_pronouns = set(['I', 'me','mine','we','ours'])

def clean_tweet(tweet): 
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) #clean tweet 
  
def get_tweet_polarity(tweet): 
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'                              #decide on tweet's polarity

def get_tweet_subjectivity(tweet): 
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.subjectivity > 0: 
            return 'positive'
        elif analysis.sentiment.subjectivity == 0: 
            return 'neutral'
        else: 
            return 'negative'                            #decide on tweet's subjectivity



features=[]

for index, tweet in df.iterrows():
    
    try:
        r = re.compile(tweet.text)
    except:
        r = re.compile(clean_tweet(tweet.text))
    
    if re.search(r'\?', tweet.text)==None:              # check for question mark and multiple question marks
        has_qmark = 0
    else:
        if re.search(r'\?\?', tweet.text)==None:
            has_qmark = 1
        else:
            if re.search(r'\?\?\?', tweet.text)==None:
                has_qmark = 2
            else:
                has_qmark = 3                  


    if re.search(r'\!', tweet.text)==None:             # check for exclamation mark and multiple exclamation marks
        has_exclamation = 0
    else:
        if re.search(r'\!\!', tweet.text)==None:
            has_exclamation = 1
        else:
            if re.search(r'\!\!\!', tweet.text)==None:
                has_exclamation = 2
            else:
                has_exclamation = 3
                
    
    if re.search(r'\@', tweet.text)==None:             # check for mentions
        has_mention = False
    else:
        has_mention = True


    if re.search(r'\#', tweet.text)==None:             # check for hashtags
        has_hash = False
    else:
        has_hash = True


    if any(re.search(line, tweet.text) for line in emoticons):            # check for emoticons
        has_emoticons = True
    else:
        has_emoticons = False
        
        
    if any(r.match(line) for line in first_pronouns):                      # check for first pronoun words
        has_first = True
    else:
        has_first = False

        
    new_tweet={'id': tweet.id,
               'has_qmark': has_qmark,
               'has_exclamation': has_exclamation,
               'has_mention': has_mention,
               'has_hash': has_hash,
               'has_first': has_first,
               'polarity': get_tweet_polarity(tweet.text),
               'subjectivity': get_tweet_subjectivity(tweet.text)}
    
    features.append(new_tweet)

features_pd=pd.DataFrame(features)
features_pd.to_csv('textfeatures.csv')