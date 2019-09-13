#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:00:41 2019

@author: ankyriakide
"""

import os
import pandas as pd
import json

from convert_veracity_annotations import convert_annotations

main_path = 'all-rnr-annotated-threads'  #Set working directory in the project, to read over the file with the data
tweets_list = [] #Create a "dictionary" that writes in everything ran over the file in the main path. All  source tweets will be listed in here as rows.

with os.scandir(main_path) as events:
    for event in events: 
        if not event.name.startswith('.'):
            events_path=str(main_path)+'/'+str(event.name) 
            with os.scandir(events_path) as labels:
                for label in labels:  
                    if not label.name.startswith('.'): 
                        labels_path=events_path+'/'+str(label.name)
                        with os.scandir(labels_path) as tweets:
                            for tweet in tweets: 
                                 if not tweet.name.startswith('.'): 
                                    tweets_path=labels_path+'/'+str(tweet.name)
                                    
                                    f = open(str(tweets_path)+'/annotation.json', "r")
                                    annotation = json.load(f) 
                                    if 'category' in annotation.keys():
                                        category = annotation['category']
                                    else:
                                        category = None
            
                                    if 'misinformation' in annotation.keys():
                                        misinformation = annotation['misinformation']
                                    else:
                                        misinformation = None
            
                                    if 'true' in annotation.keys():
                                        is_true = annotation['true']
                                    else:
                                        is_true = None
                                    
                                    if 'links' in annotation.keys():
                                        links = annotation['links']
                                    else:
                                        links = None
                                    
                                    if 'is_turnaround' in annotation.keys():
                                        is_turnaround = annotation['is_turnaround']
                                    else:
                                        is_turnaround = None
                                    
                                    f = open(str(tweets_path)+'/structure.json')
                                    structure = f.read()
                                    
                                    f = open(str(tweets_path)+'/source-tweets/'+str(tweet.name)+'.json')
                                    source_tweet = json.load(f)
                                    
                                    
                                    reactions_nr = len(list(os.scandir(str(tweets_path)+'/reactions/')))
                                    
                                    new_tweet = {'event': event.name,
                                                 'label': label.name,
                                                 'annotation': annotation,
                                                 'annotation_is_rumour': annotation['is_rumour'],
                                                 'annotation_category': category,
                                                 'annotation_misinformation': misinformation,
                                                 'annotation_true': is_true,
                                                 'annotation_links': links,
                                                 'annotation_is_turnaround': is_turnaround,
                                                 'annotation_veracity': convert_annotations(annotation),
                                                 'structure': structure,
                                                 'source_tweet': source_tweet,
                                                 'reactions_nr': reactions_nr}
                                    
                                    new_tweet.update(source_tweet) #Add tweet fields
                                    new_tweet['tweet_created_at']=new_tweet['created_at']
                                    new_tweet.update(source_tweet['user']) #Expand user fields
                                    
                                    tweets_list.append(new_tweet)

tweets_df = pd.DataFrame(tweets_list)