#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:26:16 2019

@author: ankyriakide
"""
import pandas as pd

from tweepy import API

#run tweet authentication first

df=pd.read_csv('df_id.csv', dtype='str')

the_ma_tweets=[]

for index, row in df.iterrows():
    print(row)
    try:
        this_tweet = api.get_status(row.tweet_id)
        print(this_tweet)
        #tweet = {'text': this_tweet.text}
    except:
        #tweet = {'text': None}
        this_tweet = None
        
    the_ma_tweets.append(this_tweet)

the_ma_tweets_df=pd.DataFrame(the_ma_tweets)    
 

ma_tweets=[]

for this_tweet in range(3600):
    print(row)
    try:
        tweet=the_ma_tweets_df.loc[this_tweet,0] #draws each row
        temp={'tweet_id': tweet.id_str,
              'text': tweet.text,
              'coordinates': tweet.coordinates,
              'created_at': tweet.created_at,
              'tweet_favourited_count': tweet.favorite_count,
              'retweet_count': tweet.retweet_count,
              'entities': tweet.entities,
              'default_profile_image': tweet.user.default_profile_image,
              'description': tweet.user.description,
              'users_favourites_count': tweet.user.favourites_count,
              'followers_count': tweet.user.followers_count,
              'followings_count': tweet.user.friends_count,
              'user_activity_count': tweet.user.statuses_count,
              'verified': tweet.user.verified
              }
    
    except:
           temp={'tweet_id': None,
                 'text': None,
                 'coordinates': None,
                 'created_at': None,
                 'tweet_favourited_count': None,
                 'retweet_count': None,
                 'entities': None,
                 'default_profile_image': None,
                 'description': None,
                 'users_favourites_count': None,
                 'followers_count': None,
                 'followings_count': None,
                 'user_activity_count': None,
                 'verified': None
                 }
          
    ma_tweets.append(temp)
    
ma_tweets_pd=pd.DataFrame(ma_tweets)
    
##the_ma_tweets_df.to_data('ma_tweets.data')




####
#users_df=pd.read_csv('username.csv', dtype='str')
#
#the_users=[]
#
#for index, row in users_df.iterrows():
#    print(row)
#    try:
#        user_info=api.get_user(row.username)
#        user = {'friends_count': user_info.friends_count,
#                'followers_count': user_info.followers_count,
#                'default_profile_image': user_info.default_profile_image,
#                'verified': user_info.verified,
#                'description': user_info.description,
#                'favourites_count': user_info.favourites_count,
#                'statuses_count': user_info.statuses_count
#               }
#    except:
#           user = {'friends_count': None,
#                   'followers_count': None,
#                   'default_profile_image': None,
#                   'verified': None,
#                   'description': None,
#                   'favourites_count': None,
#                   'statuses_count':None
#                  }
#      
#    the_users.append(user)
#the_users_pd = pd.DataFrame(the_users)