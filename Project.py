import tweepy
import config
import numpy as np
import pandas as pd

#Get credentials from file (config.py) and create the a client instance.
def getClient():
    client = tweepy.Client(bearer_token = config.BEARER_TOKEN,
                           consumer_key = config.API_KEY,
                           consumer_secret = config.API_KEY_SECRET,
                           access_token = config.ACCESS_TOKEN,
                           access_token_secret = config.ACCESS_TOKEN_SECRET)
    return client

#Function to get tweets.
def getTweetsInfo():
    client = getClient()
    #These are the coluns for the dataframe that is created leater in the function.
    columns = ['id', 'Tweet', 'Time', 'Language', 'Source', 'Author_id', 'retweet_count', 'reply_count', 'like_count', 'quote_count']
    data = []
    #The argument tight after the method search_recent_tweets is the query. "-is:retweet -is:reply -is:quote" is to ignore retweets, replies, and quotes.
    for tweet in tweepy.Paginator(client.search_recent_tweets,'covid OR Covid OR COVID OR hoax OR Hoax -is:retweet -is:reply -is:quote',
                                max_results=100, expansions=['author_id', 'entities.mentions.username'],
                                tweet_fields=['created_at', 'lang', 'public_metrics', 'source', 'id']).flatten(limit=10000):
        data.append([tweet.id, tweet.text, tweet.created_at, tweet.lang, tweet.source, tweet.author_id, tweet.public_metrics['retweet_count'], tweet.public_metrics['reply_count'], tweet.public_metrics['like_count'], tweet.public_metrics['quote_count']])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('tweets.csv')
    return data

#Call function to get tweets.
info = getTweetsInfo()

#The functions below are optional in case we want to look for tweets by username and/or open the csv file created previously.
#If we want tweets by username, the getTweetsInfo function needs to be modified by replacing the method search_recent_tweets with the moethod get_users_tweets.

def getUserInfo():
    client = getClient()
    user = client.get_user(username = 'username')
    return user.data.id

def openfile():
    df = pd.read_csv(f'tweets.csv')
    return df