from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
import hidden

#Loads csv file with results into a dataframe.
df = pd.read_csv(f'tweets.csv')

#Roberta model for sentiment analysis.
roberta = 'cardiffnlp/twitter-roberta-base-sentiment'

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']
neg = []
neu = []
pos = []

for tweet in df['Tweet']:
    tweet_words = []

    #Removes the username and any website link.
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    for i in range(len(scores)):
        l = labels[i]
        s = scores[i]
        if l == 'Negative':
            neg.append(s)
        elif l == 'Neutral':
            neu.append(s)
        else:
            pos.append(s)

df['Negative'] = neg
df['Neutral'] = neu
df['Positive'] = pos

def getLength(text):
    return len(text)
    
df['Length of tweet'] = df['Tweet'].apply(getLength)
df = df.drop(columns=['Unnamed: 0'])

M = max(df['Time'])
m= min(df['Time'])
MM = df.loc[df['Time'] == M, ['Tweet']]
mm = df.loc[df['Time'] == m, ['Tweet']]
print('First tweet(s)', mm, 'at', m)
print('Last tweet(s)', MM, 'at', M)
Most_common_source = statistics.mode(df['Source'])
print('Most Common Source: ', Most_common_source)
Most_common_language = statistics.mode(df['Language'])
print('Most Common Language: ', Most_common_language)
max_retweet = max(df['retweet_count'])
max_retweet_tweet = df.loc[df['retweet_count'] == max_retweet, ['Tweet']]
print('Most retweeted tweet', max_retweet_tweet, 'retweeted', max_retweet, 'times')
df['Sentiment'] = df[['Negative', 'Neutral', 'Positive']].idxmax(axis="columns")
Language_count = df['Language'].value_counts(normalize=True).reset_index()
Source_count = df['Source'].value_counts(normalize=True).reset_index()
Sentiment_count = df['Sentiment'].value_counts().reset_index()

fig, axs = plt.subplots(3, 2)

df_draw = Language_count.copy()
df_draw.loc[df_draw['Language'] < 0.02, 'index'] = 'other'
df_draw = df_draw.groupby('index')['Language'].sum().reset_index()
axs[0, 0].pie(df_draw['Language'], labels=df_draw['index'], autopct='%.0f%%', radius=1.3)
axs[0, 0].title.set_text('Language Distribution')

df_draw2 = Source_count.copy()
df_draw2.loc[df_draw2['Source'] < 0.02, 'index'] = 'other'
df_draw2 = df_draw2.groupby('index')['Source'].sum().reset_index()
axs[0, 1].pie(df_draw2['Source'], labels=df_draw2['index'], autopct='%.0f%%', radius=1.3)
axs[0, 1].title.set_text('Source Distribution')

axs[1, 0].bar(Sentiment_count['index'], Sentiment_count['Sentiment'])
axs[1, 0].title.set_text('Sentiment Analysis')
axs[1, 0].set_xlabel('sentiment')
axs[1, 0].set_ylabel('counts')

axs[1, 1].hist(df['Negative'], bins=100)
axs[1, 1].title.set_text('Negative Scores')
axs[1, 1].set_xlabel('score')
axs[1, 1].set_ylabel('frequency')
fig.subplots_adjust(hspace=0.5)
axs[2, 0].hist(df['Neutral'], bins=100)
axs[2, 0].title.set_text('Neutral Scores')
axs[2, 0].set_xlabel('score')
axs[2, 0].set_ylabel('frequency')
axs[2, 1].hist(df['Positive'], bins=100)
axs[2, 1].title.set_text('Positive Scores')
axs[2, 1].set_xlabel('score')
axs[2, 1].set_ylabel('frequency')

mx = df.select_dtypes(include=np.number).corr()
fig2 = plt.subplots(1)
sns.heatmap(mx, annot=True, cmap='YlGnBu')

plt.show()
s = df.describe()
print(s)

df.to_csv('tweets2.csv')

#Connect to database
import psycopg2
import hidden

secrets = hidden.secrets()
conn = psycopg2.connect(host=secrets['host'], database=secrets['database'], user=secrets['user'], password=secrets['pass'])
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS Tweets CASCADE;')
cur.execute('CREATE TABLE IF NOT EXISTS Tweets (id SERIAL, Tweet_id VARCHAR(20), Tweet VARCHAR(255), Time TIMESTAMPTZ, Language VARCHAR(3), Source VARCHAR(50), Author_id VARCHAR(20), retweet_count INTEGER, reply_count INTEGER, like_count INTEGER, quote_count INTEGER);')
conn.commit()

cur.execute('\COPY Tweets(id, Tweet_id, Tweet, Time, Language, Source, Author_id, retweet_count, reply_count, like_count, quote_count) FROM "tweets2.csv" WITH DELIMITER "," CSV HEADER;')
conn.commit()
cur.close()