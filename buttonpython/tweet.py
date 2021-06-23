import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tweepy import OAuthHandler
import sys
import base64
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
def cleanUpTweet(txt):
    txt = re.sub(r'@[A-Za-z0-9_]+','',txt)
    txt = re.sub(r'#','',txt)
    txt = re.sub(r'RT : ','',txt)
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
    return(txt)



def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity



def getTextAnalysis(a):
    if a<0:
        return "Negative"
    elif a>0:
        return "Positive"
    

def sentiment(twitterAccount, number):
    consumer_key = 'NxxZKWT2oa1rZ6vxn4ecxGjw2'
    consumer_secret = '0fmd0BtxzpdxjQJxvczABsAm7KDArbLyatUkz4dSAnxq7opKBl'
    access_token = '1379042437214052357-Kavf1mm1H9GDG9p7HkIopW0yfsGWyF'
    access_secret = 'hsB1Vkpb8YICQXh7nNVGR1fA0qcg5zCM46gJ3Cy3XwhSm'

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    twitterApi = tweepy.API(auth, wait_on_rate_limit=True)

    tweets = tweepy.Cursor( twitterApi.user_timeline, screen_name = twitterAccount, count = None, since_id = None, max_id = None, trim_user = True, exclude_replies = True, contributor_details = False, include_entities = False).items(number);
    df = pd.DataFrame(data = [(tweet.text , tweet.created_at) for tweet in tweets],columns = ['Tweet','Created'])
    
    exist = True
    
    try:
        df['Tweet'] = df['Tweet'].apply(cleanUpTweet)

        df['Subjectivity'] = df['Tweet'].apply(getTextSubjectivity)
        df['Polarity'] = df['Tweet'].apply(getTextPolarity)

        df = df.drop(df[df['Tweet']==''].index)
    
        df = df.drop(df[df['Subjectivity']<0.1].index)
        df['Score'] = df['Polarity'].apply(getTextAnalysis)
        avg_sub = df['Subjectivity'].mean()
        avg_pol = df['Polarity'].mean()
        positive = df[df['Score']=="Positive"]
        pos = (positive.shape[0]/df.shape[0])*100
        
        
        negative = df[df['Score']=="Negative"]
        neg = (negative.shape[0]/df.shape[0])*100
        textual = """"""
        for i in df['Tweet']:
            textual += i + "\n"
    
        vals = [pos,neg]
    
        plt.switch_backend('AGG')
        plt.figure(figsize=(5,5))
        plt.pie(vals,labels=['Positive','Negative'],startangle=90)
        plt.legend()
        plt.title("Percentage of Tweets for @{0}".format(twitterAccount))
        graph = get_graph()
        nums = [positive.shape[0],negative.shape[0],avg_sub,avg_pol]
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(textual)
        plt.switch_backend('AGG')
        plt.figure(figsize = (5, 5), facecolor = None)
        plt.imshow(wordcloud)
        plt.title("Word Cloud of Most Frequently Used Words")
        plt.axis("off")
        word = get_graph()
    
    except Exception:
        exist = False

    
    
    
    if exist!=False:
        return( graph,textual,nums,word)
    else:
        return(None,None,[-1,-1,-1,-1],None)
    

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer , format = 'png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph






