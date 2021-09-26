import enchant
import spacy
from spacy.lang.en import English
from matplotlib import pyplot as plt
import tweepy
import pandas as pd
import numpy as np
import re
import string
import preprocessor as p
from nrclex import NRCLex
import demoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
import en_core_web_sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlalchemy
from sqlalchemy import create_engine
import time

start_time = time.time()

consumerKey = "8udhSgxLFBH99zgN06DuSYOQi"
consumerSecret = "pIPXXFRUqStC3Z6J1YMPdz7W4AwD8bioGK7hMueFdks7pm0OIc"
accessToken = "1083286165845954560-ceHxFmX1Rak89Y5UX7yFaJzOVnF97t"
accessTokenSecret = "MN9dnShe69FTwc0KxsBn9ooQN4QGtJZtMlG1vRKgVdus0"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)
print("API Authentication")

nlp = en_core_web_sm.load()
d = enchant.Dict("en_US")

def tweet_cleaning(x, keyword):

    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.NUMBER)
    x = p.clean(x)
    x = demoji.replace_with_desc(x)
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            x = x.replace(separator, ' ')
    words = []
    for word in x.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)

    return ' '.join(words)


def data_generation(cleaned_tweets_list):
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    token_tweets_list = []
    noun_phrases_list = []
    for tweet in cleaned_tweets_list:
        tweet_tokens, noun_phrases = tweet_processing(tweet, spacy_stopwords)
        token_tweets_list.append(tweet_tokens)
        noun_phrases_list.append(noun_phrases)
    return token_tweets_list, noun_phrases_list


def tweet_processing(text, spacy_stopwords):
    my_doc = nlp(text)
    tokens = []
    noun_phrases = []
    for token in my_doc:
        if token.is_punct == False and token.text not in spacy_stopwords:
            tokens.append(token.text)
    for chunk in list(my_doc.noun_chunks):
        noun_phrases.append(chunk.text)
    return tokens, noun_phrases


def get_tweets(keyword, noOfTweet):
    tweets = tweepy.Cursor(api.search, q=keyword, lang="en", tweet_mode = "extended").items(noOfTweet)
    tweet_list = []
    cleaned_tweets_list = []
    dates = []
    ids_list = []
    print("Obtained Tweets, moving to cleaning")
    for tweet in tweets:
        tweet_list.append(tweet._json["full_text"])
        dates.append(tweet.created_at)
        ids_list.append(tweet._json["id"])
        cleaned_tweet = tweet_cleaning(tweet._json["full_text"], keyword)
        cleaned_tweets_list.append(cleaned_tweet)

    return tweet_list, cleaned_tweets_list, dates, ids_list


def aggregate_emotion(token_tweets_list) :
    emotion_dict = {'fear': 0.0, 'anger': 0.0, 'anticip': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0, 'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
    emotion_counter_dict = {'fear': 0, 'anger': 0, 'anticip': 0, 'trust': 0, 'surprise': 0, 'positive': 0, 'negative': 0, 'sadness': 0, 'disgust': 0, 'joy': 0}
    counter = 0
    total = 0
    emotion_list=[]

    for tokens in token_tweets_list:
        for token in tokens:
            emotion = NRCLex(token)
            for k1,v1 in emotion.affect_frequencies.items():
                for k2, v2 in emotion_dict.items():
                    if k1 == k2 and v1 > 0:
                        emotion_dict[k2] = v2 + v1
                        emotion_list.append(k2)
                        total = total + v1
                        for k3, v3 in emotion_counter_dict.items():
                            if k1 == k3:
                                counter += 1
                                emotion_counter_dict[k3] += 1
                    else:
                        emotion_list.append("None")

    print("\nCounter:", counter, "Total:", total)
    return emotion_dict, emotion_counter_dict, counter, total, emotion_list


def percentage(part,whole):
    return 100 * float(part)/float(whole)


def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    pos = 0
    neg = 0
    neu = 0

    if sentiment_dict['compound'] >= 0.05:
        pos += 1

    elif sentiment_dict['compound'] <= - 0.05:
        neg += 1

    else:
        neu += 1
    return pos, neg, neu

keyword = input("Please enter keyword or hashtag to search: ")
noOfTweet = int(input("Please enter how many tweets to analyze: "))
tweet_list, cleaned_tweets_list, dates, ids_list = get_tweets(keyword, noOfTweet)
token_tweets_list, noun_phrases_list = data_generation(cleaned_tweets_list)
print("Obtained tweets:", len(cleaned_tweets_list))

overall_pos = 0
overall_neg = 0
overall_neu = 0
sent_list = []

for x in cleaned_tweets_list :
    pos, neg, neu = sentiment_scores(x)
    if pos == 1 :
        sent_list.append('Postive')
    elif neg == 1 :
        sent_list.append('Negative')
    else :
        sent_list.append('Neutral')
    overall_pos = overall_pos + pos
    overall_neg = overall_neg + neg
    overall_neu = overall_neu + neu

print("Positive percentage = ", percentage(overall_pos, len(cleaned_tweets_list)))
print("Negative percentage = ", percentage(overall_neg, len(cleaned_tweets_list)))
print("Neutral percentage = ", percentage(overall_neu, len(cleaned_tweets_list)))

emotion_dict_noun, emotion_counter_dict_noun, counter_noun, total_noun, emotion_list = aggregate_emotion(noun_phrases_list)
print("\nThe percentage of noun phrases emotions for:", keyword)
emotion_percentages = []
for k2, v2 in emotion_counter_dict_noun.items() :
    p = percentage(v2,counter_noun)
    print("Emotion :", k2, "Percentage:", p)
    emotion_percentages.append(p)

tweet_df = pd.DataFrame(columns=['Tweet ID', 'Date Created', 'Tweet','Tweet_Text','Sentiment'])
tweet_df['Tweet Id'] = ids_list
tweet_df['Date Created'] = dates
tweet_df['Tweet'] = tweet_list
tweet_df['Tweet_Text'] = cleaned_tweets_list
tweet_df['Sentiment'] = sent_list

engine = create_engine('postgresql://postgres:ameya@localhost:5432/tweet_hype')
tweet_df.to_sql("Wisdom of the Vanir", engine)

print("\nTime Taken:",(time.time()-start_time),"Seconds")


