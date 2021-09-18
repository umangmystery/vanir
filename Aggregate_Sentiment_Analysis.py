import enchant
import spacy
from spacy.lang.en import English
from matplotlib import pyplot as plt
import tweepy
import pandas as pd
import numpy as np
import re
import string
from nrclex import NRCLex
import demoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
import en_core_web_sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
print("Libraries Imported")
start_time = time.time()

consumerKey = "8udhSgxLFBH99zgN06DuSYOQi"
consumerSecret = "pIPXXFRUqStC3Z6J1YMPdz7W4AwD8bioGK7hMueFdks7pm0OIc"
accessToken = "1083286165845954560-ceHxFmX1Rak89Y5UX7yFaJzOVnF97t"
accessTokenSecret = "MN9dnShe69FTwc0KxsBn9ooQN4QGtJZtMlG1vRKgVdus0"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)
print("API Authentication")

#nlp = English()
nlp = en_core_web_sm.load()
d = enchant.Dict("en_US")

def tweet_cleaning(x, keyword):  # remove link
    remove_RT = lambda x: re.compile('\#').sub('', re.compile('RT @').sub('@', x, count=1).strip())
    x = remove_RT(x)
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            x = x.replace(separator, ' ')
    words = []
    for word in x.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:# and d.check(word) == True or word == keyword:
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
    tweet_size = []
    dates = []
    print("Obtained Tweets, moving to cleaning")
    for tweet in tweets:
        tweet_list.append(tweet._json["full_text"])
        dates.append(tweet.created_at)
        cleaned_tweet = tweet_cleaning(tweet._json["full_text"], keyword)
        tweet_size.append(len(cleaned_tweet))
        cleaned_tweets_list.append(cleaned_tweet)

    return tweet_list, cleaned_tweets_list, tweet_size, dates


def replace_emoji(cleaned_tweets_list):
    for x in range(len(cleaned_tweets_list)) :
        if demoji.findall(cleaned_tweets_list[x]) :
            cleaned_tweets_list[x] = demoji.replace_with_desc(cleaned_tweets_list[x])
    return cleaned_tweets_list


def aggregate_emotion(token_tweets_list) :
    emotion_dict = {'fear': 0.0, 'anger': 0.0, 'anticip': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0, 'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
    emotion_counter_dict = {'fear': 0, 'anger': 0, 'anticip': 0, 'trust': 0, 'surprise': 0, 'positive': 0, 'negative': 0, 'sadness': 0, 'disgust': 0, 'joy': 0}
    counter = 0
    total = 0

    for tokens in token_tweets_list:
        for token in tokens:
            #print("\n", token)
            emotion = NRCLex(token)
            #print('\n', emotion.raw_emotion_scores, "- Datatype:", type(emotion.raw_emotion_scores))
            #print('\n', emotion.top_emotions, "- Datatype:", type(emotion.raw_emotion_scores))
            #print('\n', emotion.affect_frequencies, "- Datatype:", type(emotion.affect_frequencies))
            for k1,v1 in emotion.affect_frequencies.items():
                for k2, v2 in emotion_dict.items():
                    if k1 == k2 and v1 > 0:
                        emotion_dict[k2] = v2 + v1
                        total = total + v1
                        for k3, v3 in emotion_counter_dict.items():
                            if k1 == k3:
                                counter += 1
                                emotion_counter_dict[k3] += 1

    print("\nCounter:", counter, "Total:", total)
    return emotion_dict, emotion_counter_dict, counter, total


def percentage(part,whole):
    return 100 * float(part)/float(whole)


def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    pos = 0
    neg = 0
    neu = 0

    # print("\nOverall sentiment dictionary is : ", sentiment_dict)
    # print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    # print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    # print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")
    #
    # print("Sentence Overall Rated As", end=" ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        pos += 1

    elif sentiment_dict['compound'] <= - 0.05:
        neg += 1

    else:
        #print("Neutral")
        neu += 1
    return pos, neg, neu


keyword = input("Please enter keyword or hashtag to search: ")
noOfTweet = int(input("Please enter how many tweets to analyze: "))
tweet_list, cleaned_tweets_list, tweet_size, dates = get_tweets(keyword, noOfTweet)
cleaned_tweets_list = replace_emoji(cleaned_tweets_list)
print("Cleaned Tweets, calculating aggregate emotions")
token_tweets_list, noun_phrases_list = data_generation(cleaned_tweets_list)
print("Obtained tweets:", len(cleaned_tweets_list))

overall_pos = 0
overall_neg = 0
overall_neu = 0

for x in cleaned_tweets_list :
    #print("\n",x)
    pos, neg, neu = sentiment_scores(x)
    overall_pos = overall_pos + pos
    overall_neg = overall_neg + neg
    overall_neu = overall_neu + neu

print("Positive percentage = ", percentage(overall_pos, len(cleaned_tweets_list)))
print("Negative percentage = ", percentage(overall_neg, len(cleaned_tweets_list)))
print("Neutral percentage = ", percentage(overall_neu, len(cleaned_tweets_list)))

# percentages = [percentage(overall_pos, len(cleaned_tweets_list)), percentage(overall_neg, len(cleaned_tweets_list)), percentage(overall_neu, len(cleaned_tweets_list))]
# labels_list = ['Positive','Negative','Neutral']
# fig = plt.figure(figsize=(10,7))
# plt.pie(percentages, labels=labels_list)
# plt.show()


emotion_dict_noun, emotion_counter_dict_noun, counter_noun, total_noun = aggregate_emotion(noun_phrases_list)
# print("\nThe aggregate of noun phrases emotions for:", keyword)
# for k1,v1 in emotion_dict_noun.items() :
#     print("Emotion:", k1,"Score:", percentage(v1,total_noun))
print("\nThe percentage of noun phrases emotions for:", keyword)
emotion_percentages = []
for k2, v2 in emotion_counter_dict_noun.items() :
    p = percentage(v2,counter_noun)
    print("Emotion :", k2, "Percentage:", p)
    emotion_percentages.append(p)

# r = np.arange(len(emotion_counter_dict_noun))
# width = 0.25
# plt.bar(r, emotion_percentages, color = 'r',
#         width = width, edgecolor = 'black',
#         label='Emotion Percentages')
# plt.xticks(r + width/2,['Fear','Anger','Anticipation','Trust', 'Surprise', 'Positive', 'Negative', 'Sadness', 'Disgust', 'Joy'])
# plt.xlabel("Overall Emotions Detected")
# plt.ylabel("Number of tweets")
# plt.title("Sentiment Distributions")
# plt.legend()
# plt.show()

print("------ TIme: ", time.time() - start_time)



#Extra Code for reference:
#emotion_dict_reg,emotion_counter_dict_reg, counter_reg, total_reg = aggregate_emotion(token_tweets_list)
# print("\nThe aggregate of overall emotions for:", keyword)
# for k3,v3 in emotion_dict_reg.items() :
#     print("Emotion:", k3,"Score:", percentage(v3,total_reg))
# print("\nThe percentage of overall emotions for:", keywoiohrd)
# for k4, v4 in emotion_counter_dict_noun.items() :
#     print("Emotion :", k4, "Percentage:", percentage(v4,counter_reg))

