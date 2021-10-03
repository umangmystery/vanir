import en_core_web_sm
import enchant
import spacy
from kafka import KafkaConsumer
import json

import re
import string
import demoji
import preprocessor as p
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date, insert, BIGINT
from sqlalchemy.orm import sessionmaker
import psycopg2

topic_name = 'twitter_stream'

# Downloading english dictionary
nlp = en_core_web_sm.load()
d = enchant.Dict("en_US")

# Connecting to the database
# TODO: update credentials and use env variables
conn = psycopg2.connect("host=host.docker.internal dbname=test_db user=root password=root")
cursor = conn.cursor()

engine = create_engine('postgresql://root:root@host.docker.internal:5432/test_db')
Session = sessionmaker()
Session.configure(bind=engine)



consumer = KafkaConsumer(
    topic_name,
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     auto_commit_interval_ms =  5000,
     fetch_max_bytes = 128,
     max_poll_records = 100,
     value_deserializer=lambda x: json.loads(x.decode('utf-8')))


# create table if it doesnt exist
# create_table_query = "CREATE TABLE [IF NOT EXISTS] cleaned_tweets(id numeric PRIMARY KEY, cleaned_tweet)"
# insert_query = "INSERT INTO public.raw_tweets(id, raw_tweet) VALUES"
# insert_query = "INSERT INTO public.tweets(id, tweet) VALUES"
insert_query ="INSERT INTO public.aggregated_emotions('Id', 'Date Created', 'Clean tweet', 'Sentiment', 'Fear', 'Anger', 'Anticipation', 'Trust', 'Surprise', 'Positive', 'Negative', 'Sadness', 'Disgust', 'Joy')"


def tweet_cleaning(x):  # remove link
    # remove_RT = lambda x: re.compile('\#').sub('', re.compile('RT @').sub('@', x, count=1).strip())
    # x = remove_RT(x)
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
            if word[0] not in entity_prefixes:# and d.check(word) == True or word == keyword:
                words.append(word)

    return ' '.join(words)


def tweet_processing(clean_tweet):
    my_doc = nlp(clean_tweet)
    noun_phrases = []
    for chunk in list(my_doc.noun_chunks):
        noun_phrases.append(chunk.text)
    return noun_phrases


def sentiment_scores(clean_tweet):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(clean_tweet)

    if sentiment_dict['compound'] >= 0.05:
        return "Postive"

    elif sentiment_dict['compound'] <= - 0.05:
        return "Negative"
    else:
        return "Neutral"


def aggregate_emotion(noun_phrases):
    emotion_dict = {'fear': 0.0, 'anger': 0.0, 'anticip': 0.0, 'trust': 0.0, 'surprise': 0.0, 'positive': 0.0,
                    'negative': 0.0, 'sadness': 0.0, 'disgust': 0.0, 'joy': 0.0}
    emotion_counter_dict = {'fear': 0, 'anger': 0, 'anticip': 0, 'trust': 0, 'surprise': 0, 'positive': 0,
                            'negative': 0, 'sadness': 0, 'disgust': 0, 'joy': 0}
    total = 0
    counter = 0

    for chunk in noun_phrases:
        for noun in chunk:
            emotion = NRCLex(noun)
            for k1,v1 in emotion.affect_frequencies.items():
                for k2, v2 in emotion_dict.items():
                    if k1 == k2 and v1 > 0:
                        emotion_dict[k2] = v2 + v1

                        total = total + v1
                        for k3, v3 in emotion_counter_dict.items():
                            if k1 == k3:
                                counter += 1
                                emotion_counter_dict[k3] += 1

    return emotion_dict, emotion_counter_dict, counter, total


# Creating Polarity dataframe
# tweet_df = pd.DataFrame(columns=['Tweet ID', 'Date Created', 'Clean tweet', 'Sentiment',
#                                  'Fear', 'Anger', ' Anticipation', 'Trust',
#                                  'Surprise', 'Positive', 'Negative', 'Sadness', 'Disgust', 'Joy'])

Variable_tableName = "aggregated_emotions"
# if not engine.inspect(engine).has_table(engine, Variable_tableName):  # If table don't exist, Create.
metadata = MetaData(engine)
# Create a table with the appropriate Columns
table = Table(Variable_tableName, metadata,
      Column('id', BIGINT, primary_key=True, nullable=False),
      Column('date_created', Date), Column('clean_tweet', String),
      Column('sentiment', String), Column('fear', Float),
      Column('anger', Float), Column('anticipation', Float),
      Column('trust', Float), Column('surprise', Float),
      Column('positive', Float), Column('negative', Float),
      Column('sadness', Float), Column('disgust', Float),
      Column('joy', Float))
# Implement the creation
metadata.create_all(bind=None, tables=None, checkfirst=True)


for message in consumer:
    tweets = json.loads(json.dumps(message.value))
    if 'extended_tweet' in tweets.keys():
        text = tweets['extended_tweet']['full_text']
    else:
        text = tweets['text']

    # print(tweets)
    clean_text = tweet_cleaning(text)
    noun_phrases = tweet_processing(clean_text)
    polarity = sentiment_scores(clean_text)
    emotion_dict, emotion_counter_dict, counter, total = aggregate_emotion([noun_phrases])
    # print("Emotion Counter Dict: ", emotion_counter_dict)
    # print("Noun Phrases:", noun_phrases)
    # print("Emotion Dict", emotion_dict)

    try:
        with engine.connect() as conn:
            # ins = table.insert().values(Id= tweets['id'], Date_created= tweets['created_at'], clean_tweet= clean_text,
            #                 "Sentiment"= polarity, "Fear": emotion_counter_dict['fear'],
            #                 "Anger"=emotion_counter_dict['anger'],
            #                 "Anticipation": emotion_counter_dict['anticip'], "Trust": emotion_counter_dict['trust'],
            #                    "Surprise": emotion_counter_dict['surprise'], "Positive": emotion_counter_dict['positive'],
            #                    "Negative": emotion_counter_dict['negative'], "Sadness": emotion_counter_dict['sadness'],
            #                    "Disgust":emotion_counter_dict['disgust'], "Joy": emotion_counter_dict['joy'])
            # conn.execute(ins)
            result = conn.execute(insert(table),
                       [
                           {"id": tweets['id'], "date_created": tweets['created_at'], "clean_tweet": clean_text,
                            "sentiment": polarity, "fear": emotion_counter_dict['fear'],
                            "anger":emotion_counter_dict['anger'],
                            "anticipation": emotion_counter_dict['anticip'], "trust": emotion_counter_dict['trust'],
                               "surprise": emotion_counter_dict['surprise'], "positive": emotion_counter_dict['positive'],
                               "negative": emotion_counter_dict['negative'], "sadness": emotion_counter_dict['sadness'],
                               "disgust":emotion_counter_dict['disgust'], "joy": emotion_counter_dict['joy']
                            }
                       ])

            # conn.commit()
        # print(emotion_dict)
        # try:
        #     data_tuple = (tweets['id'], tweets['created_at'], clean_text, polarity,emotion_counter_dict['fear'],
        #                   emotion_counter_dict['anger'], emotion_counter_dict['anticip'],emotion_counter_dict['trust'],
        #                   emotion_counter_dict['surprise'], emotion_counter_dict['positive'], emotion_counter_dict['negative'],
        #                   emotion_counter_dict['sadness'], emotion_counter_dict['disgust'], emotion_counter_dict['joy'])
        #     # print(data_tuple)
        #     # Inserting Raw Data in postgres table
        #     cursor.execute(insert_query + str(data_tuple))
        # except:
        #     pass
        # finally:
        #     conn.commit()
    except:
        pass
