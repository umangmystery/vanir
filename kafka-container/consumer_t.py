from kafka import KafkaConsumer
import json
import psycopg2
import re
import string
import demoji

topic_name = 'twitter_stream'

# Connecting to the database
# TODO: update credentials and use env variables
conn = psycopg2.connect("host=host.docker.internal dbname=test_db user=root password=root")
cursor = conn.cursor()      

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
insert_query = "INSERT INTO public.tweets(id, tweet) VALUES"

def tweet_cleaning(x):  # remove link
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

def replace_emoji(text):   
    if demoji.findall(text) :
        text = demoji.replace_with_desc(text)
    return text



for message in consumer:
    tweets = json.loads(json.dumps(message.value))
    if 'extended_tweet' in tweets.keys():
        text = tweets['extended_tweet']['full_text']
    else:
        text = tweets['text']


    clean_text = tweet_cleaning(text)
    clean_text = replace_emoji(clean_text)

    keyword = tweets['keyword']
    
    try:
        data_tuple = (tweets['id'], clean_text)
        # Inserting Raw Data in postgres table
        cursor.execute(insert_query + str(data_tuple))
    except:
        pass
    finally:
        conn.commit()
