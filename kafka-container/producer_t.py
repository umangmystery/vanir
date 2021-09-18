"""API ACCESS KEYS"""

# access_token = ""
# access_token_secret = ""
# consumer_key = ""
# consumer_secret = ""




from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import KafkaProducer
import json 
producer = KafkaProducer(bootstrap_servers='localhost:9092') #Same port as your Kafka server


topic_name = "twitter_stream"


class twitterAuth():
    """SET UP TWITTER AUTHENTICATION"""

    def authenticateTwitterApp(self):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        return auth



class TwitterStreamer():

    """SET UP STREAMER"""
    def __init__(self, stop_limit):
        self.twitterAuth = twitterAuth()
        self.limit = stop_limit

    def stream_tweets(self):
        # while True:
        listener = ListenerTS(stop_limit = self.limit) 
        auth = self.twitterAuth.authenticateTwitterApp()
        stream = Stream(auth, listener)
        stream.filter(track=["Apple"], stall_warnings=True, languages= ["en"])


class ListenerTS(StreamListener):
    
    def __init__(self, stop_limit):
        self.limit = stop_limit

    counter = 0

    def on_data(self, raw_data):
            
            # print("Adding to queue", raw_data)
            ListenerTS.counter += 1
            print(ListenerTS.counter)

            # Converting json to dict inorder to append keyword to the raw data
            tw_dict = json.loads(raw_data)
            tw_dict['keyword'] = 'Apple'
            raw_data = json.dumps(tw_dict)

            producer.send(topic_name, str.encode(raw_data))
            
            if ListenerTS.counter == self.limit:
                return False
            return True


if __name__ == "__main__":
    TS = TwitterStreamer(stop_limit = 1000)
    TS.stream_tweets()