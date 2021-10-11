"""API ACCESS KEYS"""

# access_token = ""
# access_token_secret = ""
# consumer_key = ""
# consumer_secret = ""

consumer_key = "8udhSgxLFBH99zgN06DuSYOQi"
consumer_secret = "pIPXXFRUqStC3Z6J1YMPdz7W4AwD8bioGK7hMueFdks7pm0OIc"
access_token = "1083286165845954560-ceHxFmX1Rak89Y5UX7yFaJzOVnF97t"
access_token_secret = "MN9dnShe69FTwc0KxsBn9ooQN4QGtJZtMlG1vRKgVdus0"


from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import KafkaProducer
import time
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

t_end = time.time() + 10

class ListenerTS(StreamListener):
    
    def __init__(self, stop_limit):
        self.limit = stop_limit

    counter = 0

    def on_data(self, raw_data):
            
            # print("Adding to queue", raw_data)
            ListenerTS.counter += 1
            print(ListenerTS.counter)
            producer.send(topic_name, str.encode(raw_data))
            
            if time.time()>= t_end:
                return False
            return True


if __name__ == "__main__":
    TS = TwitterStreamer(stop_limit = 10)
    TS.stream_tweets()