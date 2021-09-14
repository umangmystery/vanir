
from textblob import TextBlob
from spacy.lang.en import English
import tweepy
import pandas as pd
import numpy as np
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
print("Libraries Imported")

consumerKey = "8udhSgxLFBH99zgN06DuSYOQi"
consumerSecret = "pIPXXFRUqStC3Z6J1YMPdz7W4AwD8bioGK7hMueFdks7pm0OIc"
accessToken = "1083286165845954560-ceHxFmX1Rak89Y5UX7yFaJzOVnF97t"
accessTokenSecret = "MN9dnShe69FTwc0KxsBn9ooQN4QGtJZtMlG1vRKgVdus0"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)

print("API Authentication Successful")
nlp = English()

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
            if word[0] not in entity_prefixes:
                words.append(word)

    return ' '.join(words)
print("Created - tweet_cleaning")

def tweet_sentiment_analysis(x):
    the_tweet = TextBlob(x)
    pol = the_tweet.sentiment.polarity
    sub = the_tweet.sentiment.subjectivity
    score = SentimentIntensityAnalyzer().polarity_scores(x)
    neg = score['neg']
    pos = score['pos']
    neu = score['neu']
    comp = score['compound']
    sentiment = ''
    if pos > neg :
        sentiment = 'Positive'
    elif pos < neg :
        sentiment = 'Negative'
    else :
        sentiment = 'Neutral'
    return pol, sub, score, neg, pos, neu, comp, sentiment
    #return int(pol*100), int(sub*100), score, int(neg*100),int(pos*100) , int(neu*100), int(comp*100),sentiment
print("Created - tweet_sentiment_analysis")

def tweet_tokenizer(x):
    text = x
    my_doc = nlp(text)
    tokens = []
    for token in my_doc:
        tokens.append(token.text)

    return tokens
print("Created - tweet_tokenizer")

def percentage(part,whole):
    return 100 * float(part)/float(whole)
print("Created - percentage")

def get_sentiment_percentage(cleaned_tweets_list, positive_tweet_list, negative_tweet_list, neutral_tweet_list) :
    positive_percentage = percentage(len(positive_tweet_list), len(cleaned_tweets_list))
    negative_percentage = percentage(len(negative_tweet_list), len(cleaned_tweets_list))
    neutral_percentage = percentage(len(neutral_tweet_list), len(cleaned_tweets_list))
    return positive_percentage, negative_percentage, neutral_percentage
print("Created - get_sentiment_percentage")

def final_data_generation(cleaned_tweets_list, tweet_size):
    counter = 0
    polarity_list = []
    subjectivity_list = []
    positive_tweet_list = []
    negative_tweet_list = []
    neutral_tweet_list = []
    score_tweet_list = []
    positivty = []
    negativity = []
    neutralness = []
    compoundness = []

    for x in cleaned_tweets_list:
        if counter % 500 == 0:
            print("Counter -", counter,"/",(len(cleaned_tweets_list) - counter))
        tweet_pol, tweet_sub, score, neg, pos, neu, comp, sentiment = tweet_sentiment_analysis(x)
        positivty.append(pos)
        negativity.append(neg)
        neutralness.append(neu)
        compoundness.append(comp)
        polarity_list.append(tweet_pol)
        subjectivity_list.append(tweet_sub)
        score_tweet_list.append(sentiment)

        if pos > neg:
            positive_tweet_list.append(x)
        elif neg > pos:
            negative_tweet_list.append(x)
        elif pos == neg:
            neutral_tweet_list.append(x)
        counter += 1



    positive_percentage, negative_percentage, neutral_percentage = get_sentiment_percentage(cleaned_tweets_list,
                                                                                            positive_tweet_list,
                                                                                            negative_tweet_list,
                                                                                            neutral_tweet_list)

    return polarity_list,subjectivity_list,positive_percentage, negative_percentage, neutral_percentage,score_tweet_list,positivty,negativity,neutralness,compoundness
print("Created - final_data_generation")

def get_tweets(keyword, noOfTweet):
    tweets = tweepy.Cursor(api.search, q=keyword, lang="en").items(noOfTweet)
    tweet_list = []
    cleaned_tweets_list = []
    tweet_size = []
    for tweet in tweets:
        tweet_list.append(tweet.text)
        cleaned_tweet = tweet_cleaning(tweet.text)
        tweet_size.append(len(cleaned_tweet))
        cleaned_tweets_list.append(cleaned_tweet)

    return tweet_list, cleaned_tweets_list, tweet_size
print("Created - get_tweets")

# def get_best_randomstate(X,y):
#     ts_score=[]
#     for j in range(1000):
#         X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=j)
#         logreg.fit(X_train,y_train)
#         y_pred=logreg.predict(X_test)
#         ts_score.append(metrics.accuracy_score(y_test, y_pred))
#
#     J= ts_score.index(np.max(ts_score))
#     return J
# print("Created - get_best_randomstate")

#Cleaning the Data
print("\nData cleaning Started\n")
df_training = pd.read_csv(r'C:\Users\ameya\PycharmProjects\Predictive_Sentiment_Analysis\Kaggle_Twitter_Sentiment_Analysis.csv', encoding='cp1252')
df_training.drop_duplicates(inplace=True)
print("Imported DF and dropped duplicates")
training_tweet_list = df_training["@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"]
training_cleaned_tweets = []
for tweet in training_tweet_list :
    cleaned_tweet = tweet_cleaning(tweet)
    training_cleaned_tweets.append(cleaned_tweet)
print("Cleaned Tweets")
print(len(training_cleaned_tweets))
# for i in range(20):
#     print(i, "-", training_cleaned_tweets[i])


#Changing the way data is prepared
polarity_list,subjectivity_list,positive_percentage, negative_percentage, neutral_percentage,score_tweet_list,positivty,negativity,neutralness,compoundness = final_data_generation(training_cleaned_tweets, len(training_cleaned_tweets))
polarity_list = np.array(polarity_list)
# subjectivity_list = np.array(subjectivity_list)
# positive_percentage = np.array(positive_percentage)
# negative_percentage = np.array(negative_percentage)
# neutral_percentage = np.array(neutral_percentage)
# score_tweet_list = np.array(score_tweet_list)
# positivty = np.array(positivty)
# negativity = np.array(negativity)
# neutralness = np.array(neutralness)
# compoundness = np.array(compoundness)
df_training = pd.DataFrame(columns=['Tweet', 'Polarity', 'Subjectivity', 'Sentiment', 'Positivity', 'Negativity', 'Neutralness', 'Compoundness'])
df_training['Tweet'] = training_cleaned_tweets
df_training['Polarity'] = polarity_list
df_training['Subjectivity'] = subjectivity_list
df_training['Sentiment'] = score_tweet_list
df_training['Positivity'] = positivty
df_training['Negativity'] = negativity
df_training['Neutralness'] = neutralness
df_training['Compoundness'] = compoundness
df_training.drop_duplicates(inplace=True)
print(df_training.head(10))
enc = LabelEncoder()
df_training['Sentiment'] = enc.fit_transform(df_training['Sentiment'])
print("df_training.shape", df_training.shape)
print(df_training.head())
feature_cols = ['Subjectivity','Sentiment','Positivity','Negativity','Compoundness']
df_training.to_csv(index = False)
X = df_training[feature_cols] # Features
y = df_training.Polarity # Target variable
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=1234)
#Trying using pytorch
#Scaling the data
n_samples, n_features = X.shape
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(np.asarray(X_train.astype(np.float32)))
X_test = torch.from_numpy(np.asarray(X_test.astype(np.float32)))
y_train = torch.from_numpy(np.asarray(y_train.astype(np.float32)))
y_test = torch.from_numpy(np.asarray(y_test.astype(np.float32)))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#model

class LogisticRegression(nn.Module) :
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(n_features)

#Loss and Optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#Training Loop
num_epochs = 100
for epoch in range(num_epochs) :
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    #backward pass
    loss.backward()

    #updates
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0 :
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad() :
    y_predicted = model(X_test)
    y_predicted_clss = y_predicted.round()
    acc = y_predicted_clss.eq(y_test).sum()/float(y_test.shape[0])
    print(f"Accuracy ={acc:.4f}")

#_____________________________________________________________________________________________________________________________________________________________________________________
# polarity_list,subjectivity_list,positive_percentage, negative_percentage, neutral_percentage,score_tweet_list,positivty,negativity,neutralness,compoundness = final_data_generation(training_cleaned_tweets[:500], len(training_cleaned_tweets[:500]))
# df_training = pd.DataFrame(columns=['Tweet', 'Polarity', 'Subjectivity', 'Sentiment', 'Positivity', 'Negativity', 'Neutralness', 'Compoundness'])
# df_training['Tweet'] = training_cleaned_tweets[0:500]
# df_training['Polarity'] = polarity_list
# df_training['Subjectivity'] = subjectivity_list
# df_training['Sentiment'] = score_tweet_list
# df_training['Positivity'] = positivty
# df_training['Negativity'] = negativity
# df_training['Neutralness'] = neutralness
# df_training['Compoundness'] = compoundness
# df_training.drop_duplicates(inplace=True)
# df_training.shape
# enc = LabelEncoder()
# df_training['Sentiment'] = enc.fit_transform(df_training['Sentiment'])
# print("df_training.shape", df_training.shape)
# print(df_training.head())
#
# feature_cols = ['Subjectivity','Sentiment','Positivity','Negativity','Compoundness']
# X = df_training[feature_cols] # Features
# y = df_training.Polarity # Target variable
# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=1234)
# logreg = LogisticRegression(verbose=True)
# J = get_best_randomstate(X, y)
# print("Best randomstate:", J)
# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=J)
# logreg.fit(X_train,y_train)
#
# y_pred=logreg.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Getting Tweets

# keyword = input("Please enter keyword or hashtag to search: ")
# noOfTweet = int(input ("Please enter how many tweets to analyze: "))
# tweet_list, cleaned_tweets_list, tweet_size = get_tweets(keyword, noOfTweet)
# polarity_list,subjectivity_list,positivty, positive_percentage, negative_percentage, neutral_percentage, score_tweet_list, positivty, negativity, neutralness, compoundness = final_data_generation(cleaned_tweets_list, tweet_size)
# print("Cleaned Tweets:", len(cleaned_tweets_list))
# df_tweetdata = pd.DataFrame(columns=['Tweet', 'Polarity', 'Subjectivity', 'Sentiment', 'Positivity', 'Negativity', 'Neutralness', 'Compoundness'])
# df_tweetdata['Tweet'] = cleaned_tweets_list
# df_tweetdata['Polarity'] = polarity_list
# df_tweetdata['Subjectivity'] = subjectivity_list
# df_tweetdata['Sentiment'] = score_tweet_list
# df_tweetdata['Positivity'] = positivty
# df_tweetdata['Negativity'] = negativity
# df_tweetdata['Neutralness'] = neutralness
# df_tweetdata['Compoundness'] = compoundness
# print(df_tweetdata.shape)
# df_tweetdata.drop_duplicates(inplace=True)
# df_tweetdata.shape
#
# print("Percentage of Positive:\t",positive_percentage)
# print("Percentage of Negative:\t",negative_percentage)
# print("Percentage of Neutral:\t",neutral_percentage)
#
#
# enc = LabelEncoder()
# df_tweetdata.Sentiment = enc.fit_transform(df_tweetdata.Sentiment)
# print(df_tweetdata.head(20))

