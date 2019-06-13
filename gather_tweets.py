import key_secret
import tweepy
import csv


class TwitterMining:
    def __init__(self):
        self.__tweets_csv_path = '\\data\\Twitter\\tweets.csv'
        self.__auth = tweepy.OAuthHandler(key_secret.consumer_key, key_secret.consumer_secret)
        self.__auth.set_access_token(key_secret.access_token, key_secret.access_token_secret)
        self.__api = tweepy.API(self.__auth, wait_on_rate_limit=True)

    def mine_data(self, query="#Brexit", since="2019-06-01", count=10, lang="en"):
        with open(self.__tweets_csv_path, mode='a') as file:
            file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for tweet in tweepy.Cursor(self.__api.search, q=query, count=count, lang=lang,
                                       since=since, tweet_mode='extended').items():
                created_at = tweet.created_at
                text = tweet.retweeted_status.full_text if tweet.full_text.startswith("RT @") else tweet.full_text
                file.writerow([created_at, text])


tm = TwitterMining()
tm.mine_data()
