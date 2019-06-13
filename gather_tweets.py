import tweepy
import key_secret

auth = tweepy.OAuthHandler(key_secret.consumer_key, key_secret.consumer_secret)
auth.set_access_token(key_secret.access_token, key_secret.access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

for tweet in tweepy.Cursor(api.search, q="#Brexit", count=10, lang="en", since="2019-06-01",
                           tweet_mode='extended').items():
    print(tweet.created_at, tweet.retweeted_status.full_text if tweet.full_text.startswith("RT @") else tweet.full_text)
