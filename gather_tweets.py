import tweepy

consumer_key = 'iFi38M09e6B41RMsndwojvNLf'
consumer_secret = 'TmtrQZK4e7FNss99uraM5CLqg96Fpvw0tQtUCDQn9nTzBqJ6um'
access_token = '2820632585-jG2aBWsjgkY7y8Wh4PSOjTPiz1ki54qFJNl8yMc'
access_token_secret = 'WJQpNLRorL8OpG5UCDxygFOIMasP0jH4Bif9fgFSpPzcS'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

for tweet in tweepy.Cursor(api.search, q="#Brexit", count=10, lang="en", since="2019-06-01",
                           tweet_mode='extended').items():
    print(tweet.created_at, tweet.retweeted_status.full_text if tweet.full_text.startswith("RT @") else tweet.full_text)
