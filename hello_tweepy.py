#!/usr/bin/env python

import tweepy
import ConfigParser

config = ConfigParser.ConfigParser()

# This file has been added to gitignore
config.readfp(open(r'config.txt'))

consumer_key = config.get("Keys", "consumer_key")
consumer_secret = config.get("Keys", "consumer_secret")
access_token = config.get("Keys", "access_token")
access_token_secret = config.get("Keys", "access_token_secret")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text