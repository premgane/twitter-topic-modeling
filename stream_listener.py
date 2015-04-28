#!/usr/bin/env python

from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import sys
import ConfigParser

config = ConfigParser.ConfigParser()

# This file has been added to gitignore
config.readfp(open(r'config.txt'))

consumer_key = config.get("Keys", "consumer_key")
consumer_secret = config.get("Keys", "consumer_secret")
access_token = config.get("Keys", "access_token")
access_token_secret = config.get("Keys", "access_token_secret")

'''
Adapted from the very helpful example in the tweepy repo:
https://github.com/tweepy/tweepy/blob/master/examples/streaming.py
'''
class StreamListener(StreamListener):
    ''' A basic listener that just prints interesting information from received tweets to stdout.
    '''

    # This set contains all the tweet IDs that we've already printed
    # This will reduce the number of RT'd tweets' text we'll print
    printed_rts = set()

    def on_data(self, data):
        tweet = json.loads(data)

        out = {}

        # TODO: create function make the data population code less verbose
        # TODO: minimize the keys' string lengths to reduce the amount of output

        # This is an RT of an existing status, so let's get that info too
        if "retweeted_status" in tweet:
            rt_source = tweet["retweeted_status"]
            out["rt"] = True

            # Do some bookkeeping to prevent printing out too much redundant info
            rt_source_id = rt_source["id"]
            out["id"] = rt_source_id

            if rt_source_id not in StreamListener.printed_rts:
                StreamListener.printed_rts.add(rt_source_id)
                out["text"] = rt_source["text"]
                out["created"] = rt_source["created_at"]
            
            out["user"] = rt_source["user"]["id"]
            out["rts"] = rt_source["retweet_count"]
            out["favs"] = rt_source["favorite_count"]
            
            
        elif "created_at" in tweet: # Sometimes, crucial fields are missing. In those cases, skip.

            # Some basic tweet information
            out["created"] = tweet["created_at"]
            out["id"] = tweet["id"]
            out["text"] = tweet["text"]

            if "in_reply_to_status_id" in tweet and tweet["in_reply_to_status_id"] is not None:
                out["reply2id"] = tweet["in_reply_to_status_id"]
                out["reply2user"] = tweet["in_reply_to_user_id"]

            if tweet["retweet_count"] > 0:
                out["rts"] = tweet["retweet_count"]

            if tweet["favorite_count"] > 0:
                out["favs"] = tweet["favorite_count"]
            
            out["user"] = tweet["user"]["id"]

        else:
            return True
        
        print(json.dumps(out))
        return True

    def on_error(self, status):
        # Simply write to stderr and move on
        # TODO: more robust error handling based on common error modes
        sys.stderr.write(status)
        return

if __name__ == '__main__':
    listener = StreamListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, listener)

    # Just get tweets related to the Baltimore protests, since those are the most interesting hashtags right now
    # (2015-04-28, 1:00 AM)
    stream.filter(track=['#Baltimore', '#BaltimoreRiots'])