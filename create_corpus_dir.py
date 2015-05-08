#!/usr/bin/env python

import sys
import nltk
import json
import os
import numpy
import re
import nltk.stem.porter
import nltk.corpus
from nltk.corpus import PlaintextCorpusReader
from nltk import cluster
from nltk.cluster import util
from nltk.cluster import api
from nltk.cluster import euclidean_distance
from nltk.cluster import cosine_distance

if len(sys.argv) != 2:
	print "Usage: python " + str(sys.argv[0]) + " filename.txt"
	print "Where filename.txt is the output of stream_listener.py"
	exit(-1)

filename = sys.argv[1]

# Make new dir for the corpus. If it already exists, we'll skip this part
corpusdir = filename + '_corpus/'
if not os.path.isdir(corpusdir):
	os.mkdir(corpusdir)

	f = open(filename, 'r')

	# Read all JSON objects, write out just the text to the corpus directory
	i = 1
	for jsonString in f:
		jsonObj = json.loads(jsonString)
		if ('text' in jsonObj):
			text_out_filename = str(i) + ".txt"

			text_out = open(corpusdir + text_out_filename, 'w')

			docstring = jsonObj['text'].encode('utf-8') + "\n"

			# Strip all urls
			urlRegex = r'(https?://[^\s]+)'
			urlList = re.findall(urlRegex, docstring)
			docstring = re.sub(urlRegex, '', docstring)

			# print docstring

		    # TODO: follow all URLs and append their content to docstring?
		    # TODO: remove all # symbols. it's messing up the clustering - overuse of hashtags gets its own cluster
		    
		    # lowercase all our words to cut down on dimensionality
		    # tokenization happens when we read the files using PlainTextCorpusReader
			docstring = docstring.lower()

			# TODO: strip out all rare words?

			text_out.write(docstring)
			text_out.close()
			i += 1

	f.close()

tweet_corpus = PlaintextCorpusReader(corpusdir, '.*\.txt')

unique_terms = list(set(tweet_corpus.words()))

# Adapted from http://www.csc.villanova.edu/~matuszek/spring2012/CorpusArray.py
# TODO: make it less reliant on globals
def BOW(document):
    #print type(document)
    document = nltk.Text(tweet_corpus.words(document))
    word_counts = []
    for word in unique_terms:
        word_counts.append(document.count(word))
    #print word_counts
    return word_counts

vectors = [numpy.array(BOW(f)) for f in tweet_corpus.fileids()]
print "Vectors created."
print "First 10 words are", unique_terms[:10]
print "First 10 counts for first document are", vectors[0][0:10]

CLUSTERS = 2
kmeans_clusterer = cluster.KMeansClusterer(CLUSTERS, cosine_distance)

print "Starting Clustering"
clusters = kmeans_clusterer.cluster(vectors, assign_clusters=False, trace=False)
#print 'Clustered:', vectors
#print 'As:', clusters
print "Number of clusters: ", kmeans_clusterer.num_clusters()
print "Means:", kmeans_clusterer.means()
print "Cluster names: ", str(kmeans_clusterer.cluster_names())


# Go through the docs in the same order as we did when we created feature vectors
# Create a dict of tweet => cluster ID
cluster_dict = {}
for i, fileid in enumerate(tweet_corpus.fileids()):
	tokens = tweet_corpus.words(fileid)
	document = nltk.Text(tokens)
	tweet = ' '.join([token.encode('utf-8') for token in tokens])
	cluster_id = kmeans_clusterer.classify_vectorspace(vectors[i])
	#print "Cluster: " + str(cluster_id)
	#print "\tTokens: " + tweet
	cluster_dict[tweet] = cluster_id

for i in range(CLUSTERS):
	for (tweet, cluster_id) in cluster_dict.iteritems():
		if i == cluster_id:
			print "Cluster " + str(i) + ": " + tweet
