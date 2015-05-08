#!/usr/bin/env python

# Adapted from http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

import nltk
import string
import os
import sys
import json
import re
from nltk import cluster
from nltk.cluster import util
from nltk.cluster import api
from nltk.cluster import euclidean_distance
from nltk.cluster import cosine_distance
import glob
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer

if len(sys.argv) != 2:
    print "Usage: python " + str(sys.argv[0]) + " filename.txt"
    print "Where filename.txt is the output of stream_listener.py"
    exit(-1)

# Given a list of labels and the corresponding text, print out the results of clustering
def printClusteringResults(labels_list, vectors_arrays):
    
    # First, print out Silhouette Coefficient to evaluate the clustering.
    silhouette_score_sample_size = int(round(len(labels_list) * 0.5))
    print "\n\nCalculating silhouette_score with sample size", silhouette_score_sample_size ,"..."
    print "Silhouette score: ", silhouette_score(vectors_arrays, labels, metric='euclidean',
        sample_size=silhouette_score_sample_size)

    # Go through the docs in the same order as we did when we created feature vectors
    # Create a dict of tweet => cluster ID
    cluster_dict = {}
    total_cluster_count = 0
    for i, fileid in enumerate(glob.glob(path + '/*.txt')):
        #print "file: " + fileid + " with sum: " + str(sum(vectors_arrays[i]))
        cluster_id = labels_list[i]
        #print "Cluster: " + str(cluster_id)
        #print "\tTokens: " + tweet
        cluster_dict[fileid] = cluster_id
        total_cluster_count = max(cluster_id, total_cluster_count)

    for i in range(CLUSTERS):
        for (fileid, cluster_id) in cluster_dict.iteritems():
            if i == cluster_id:
                with open(fileid, 'r') as tweet_doc:
                    print "Cluster " + str(i) + ": " + tweet_doc.read()

filename = sys.argv[1]

# Make new dir for the corpus. If it already exists, we'll skip this part
path = 'corpus_' + filename + '/'
if not os.path.isdir(path):
    print "Creating corpus files..."
    os.mkdir(path)

    f = open(filename, 'r')

    # Read all JSON objects, do some preprocessing
    i = 1
    for jsonString in f:
        jsonObj = json.loads(jsonString)
        if ('text' in jsonObj):
            # Sort into files - one file per user
            text_out_filename = str(jsonObj['user']) + ".txt"

            docstring = jsonObj['text'].encode('utf-8') + "\n"

            # Strip all urls
            urlRegex = r'(https?://[^\s]+)'
            urlList = re.findall(urlRegex, docstring)
            docstring = re.sub(urlRegex, '', docstring)

            # All space characters become just one space
            docstring = re.sub(r'(https?://[^\s]+)', '', docstring)

            # Strip all pound symbols. 
            # Otherwise, we'd cluster together all those who overuse hashtags
            docstring = re.sub(r'\#', '', docstring)

            # Replace all newlines and space chars with a single space
            docstring = re.sub(r'\s+', ' ', docstring)

            # Append to this user's file if it already exists
            writemode = 'w'
            if os.path.isfile(path + text_out_filename):
                writemode = 'a'

            with open(path + text_out_filename, writemode) as outfile:
                outfile.write(docstring)

            i += 1

    f.close()




token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        token_dict[file] = no_punctuation
        shakes.close()

'''   
#this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
document_term_matrix = tfidf.fit_transform(token_dict.values())
'''

print "Calling transform on the HashingVectorizer..."

hv = HashingVectorizer(tokenizer=tokenize, stop_words='english', n_features=2**10, non_negative=True)
document_term_matrix = hv.transform(token_dict.values())

tfidftrans = TfidfTransformer()

# Sparse matrix representation
vectors = tfidftrans.fit_transform(document_term_matrix)
vectors_as_arrays = vectors.toarray()

print "Vectors created."
print "First 10 counts for first document are", vectors_as_arrays[0][0:10]

CLUSTERS = 4

# Use Scikit-Learn clustering
clusterer = MiniBatchKMeans(n_clusters=CLUSTERS, init='k-means++')

print "Starting Clustering..."
clusters_model = clusterer.fit_predict(vectors)
labels = clusterer.labels_

print printClusteringResults(labels, vectors_as_arrays)





'''
# Use NLTK Clustering

kmeans_clusterer = cluster.KMeansClusterer(CLUSTERS, euclidean_distance, avoid_empty_clusters=True)

print "Starting Clustering..."
clusters = kmeans_clusterer.cluster(vectors_as_arrays, assign_clusters=False, trace=False)

#print 'Clustered:', vectors
#print 'As:', clusters
print "Number of clusters: ", kmeans_clusterer.num_clusters()
print "Means: ", kmeans_clusterer.means()
print "Number of features: ", len(kmeans_clusterer.means()[0])
print "Cluster names: ", str(kmeans_clusterer.cluster_names())

labels = []
for feature_array in vectors_as_arrays:
    labels.append(kmeans_clusterer.classify_vectorspace(feature_array))

printClusteringResults(labels, vectors_as_arrays)
'''


