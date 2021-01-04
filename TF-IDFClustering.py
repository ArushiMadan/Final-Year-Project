
"""
Testing a classification system based on TF-IDF and Clustering
@author: arushimadan

"""
#Importing all necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.metrics import silhouette_samples, silhouette_score

#Importing data from 796 review text files
import os
#Can change stopDir to stemDir, if want to run clustering on stemmed files
path = "/Users/arushimadan/Desktop/Dissertation/WebScrapedData/stopDir/"

_, dirs, files = next(os.walk(path))
document = []

for file in files:

    with open(os.path.join(path, file), 'r') as f:
      line = f.readlines()
      line = [l.strip('\n') for l in line]
      line = ' '.join(line)
      document.append(line)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(document)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2)
new_X = svd.fit_transform(X)
plt.scatter(new_X[:,0], new_X[:,1])
plt.title('Truncated SVD')
plt.show()

wcss=[]

#Setting parameters for clustering
for true_k in range (2,24):
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
    model.fit(X)
    cluster_labels = model.fit_predict(X)
    inertia=model.inertia_
    print(model.inertia_)
    wcss.append(model.inertia_)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", true_k,
          "The average silhouette_score is :", silhouette_avg)
    

#Centroids and terms
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


#Prints clusters and terms within them
#for i in range(true_k):
# print('Cluster %d:' %i),
# for ind in order_centroids[i, :10]:
#     print('%s'% terms[ind])

#Elbow method to understand inertia vs clusters
plt.plot(range(len(wcss)), wcss, 'b')
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#Test a prediction
print('\n')
print("Prediction")
X = vectorizer.transform(["I am unable to login. Keeps throwing the wrong pinsentry error!"])
predicted = model.predict(X)
print(predicted)
