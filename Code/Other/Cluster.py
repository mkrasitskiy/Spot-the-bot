#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd


import os
import itertools

import functools


import operator
import re


import scipy as sp

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

from scipy.linalg import toeplitz

import math

from matplotlib import pyplot as plt



from bs4 import BeautifulSoup
from time import time

import nolds
import skedm as edm
from scipy.signal import argrelextrema
import json
import measures1 as nolds1

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

import Complexity_entropy as complexity
from math import factorial
from nolitsa import data, dimension
#import hdbscan
from sklearn.cluster import DBSCAN


# In[3]:




from scipy.special import gamma
from sklearn.neighbors import KDTree
from collections import defaultdict
from tqdm import tqdm_notebook
from subprocess import call
from sklearn.preprocessing import StandardScaler


from mpl_toolkits.mplot3d import axes3d, Axes3D
import os
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (2 * 13, 2 * 6)


# In[4]:


class WishartClassifier:
    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.significance_level = significance_level  # Significance level
    
    def fit(self, X):
        from sklearn.neighbors import KDTree
        kdt = KDTree(X, metric='euclidean')
        
        #add one because you are your neighb.
        distances, neighbors = kdt.query(X, k = self.wishart_neighbors + 1, return_distance = True)
        neighbors = neighbors[:, 1:]
        inv_neighbors = [[] for i in range(neighbors.shape[0])]
        for i in range(neighbors.shape[0]):
            for j in neighbors[i]:
                inv_neighbors[j].append(i)
        
        distances = distances[:, -1]
        indexes = np.argsort(distances)
        
        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype = int) - 1
        
        #index in tuple
        #min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)
               
        for index in indexes:
            neighbors_clusters = np.concatenate([self.object_labels[neighbors[index]], self.object_labels[neighbors[index]]])
            unique_clusters = np.unique(neighbors_clusters).astype(int)
            unique_clusters = unique_clusters[unique_clusters != -1]
            
            if len(unique_clusters) == 0:
                self._create_new_cluster(index, distances[index])
            else:
                max_cluster = unique_clusters[-1]
                min_cluster = unique_clusters[0]
                if max_cluster == min_cluster:
                    if self.clusters[max_cluster][-1] < 0.5:
                        self._add_elem_to_exist_cluster(index, distances[index], max_cluster)
                    else:
                        self._add_elem_to_noise(index)
                else:
                    my_clusters = self.clusters[unique_clusters]
                    flags = my_clusters[:, -1]
                    if np.min(flags) > 0.5:
                        self._add_elem_to_noise(index)
                    else:
                        significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                        significan *= self.wishart_neighbors
                        significan /= size
                        significan /= np.power(np.pi, dim / 2)
                        significan *= gamma(dim / 2 + 1)
                        significan_index = significan >= self.significance_level
                    
                        significan_clusters = unique_clusters[significan_index]
                        not_significan_clusters = unique_clusters[~significan_index]
                        significan_clusters_count = len(significan_clusters)
                        if significan_clusters_count > 1 or min_cluster == 0:
                            self._add_elem_to_noise(index)
                            self.clusters[significan_clusters, -1] = 1
                            for not_sig_cluster in not_significan_clusters:
                                if not_sig_cluster == 0:
                                    continue
                            
                                for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                    self._add_elem_to_noise(bad_index)
                                self.clusters_to_objects[not_sig_cluster].clear()
                        else:
                            for cur_cluster in unique_clusters:
                                if cur_cluster == min_cluster:
                                    continue
                            
                                for bad_index in self.clusters_to_objects[cur_cluster]:
                                    self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                self.clusters_to_objects[cur_cluster].clear()
                            self._add_elem_to_exist_cluster(index, distances[index], min_cluster)
                
        return self.clean_data()
    
    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq :  index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype = int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result
    
    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)
    
    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis = 0)
    
    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)


# In[5]:


def answers_to_dict(labels):
    res = defaultdict(set)
    for index, label in enumerate(labels):
        res[label].add(index)
    return res

def match_cluster(real, predict):
    real_dict = answers_to_dict(real)
    predict_dict = answers_to_dict(predict)
    answers = [0] * (max(predict) + 1)
    
    for cluster_index, real_set in real_dict.items():
        real_set = real_dict[cluster_index]
        best = None
        best_size = 0
        for predict_index, predict_set in predict_dict.items():
            if predict_index == 0:
                continue
            size = len(real_set & predict_set)
            if size > best_size:
                best_size = size
                best = predict_index

        if best is not None:
            answers[best] = cluster_index
            del predict_dict[best]
    
    for index, val in enumerate(predict):
        if val == 0:
            continue
        if answers[val] == 0:
            predict[index] = -1
        else:
            predict[index] = answers[val]


# In[6]:


def gen_samples(count, dim):
    X = np.concatenate([np.random.normal(size = (SIZE, dim)) + 15 * np.random.normal(size = (dim))                        for _ in range(count)])
    true = []
    for i in range(count):
        true += [i + 1] * SIZE
    return X, true


# In[ ]:


print('Loading data...')

#Import Data Books
sb = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/b3600s1000/sB.csv', delimiter=',')
vtb = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/b3600s1000/vtB.csv', delimiter=',')
ub = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/b3600s1000/uB.csv', delimiter=',')
wb = ub*sb


print('Loaded books...')


# In[ ]:


def reconst(lib,w,vt,k1,k):
    reconstruct = np.zeros(vt[0,:].shape)    
    for l in range(k1,k):
        reconstruct += vt[l,:]*w[lib,l]
    return reconstruct


# In[12]:


#sig_level = np.logspace(-3, 3, 6)
#sig = np.round(sig_level, 4)
#sig


# In[ ]:


def generate_cluster(w,vt,first_eig,last_eig):
    size=w.shape[0]
    
    neighbors = [1, 3, 5, 7, 10, 30]
    sig_level = np.linspace(0.01,1,20)
    
    
    Books=[]

    for i in range(size):
        book=reconst(i,w,vt,first_eig,last_eig)
        if np.sum(book)!=0:
            Books.append(book)
    
    Books=np.array(Books)
    Cluster_all=[]
    Cluster_labels=[]
    
    
    for sig in sig_level:
        for neig in neighbors:
        
            
            wishart = WishartClassifier(significance_level = sig, wishart_neighbors = neig)
            
            predict = wishart.fit(Books)
            
            Cluster_labels.append(predict)
            
            clusters = [[] for i in range(np.max(predict))]
            for index, cluster in enumerate(predict):
                clusters[cluster - 1].append(index)
            
            Cluster_all.append(clusters)
            
            for cluster in clusters:
                if len(cluster) == 1:
                    print('Dims =',dims, 'Clusters =', cluster, 'Neighboors =', neighboors, 'Sig =', sig)
            
            #means = np.empty((len(clusters), X.shape[1]))
            #for index, cluster_elems in enumerate(clusters):
            #    means[index] = np.mean(X[cluster_elems], axis = 0)
            
        
    return Cluster_all,Cluster_labels


# In[ ]:


t0 = time()
cluster_books,Cluster_labels = generate_cluster(wb,vtb,0,1000)
t1 = time()
print('Time process books:',t1 - t0)


# In[ ]:


#np.savetxt("cluster_books.csv", cluster_books, delimiter=",")


# In[ ]:


dumped = json.dumps(cluster_books, cls=NumpyEncoder)
with open('cluster_books.txt','w') as myfile:
    json.dump(dumped,myfile)

dumped = json.dumps(Cluster_labels, cls=NumpyEncoder)
with open('Cluster_labels.txt','w') as myfile:
    json.dump(dumped,myfile)



# In[ ]:





# In[ ]:




