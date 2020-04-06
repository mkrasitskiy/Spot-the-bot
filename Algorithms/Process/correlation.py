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
#import measures1 as nolds1

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

#import Complexity_entropy as complexity
from math import factorial
#from nolitsa import data, dimension
#import hdbscan
#from sklearn.cluster import DBSCAN
import scipy.stats




print('Loading data...')

#Import Data Books
sb = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/balls1000_0/sB.csv', delimiter=',')
vtb = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/balls1000_0/vtB.csv', delimiter=',')
ub = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/balls1000_0/uB.csv', delimiter=',')
wb = ub*sb


print('Loaded books...')


# In[ ]:


def reconst(lib,w,vt,k1,k):
    reconstruct = np.zeros(vt[0,:].shape) 
    r=(np.array(range(k1,k))+1)*(-1)
    for l in r:
        reconstruct += vt[l,:]*w[lib,l]
    return reconstruct



def generate_correlation(w,vt,s):
    #size=5
    size=s
    k1_=5
    k_=5
        
    
    Books_corr=[]

    for i in range(size):
        corr_=[]        
        for k1 in range(0,k1_):
                for k in range(0,k_):
                        if(k>k1):
                                book_1=reconst(i,w,vt,k1,k1+1)
                                book_2=reconst(i,w,vt,k,k+1)		
                                corr=scipy.stats.pearsonr(book_1, book_2)[0]
                                corr_.append(corr)
                                tittle='book'+str(i)+'k'+str(k1)+'k1'+str(k)
                                plt.plot(book_1, label='book'+str(i)+'k'+str(k1))
                                plt.plot(book_2,label='book'+str(i)+'k'+str(k))
                                plt.legend()
                                plt.savefig(os.path.join(os.getcwd(), tittle+'.pdf'))
                                plt.clf()

        Books_corr.append(corr_)

                
    return Books_corr

def plot(w,vt,s):
        size=s
        for i in range(size):
                book=reconst(i,w,vt,0,1000)
                tittle='book'+str(i)
                plt.plot(book, label='book'+str(i))

                plt.legend()
                plt.savefig(os.path.join(os.getcwd(), tittle+'.pdf'))
                plt.clf()

		

# In[ ]:


t0 = time()
Books_corr = generate_correlation(wb,vtb,10)
plot(wb,vtb,10)
t1 = time()
print('Time process books:',t1 - t0)


# In[ ]:


#np.savetxt("cluster_books.csv", cluster_books, delimiter=",")


# In[ ]:


dumped = json.dumps(Books_corr, cls=NumpyEncoder)
with open('Books_corr.txt','w') as myfile:
    json.dump(dumped,myfile)


# In[ ]:





# In[ ]:




