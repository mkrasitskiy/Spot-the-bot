#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Project 

# In[1]:


import numpy as np
import pandas as pd
import networkx as nx
import glob
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

#from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'notebook')



from time import time

#######################################################################################################
print('Loading data...')

#Get the information
file_list = glob.glob(os.path.join(os.getcwd(), 
                                   '/home/rfreyrebonilya/Spotthebot/ebooks2', "*"))

corpus = []

#Put the raw data in a corpus
for file_path in file_list:
    with open(file_path, "r", encoding='utf8') as file:
        #text = file.read()
        text = file.readlines()
        #print(text)
        corpus.append(text)

print('Finished corpus 1...')      
        
#remove /n in lines
corpus2=[]
for group in corpus:
    aux= [i for i in group if i!='\n']
    corpus2.append(aux)

#remove /n at the end of lines
for group in corpus2:
    for i in range(len(group)):  
        group[i] = group[i].rstrip('\n')

print('Finished corpus 2...') 
        
#remove xml sign
corpus_final=[]
for group in corpus2:
    aux= [i for i in group if i.find('<')!=0 or i.find('>')!=0]
    corpus_final.append(aux)

# #Put everything in one corpus
# corpus_final=[]
# for group in corpus3:
#     l=0
#     j=0
#     while(l!=len(group)):
        
#         aux=[]
#         if(j>=len(group)):
#             break
#         while (group[j]!='ENDOFARTICLE.'):
#             aux.append(group[j])
#             j=j+1
#         #print(aux)
#         j=j+1
#         l=l+j
#         corpus_final.append(aux)
        
        


# In[ ]:


corpus_final[0]


# In[ ]:


#number of documents
len(corpus_final)


# In[ ]:


#corpus_final[0]


# In[ ]:

###############################################################################################################
print('First pre-process...')

#run a little sample
corpus_final=corpus_final[0:2000]

print('taking sample of:',len(corpus_final))

# In[ ]:


#One corpus qith raw documents( and not empty)
corpus_final1=[]

for doc in corpus_final:
    if len(doc)!=0:
        merged = functools.reduce(operator.iconcat, doc, )
        merged=merged.lower()
    
        #remove tags
        merged=re.sub("</?.*?_>"," <> ",merged)
    
        # remove special characters and digits
        merged=re.sub("(\\d|\\W)+"," ",merged)
               
        
        corpus_final1.append(merged)


# In[ ]:


corpus_final1[0]


# ## Data Pre-processing
# 
# We will perform the following steps:
# 
#     Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
#     Words that have fewer than 3 characters are removed.
#     All stopwords are removed.
#     Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
#     Words are stemmed — words are reduced to their root form.

# In[ ]:


import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
#import pyLDAvis
#import pyLDAvis.sklearn
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:

###############################################################################################################
print('Tokenizations...')

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

t0 = time()

data_words1 = list(sent_to_words(corpus_final1))
t1 = time()
print('time Tokenization:', t1 - t0)


# In[ ]:


#print(data_words1[0])


# In[ ]:


len(data_words1)


# ### Check the words in dictionary

# In[ ]:

###############################################################################################################
print('Check the words in dictionary...')
#Spanish dictionary
#dic = pd.read_csv('espanol.txt',encoding='latin-1').values
with open('espanol.txt',encoding="latin-1") as file:  
    dic = file.readlines()
    
dic = [i for i in dic if i!='\n']
for i in range(len(dic)):  
    dic[i] = dic[i].rstrip('\n')
for i in range(len(dic)):  
    dic[i] = gensim.utils.deaccent(dic[i])


# In[ ]:


len(dic)


# In[ ]:


dic[0:10]


# In[ ]:


#lista_repetidas=['referencias','','']


# In[ ]:


data_words=[]
t0 = time()
i=1
for doc in data_words1:
    aux=[]
    for word in doc: 
        if word in dic:
            aux.append(word)
    data_words.append(aux)
    if(i%10==0):
        print('% processed:',(i/len(data_words1))*100)
    i=i+1
t1 = time()
print('time:',t1 - t0)


# In[ ]:





# In[ ]:


#print(data_words[0])


# In[ ]:


len(data_words)
np.savetxt("data_wordsB2.csv", data_words, delimiter=",",fmt='%s')

# ## Lemmatization

# In[ ]:
###############################################################################################################
print('Lemmatization...')

import es_core_news_sm
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
nlp.max_length = 9000000


# In[ ]:



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
#nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

t0 = time()
         
# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
t1 = time()
print('time:',t1 - t0)


# In[ ]:


#data_lemmatized[0]


# In[ ]:


###############################################################################################################
print('Create the Books-Word matrix...')


# ## Create the Book-Word matrix

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
spanish_stopwords = stopwords.words('spanish')
#stemmer = SnowballStemmer('spanish')


# In[ ]:


vectorizer = CountVectorizer(analyzer='word',       
                             min_df=4,                        # minimum  occurences of a word 
                             stop_words= spanish_stopwords,             # remove stop words
                             #lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

t0 = time()
data_vectorized = vectorizer.fit_transform(data_lemmatized)
t1 = time()
print('time:',t1 - t0)


# In[ ]:


print('Size Book-Word matrix:', data_vectorized.shape)


# In[ ]:


###############################################################################################################
print('sample dictionary...')
print(vectorizer.get_feature_names()[0:10])


# In[31]:


# # Materialize the sparse data
# data_dense = data_vectorized.todense()

# # Compute Sparsicity = Percentage of Non-Zero cells
# print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")


# In[32]:


A = data_vectorized.toarray()
A = A.astype(float)


# In[ ]:





# ## SVD decomposition

# In[33]:

###############################################################################################################
print('SVD decomposition...')
k=600
print('s:',k)

from scipy.sparse.linalg import svds, eigs


t0 = time()
u, s, vt = svds(A,k=1000)

t1 = time()
print('time:',t1 - t0)


# In[ ]:





# ### A little representation
# 

# In[34]:



# for i in range(len(vectorizer.get_feature_names())):
#         fig = plt.gcf()
#         fig.set_size_inches(18.5, 10.5)
#         plt.text(vt.T[i,0], vt.T[i,1], vectorizer.get_feature_names()[i])
#         plt.xlim((-0.4,0.2))
#         plt.ylim((-0.1,0.5))


# In[ ]:

###############################################################################################################
print('Saving results...')

np.savetxt("uB2.csv", u, delimiter=",")


# In[ ]:


np.savetxt("sB2.csv", s, delimiter=",")


# In[ ]:


np.savetxt("vtB2.csv", vt, delimiter=",")


# In[ ]:


np.savetxt("dictB2.csv", vectorizer.get_feature_names(), delimiter=",",fmt='%s')


# In[ ]:


np.savetxt("A_B2.csv", A, delimiter=",")

