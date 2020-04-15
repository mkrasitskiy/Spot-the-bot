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


from bs4 import BeautifulSoup
from time import time

def flat(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list

#######################################################################################################
print('Loading data...')


# Get the name of txt archives
file_list = []
with open('/home/rfreyrebonilya/Spotthebot/Bot_Human/es/truth.txt', "r", 
          encoding='utf8') as file:
    #text = file.read()
    text = file.readlines()
    #print(text)
    file_list.append(text)
    
file_list=flat(file_list)

#remove /n at the end of lines
file_list2=[]
for file in file_list: 
    file = file.rstrip('\n')
    file_list2.append(file)

#Split in human and bot and make the file path
file_list_human=[]
file_list_bot=[]

directory='/home/rfreyrebonilya/Spotthebot/Bot_Human/es/'
ext='.xml'
for file in file_list2: 
    #print(file)
    if (file.find('bot') != -1):  #is bot
        
        d=directory+re.split(':', file)[0]+ext
        #print('------',d)
        file_list_bot.append(d)
    else:
        if (file.find('female') != -1):
            d=directory+re.split(':', file)[0]+ext
            #print('------',d)
            file_list_human.append(d)
        else:
            d=directory+re.split(':', file)[0]+ext
            #print('------',d)
            file_list_human.append(d)
        
        
#Put the raw data in a corpus
corpus_human=[]
for file_path in file_list_human:
    with open(file_path, "r", encoding='utf8') as file:
        contents = file.read()
        soup = BeautifulSoup(contents,'xml')
        titles = soup.find_all('document')
        for title in titles:
            #print(title.get_text())  
            corpus_human.append(title.get_text())
            
corpus_bot=[]
for file_path in file_list_bot:
    with open(file_path, "r", encoding='utf8') as file:
        contents = file.read()
        soup = BeautifulSoup(contents,'xml')
        titles = soup.find_all('document')
        for title in titles:
            #print(title.get_text())  
            corpus_bot.append(title.get_text())
        


print('Number of bot twitters:',len(corpus_bot))
print('Number of human twitters:',len(corpus_human))



###############################################################################################################
print('First pre-process...')

#One corpus qith raw documents( and not empty)
corpus_bot1=[]

for doc in corpus_bot:
    if len(doc)!=0:
        merged = functools.reduce(operator.iconcat, doc, )
        merged=merged.lower()
    
        #remove tags
        merged=re.sub("</?.*?_>"," <> ",merged)
    
        # remove special characters and digits
        merged=re.sub("(\\d|\\W)+"," ",merged)
               
        
        corpus_bot1.append(merged)

corpus_human1=[]
for doc in corpus_human:
    if len(doc)!=0:
        merged = functools.reduce(operator.iconcat, doc, )
        merged=merged.lower()
    
        #remove tags
        merged=re.sub("</?.*?_>"," <> ",merged)
    
        # remove special characters and digits
        merged=re.sub("(\\d|\\W)+"," ",merged)
        #re.sub('\s+',' ',myString)
        
        corpus_human1.append(merged)


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

data_words_bot1 = list(sent_to_words(corpus_bot1))
t1 = time()
print(t1 - t0)

t0 = time()

data_words_human1 = list(sent_to_words(corpus_human1))
t1 = time()
print(t1 - t0)

print(data_words_bot1[0:10])
print(data_words_human1[0:10])



###############################################################################################################
print('Check the words in dictionary...')
#Spanish dictionary
#dic = pd.read_csv('espanol.txt',encoding='latin-1').values
with open('espanol.txt',encoding="utf8") as file:  
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
print('Bot...')

data_words_bot=[]
dic1 = set(dic)
t0 = time()
i=1
for doc in data_words_bot1:
    aux=[]
    for word in doc: 
        if word in dic1:
            aux.append(word)
    data_words_bot.append(aux)
    if(i%10==0):
        print('% processed:',(i/len(data_words_bot1))*100)
    i=i+1
t1 = time()
print(t1 - t0)

print('Human...')
# In[ ]:
data_words_human=[]
t0 = time()
i=1
for doc in data_words_human1:
    aux=[]
    for word in doc: 
        if word in dic1:
            aux.append(word)
    data_words_human.append(aux)
    if(i%10==0):
        print('% processed:',(i/len(data_words_human1))*100)
    i=i+1
t1 = time()
print(t1 - t0)




# In[ ]:


print(data_words_bot[0:20])
print(data_words_human[0:20])

# In[ ]:


np.savetxt("data_words_bot3.csv", data_words_bot, delimiter=",",fmt='%s')
np.savetxt("data_words_human3.csv", data_words_human, delimiter=",",fmt='%s')

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
data_lemmatized_human = lemmatization(data_words_human, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
t1 = time()
print(t1 - t0)

t0 = time()
         
# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized_bot = lemmatization(data_words_bot, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
t1 = time()
print(t1 - t0)

print(data_lemmatized_bot[0:10])
print(data_lemmatized_human[0:10])


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

#Load Books dictionary
sentiment_pos = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/sentiment/positive_words_es.txt', delimiter=',',dtype="|U")


vectorizer = CountVectorizer(analyzer='word',       
                             #min_df=4,                        # minimum  occurences of a word 
                             stop_words= spanish_stopwords,             # remove stop words
                             #lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

vectorizer1 = CountVectorizer(analyzer='word',       
                             #min_df=4,                        # minimum  occurences of a word 
                             stop_words= spanish_stopwords,             # remove stop words
                             #lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

vectorizer.fit_transform(sentiment_pos)
vectorizer1.fit_transform(sentiment_pos)

t0 = time()
data_vectorized_human = vectorizer.transform(data_lemmatized_human)
t1 = time()
print(t1 - t0)

t0 = time()
data_vectorized_bot = vectorizer1.transform(data_lemmatized_bot)
t1 = time()
print(t1 - t0)


print('Size text-Word matrix Human:', data_vectorized_human.shape)
print('Size text-Word matrix Bot:', data_vectorized_bot.shape)


# In[ ]:


###############################################################################################################
print('sample dictionary...')
print(vectorizer.get_feature_names()[0:10])
print(vectorizer1.get_feature_names()[0:10])


# In[31]:


# Materialize the sparse data
data_dense_human = data_vectorized_human.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity human: ", ((data_dense_human > 0).sum()/data_dense_human.size)*100, "%")

# Materialize the sparse data
data_dense_bot = data_vectorized_bot.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity bot: ", ((data_dense_bot > 0).sum()/data_dense_bot.size)*100, "%")

Ah = data_vectorized_human.toarray()
Ah = Ah.astype(float)

Ab = data_vectorized_bot.toarray()
Ab = Ab.astype(float)
# In[ ]:





# ## SVD decomposition

# In[33]:

###############################################################################################################
print('SVD decomposition...')
k=900
print('s:',k)

from scipy.sparse.linalg import svds, eigs


t0 = time()
uh, sh, vth = svds(Ah,k=900)

t1 = time()
print(t1 - t0)

t0 = time()
ub, sb, vtb = svds(Ab,k=900)

t1 = time()
print(t1 - t0)
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

np.savetxt("uh3.csv", uh, delimiter=",")
np.savetxt("ub3.csv", ub, delimiter=",")


# In[ ]:


np.savetxt("sh3.csv", sh, delimiter=",")
np.savetxt("sb3.csv", sb, delimiter=",")


# In[ ]:


np.savetxt("vth3.csv", vth, delimiter=",")
np.savetxt("vtb3.csv", vtb, delimiter=",")


# In[ ]:


np.savetxt("dicth3.csv", vectorizer.get_feature_names(), delimiter=",",fmt='%s')
np.savetxt("dictb3.csv", vectorizer1.get_feature_names(), delimiter=",",fmt='%s')


# In[ ]:


np.savetxt("Ah3.csv", Ah, delimiter=",")
np.savetxt("Ab3.csv", Ab, delimiter=",")

