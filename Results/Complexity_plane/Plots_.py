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

# In[2]:
print('Loading data...')

#Import Data Books
sb = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/balls1000_t/sB.csv', delimiter=',')
vtb = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/balls1000_t/vtB.csv', delimiter=',')
ub = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Books/balls1000_t/uB.csv', delimiter=',')
wb = ub*sb


print('Loaded books...')


# #Import Data Wikipedia
# sw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Wikipedia/d200000s1000/sw.csv', delimiter=',')
# vtw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Wikipedia/d200000s1000/vtw.csv', delimiter=',')
# uw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/Wikipedia/d200000s1000/uw.csv', delimiter=',')
# ww = uw*sw

# print('Loaded wikipedia...')

# #############################################################
# #Import Data Bot

sbot = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with books/new_0t/sb.csv', delimiter=',')
vtbot = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with books/new_0t/vtb.csv', delimiter=',')
ubot = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with books/new_0t/ub.csv', delimiter=',')
wbot = ubot*sbot

# # In[ ]:


# #Import Data Human

sh = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with books/new_0t/sh.csv', delimiter=',')
vth = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with books/new_0t/vth.csv', delimiter=',')
uh = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with books/new_0t/uh.csv', delimiter=',')
wh = uh*sh


# print('Loaded Comments with books dict...')
# #############################################################3
# #Import Data Bot

# sbotw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with wikipedia/sb2.csv', delimiter=',')
# vtbotw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with wikipedia/vtb2.csv', delimiter=',')
# ubotw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with wikipedia/ub2.csv', delimiter=',')
# wbotw = ubotw*sbotw

# # In[ ]:


# #Import Data Human

# shw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with wikipedia/sh2.csv', delimiter=',')
# vthw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with wikipedia/vth2.csv', delimiter=',')
# uhw = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with wikipedia/uh2.csv', delimiter=',')
# whw = uhw*shw
# print('Loaded Comments with wiki dict...')
#############################################################3
# #Import Data Bot

# sbots = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with sentiment words/sb3.csv', delimiter=',')
# vtbots = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with sentiment words/vtb3.csv', delimiter=',')
# ubots = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with sentiment words/ub3.csv', delimiter=',')
# wbots = ubots*sbots

# # In[ ]:


# #Import Data Human

# shs = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with sentiment words/sh3.csv', delimiter=',')
# vths = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with sentiment words/vth3.csv', delimiter=',')
# uhs = np.genfromtxt(fname='/home/rfreyrebonilya/Spotthebot/Resultados/comments/with sentiment words/uh3.csv', delimiter=',')
# whs = uhs*shs
# print('Loaded Comments with PositiveSent dict...')


print('Dimension Books Dict:' ,vtb.shape)
# print('Dimension Wiki Dict:' ,vtw.shape)

# print('Dimension Bot book Dict:' ,vtbot.shape)
# print('Dimension Human book Dict:' ,vth.shape)

# print('Dimension Bot wiki Dict:' ,vtbotw.shape)
# print('Dimension Human wiki Dict:' ,vthw.shape)
# print('Dimension Bot sent Dict:' ,vtbots.shape)
# print('Dimension Human sent Dict:' ,vths.shape)


#############################################################
def reconst(lib,w,vt,k1,k):
    reconstruct = np.zeros(vt[0,:].shape) 
    r=(np.array(range(k1,k))+1)*(-1)
    for l in r:
        reconstruct += vt[l,:]*w[lib,l]
    return reconstruct
        

# def plot_series(lib,w,vt,t): #t type of dict
#     columns = 3
#     rows = 4
#     tipe_dict={1:'book',
#           2:'wiki',
#           3:'bot_b',
#           4:'human_b',
#           5:'bot_w',
#           6:'human_w',
#           7:'bot_s',
#           8:'human_s'
#               }

#     tittle=tipe_dict[t]+str(lib)

#     fig, ax_array = plt.subplots(rows, columns,figsize=(25,20),squeeze=False)

#     index=0
#     #lib=1624

#     modes=[1,2,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#     for i,ax_row in enumerate(ax_array):
    
#         for j,axes in enumerate(ax_row):
        
        
#             #print(modes[index])
#             reconstruction=reconst(lib,w,vt,0,modes[index])
            
#             axes.plot(list(range(0,vt.shape[1])),reconstruction,color=".3",linewidth=1.5)
#             axes.set_title('SV used:{}'.format(modes[index]))
#             axes.set_yticklabels([])
#             axes.set_xticklabels([])
#             index=index+1
#     plt.savefig(os.path.join(os.getcwd(), tittle+'.pdf'))
# #plt.show()


# sample=list(range(0,10))

# print('Plotting samples...')
# for i in sample:
    
#     plot_series(sample[i],wb,vtb,1)
    
#     plot_series(sample[i],ww,vtw,2)
    
#     plot_series(sample[i],wbot,vtbot,3)
#     plot_series(sample[i],wh,vth,4)
    
#     plot_series(sample[i],wbotw,vtbotw,5)
#     plot_series(sample[i],whw,vthw,6)
    
#     plot_series(sample[i],wbots,vtbots,7)
#     plot_series(sample[i],whs,vths,8)

#############################################################
def phase_space(lib,w,vt,first_eig,last_eig, emb_dim,lag,order,lag1):
   
    
    #reconstruct the book/wiki/comment
    reconstruction=reconst(lib,w,vt,first_eig,last_eig)
    
    if np.sum(reconstruction)==0:
        probability=99999999999
        MPR=99999999999
        NSE=99999999999
        
    else:
    
        probability=complexity.ordinal_pat_prob(reconstruction, order, lag1)
        NSE=complexity.shannon_ent(probability,normalize=True)
        MPR=complexity.MPR_entropy(reconstruction, order, lag1)
    
    return probability,NSE,MPR

##########################################################


def generate_phase_info(w,vt,emb_dim,lag,order,lag1):
    size=100#w.shape[0]

    probability_=[]
    NSE_=[]
    MPR_=[]
    
    for i in range(size):
        probability,NSE,MPR=phase_space(i,w,vt,0,1000,emb_dim,lag,order,lag1)
        probability_.append(probability)
        NSE_.append(NSE)
        MPR_.append(MPR)
        
    return probability_,NSE_,MPR_
        
print('Processin Phase space...')

#emb_dim=20
#lag=1
#print('emb_dim:',emb_dim)
#print('lag:',lag)


#t0 = time()
#NSE_books,MPR_books=generate_phase_info(wb,vtb,emb_dim,lag,order,lag1)
#t1 = time()
#print('Time process books:',t1 - t0)

#generate_phase_info(ww,vtw,Lyapunov_wiki,MI_wiki,PhaseSpace_wiki)

# t0 = time()
# Lyapunov_bot_book,corr_bot_book,PhaseSpace_bot_book,probability_bot_book,NSE_bot_book,MPR_bot_book=generate_phase_info(wbot,vtbot,emb_dim,lag)

# t1 = time()
# print('Time process bot_books:',t1 - t0)

# t0 = time()
# Lyapunov_human_book,corr_human_book,PhaseSpace_human_book,probability_human_book,NSE_human_book,MPR_human_book=generate_phase_info(wh,vth,emb_dim,lag)
# t1 = time()
# print('Time process human_books:',t1 - t0)
#############################################################

print('Saving results...')  

#np.savetxt("lyapunov_books.csv", Lyapunov_books, delimiter=",")
# np.savetxt("Lyapunov_bot_book.csv", Lyapunov_bot_book, delimiter=",")
# np.savetxt("Lyapunov_human_book.csv", Lyapunov_human_book, delimiter=",")

#np.savetxt("corr_books.csv", corr_books, delimiter=",")
# np.savetxt("corr_bot_book.csv", corr_bot_book, delimiter=",")
# np.savetxt("corr_human_book.csv", corr_human_book, delimiter=",")

#np.savetxt("NSE_books.csv", NSE_books, delimiter=",")
# np.savetxt("NSE_bot_book.csv", NSE_bot_book, delimiter=",")
# np.savetxt("NSE_human_book.csv", NSE_human_book, delimiter=",")

#np.savetxt("MPR_books.csv", MPR_books, delimiter=",")
# np.savetxt("MPR_bot_book.csv", MPR_bot_book, delimiter=",")
# np.savetxt("MPR_human_book.csv", MPR_human_book, delimiter=",")


#dumped = json.dumps(PhaseSpace_books, cls=NumpyEncoder)
#with open('PhaseSpace_books.txt','w') as myfile:
#    json.dump(dumped,myfile)

#dumped1 = json.dumps(PhaseSpace_bot_book, cls=NumpyEncoder)
# with open('PhaseSpace_bot_book.txt','w') as myfile:
#     json.dump(dumped1,myfile)

#dumped2 = json.dumps(PhaseSpace_human_book, cls=NumpyEncoder)
# with open('PhaseSpace_human_book.txt','w') as myfile:
#     json.dump(dumped2,myfile)

#############################################################    
def plot_ps(PhaseSpace_,t): #t type of dict
    columns = 5
    rows = 10
    tipe_dict={1:'book',
          2:'wiki',
          3:'bot_b',
          4:'human_b',
          5:'bot_w',
          6:'human_w',
          7:'bot_s',
          8:'human_s'
              }

    tittle=tipe_dict[t]

    fig, ax_array = plt.subplots(rows, columns,figsize=(25,50),squeeze=False)

    index=0
    #lib=1624

    PhaseSpace_=np.asarray(PhaseSpace_)
    for i,ax_row in enumerate(ax_array):
    
        for j,axes in enumerate(ax_row):
        
            
            axes.plot(np.array(PhaseSpace_[index]).T[0],np.array(PhaseSpace_[index]).T[1])
            axes.set_title('Phase space sample:{}'.format(index))
            #axes.set_yticklabels([])
            #axes.set_xticklabels([])
            index=index+1
    plt.savefig(os.path.join(os.getcwd(), tittle+'.pdf'))
    plt.clf()


def plot_hist(lyap_,t,et):

    tipe_dict={1:'book',
          2:'wiki',
          3:'bot_b',
          4:'human_b',
          5:'bot_w',
          6:'human_w',
          7:'bot_s',
          8:'human_s'
              }

    est_dic={1:'lyapunov',
          2:'correlation'
              }

    tittle=tipe_dict[t]+'_'+est_dic[et]

    lyap_=np.array(lyap_)
    lyap_ = lyap_.astype('float')
    lyap_[lyap_ == 99999999999] = 'nan'
    lyap_ = lyap_[~np.isnan(lyap_)]
    plt.hist(lyap_, bins='auto') 
    plt.title(r'Histogram')
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(), tittle+'.pdf'))
    plt.clf()
    
def import_t(x):
    x=np.asarray(x)
    x = x.astype('float')
    x[x == 99999999999] = 'nan'
    x = x[~np.isnan(x)] 
    return x



###################################################################################################
def D_max(p,order=3,dim=1):
    x = np.array(p)
    dim=x.ndim
    N=factorial(order)**dim
    
    
    
    q_0=(-1.0/(((N+1)/N)*np.log(N+1)-2*np.log(2*N)+np.log(N)))*2
    
    #P=ordinal_pat_prob(x, order, delay)
    P_e=complexity.equi_prob(order,dim)
    #H_P=complexity.shannon_ent(P,normalize=True)
    
    q_j=(complexity.shannon_ent(np.true_divide((x+P_e),2),normalize=False) - (complexity.shannon_ent(x,normalize=False))/2
        -(complexity.shannon_ent(P_e,normalize=False))/2)*q_0
    
    return q_j

def D_min(p,n,order=3,dim=1):
    N=factorial(order)**dim
    pe=1/N
    #pe=complexity.equi_prob(order,dim) 
    Dmin=np.power(p-pe, 2)+(N-n-1)*np.power(((1-p)/(N-n-1))-pe, 2)+n*np.power(pe, 2)
    #Dmax=np.power(p-pe, 2).sum()
    
    return Dmin

def D_minenv(h,order=3,dim=1):
    N=factorial(order)**dim
    #pe=1/N
    #pe=equi_prob(order,dim) 
    Dminenv=np.exp(-h*np.log(N))-(1/N)
    #Dminenv=np.power(2,-h*np.log2(N))-(1/N)
    #Dmax=np.power(p-pe, 2).sum()
    
    return Dminenv
    
    
def max_min_complexity_entropy(order=3,dim=1,n=0):
    N=factorial(order)**dim
    points=1000
    f_max=np.linspace(1/N,0.99,points)
    f_min=np.linspace(0.01,1/(N-n),points)
    #f=(1-fmax)/(N-1)
    H=[]
    Hmin=[]
    Dmax=[]
    Dminenv=[]
    Dmin=[]
    for fmax in f_max:
        f=np.concatenate((fmax, np.multiply(np.ones(N-1),((1-fmax)/(N-1)))), axis=None)
        #h=-(1/np.log(N))*((fmax*np.log(fmax))+(1-fmax)*np.log((1-fmax)/(N-1)))
        h=complexity.shannon_ent(f,normalize=True)
        H.append(h)
        #dmax=D_max(fmax,order,dim)
        dmax=D_max(f,order,dim)
        Dmax.append(dmax)
        dminenv=D_minenv(h,order,dim)
        Dminenv.append(dminenv)
        
    for fmin in f_min:
        #f=np.concatenate((fmax, np.ones(N-1)*((1-fmax)/(N-1))), axis=None)
        hmin=-(1/np.log(N))*((fmin*np.log(fmin))+(1-fmin)*np.log((1-fmin)/(N-n-1)))
        #h=shannon_ent(f,normalize=True)
        Hmin.append(hmin)
        dmin=D_min(fmin,n,order,dim)
        Dmin.append(dmin)
        
    
    H=np.asarray(H)
    Hmin=np.asarray(Hmin)
    
    Dmax=np.asarray(Dmax)
    Dminenv=np.asarray(Dminenv)
    Dmin=np.asarray(Dmin)
    
    Cmax=H*Dmax
    Cminenv=H*Dminenv
    Cmin=Hmin*Dmin
    
    return Cmax,Cmin,Cminenv,H,Hmin





###################################################################






def plot_complex_entrop(MPR_books,MPR_bot,MPR_human,NSE_books,NSE_bot,NSE_human,o,l):
#def plot_complex_entrop(MPR_books,NSE_books,o,l):

    
    MPR_books=import_t(MPR_books)
    MPR_bot=import_t(MPR_bot)
    MPR_human=import_t(MPR_human)
    NSE_books=import_t(NSE_books)
    NSE_bot=import_t(NSE_bot)
    NSE_human=import_t(NSE_human)

    Cmax,Cmin,Cminenv,H,Hmin=max_min_complexity_entropy(order=o,dim=1,n=0)

    tittle='Complexity_plane'+'_'+'order'+ str(o) + '_' + 'lag'+ str(l)

    plt.figure(figsize=(15, 10))

    #plt.plot(H,Cmax,'--k',markersize=0.2,label='C_max')
    #plt.plot(H,Cminenv,'--k',markersize=0.2,label='C_min')

    plt.plot(NSE_books,MPR_books,'ro',markersize=0.5,label='Books:'+'(min_MPR,max_MPR)=' + '('+ str("{0:.4f}".format(np.min(MPR_books))) + ','+str("{0:.4f}".format(np.max(MPR_books)))+')'
             +'/'+'(min_NSE,max_NSE)=' + '('+ str("{0:.4f}".format(np.min(NSE_books))) + ','+str("{0:.4f}".format(np.max(NSE_books)))+')')
    
    plt.plot(NSE_bot,MPR_bot,'b+',markersize=0.5,label='Bots:'+'(min_MPR,max_MPR)=' + '('+ str("{0:.4f}".format(np.min(MPR_bot))) + ','+str("{0:.4f}".format(np.max(MPR_bot)))+')'
             +'/'+'(min_NSE,max_NSE)=' + '('+ str("{0:.4f}".format(np.min(NSE_bot))) + ','+str("{0:.4f}".format(np.max(NSE_bot)))+')')
    
    plt.plot(NSE_human,MPR_human,'g^',markersize=0.5,label='Human:'+'(min_MPR,max_MPR)=' + '('+ str("{0:.4f}".format(np.min(MPR_human))) + ','+str("{0:.4f}".format(np.max(MPR_human)))+')'
             +'/'+'(min_NSE,max_NSE)=' + '('+ str("{0:.4f}".format(np.min(NSE_human))) + ','+str("{0:.4f}".format(np.max(NSE_human)))+')')
    plt.legend()
    plt.xlabel('Normalized Shannon Entropy')
    plt.ylabel('MPR - SC')
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(), tittle+'.pdf'))
    plt.clf()
    
    
print('Plotting...')    
#############################################################   

#plot_complex_entrop(MPR_books,MPR_bot_book,MPR_human_book,NSE_books,NSE_bot_book,NSE_human_book)
#plot_complex_entrop(MPR_books,NSE_books)
  

#plot_hist(Lyapunov_books,1,1)
#plot_hist(Lyapunov_bot_book,3,1)
#plot_hist(Lyapunov_human_book,4,1)

#plot_hist(corr_books,1,2)
#plot_hist(corr_bot_book,3,2)
#plot_hist(corr_human_book,4,2)


#plot_ps(PhaseSpace_books,1)
# plot_ps(PhaseSpace_bot_book,3)
# plot_ps(PhaseSpace_human_book,4)

#################################################################

emb_dim=10
lag=1
orders=[3,4,5,6,7,8,9,10]
lags=[1,2,3,4,5]

for o in orders:
        for l in lags:
                probability_books,NSE_books,MPR_books=generate_phase_info(wb,vtb,emb_dim,lag,o,l)
                probability_bot_book,NSE_bot,MPR_bot=generate_phase_info(wbot,vtbot,emb_dim,lag,o,l)
                probability_human_book,NSE_human,MPR_human=generate_phase_info(wh,vth,emb_dim,lag,o,l)

                #plot_complex_entrop(MPR_books,NSE_books,o,l)
                plot_complex_entrop(MPR_books,MPR_bot,MPR_human,NSE_books,NSE_bot,NSE_human,o,l)
                




