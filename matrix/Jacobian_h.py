# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 19:02:07 2019

@author: NW

nonbacktracking matrix NB from networkx graph G or adjacency matrix
diagnol matrix with weights D,  initial seed size \eta, fixed threshold \theta
Jacobian_h = np.multiply(D*NB, 1-eta)

"""

#import random
#import math
import numpy as np
#import matplotlib.pyplot as plt
#import time
import networkx as nx
from scipy.sparse import csr_matrix,lil_matrix


N=100
p=0.05
G=nx.fast_gnp_random_graph(N,p)     


w = lambda x,d,m: x+d+m         # to be changed to the correct weight


# graph G, initial seed /eta, fixed fractional threshold /theta
def Jacobian_h(G,eta,theta):
    E=np.array(G.edges())
    n=len(E)
    NB = np.zeros((2*n,2*n))
    v_diag = np.zeros(2*n)
    for idx in range(n):
        e = E[idx] # e=(i,j)  i<j , idx = i<-j , idx+n = j<-i
        i = e[0]
        j = e[1]
        d_j = G.degree[j]
        if d_j >= 2:
            m_j = np.floor(d_j*theta)
            v_diag[idx] = w(eta,d_j,m_j)
        d_i = G.degree[j]
        if d_j >= 2:
            m_i = np.floor(d_i*theta)       
            v_diag[idx+n] = w(eta,d_i,m_i)
        # find indices with e[0] = j,   
        x=[]
        y=[]
        for idx_k in range(n): 
            if E[idx_k][0]==j:
                x.append(idx_k)
            if E[idx_k][1]==j and E[idx_k][0]!=i:
                y.append(idx_k)
        for idx_x in x:
            NB[idx][idx_x]=1
            NB[idx+n][idx_x+n]=1
        for idx_y in y:
            NB[idx][idx_y+n]=1
            NB[idx+n][idx_y]=1   
    D=np.diag(v_diag)
    return np.multiply(D*NB, 1-eta)

# =============================================================================
# G_inc_matrix = nx.incidence_matrix(G, oriented=True).toarray()
# get_idx = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
# for idx in range(n):
#     j=E[idx][1]
#     v_j=G_inc_matrix[j,:]    
#     x_j = get_idx(1,v_j).remove(idx)
#     y_j = get_idx(-1,v_j)
#     for k in x_j:
#         NB[idx][k]=1
#     for k in y_j:
#         NB[idx+n][k]=1
# 
# A=NB[:n,:n]
# B=NB[n:,:n]
# 
# NB=np.block([A,B],[B,A])
# =============================================================================
            
            
            
    