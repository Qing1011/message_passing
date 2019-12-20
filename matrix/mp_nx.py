#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
from scipy.special import comb
import itertools
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs
import pickle
import multiprocessing as mp
from heapq import nsmallest
from scipy.sparse import csr_matrix,lil_matrix
from decimal import Decimal

# def weight(x,d,m):
#     return comb(d-2, m-1, exact=True)*x**(m-1)*(1-x)**(d-m-1)

def weight(x,d,m):
    a = Decimal(comb(d-2, m-1, exact=True))
    b = Decimal(x**(m-1))
    c = Decimal((1-x)**(d-m-1))
    return float(a*b*c)


def Jacobian_h(G,eta,theta):
    E = np.array(list(G.edges()))
    n = len(E)
    NB = lil_matrix((2*n,2*n))
    v_diag = np.zeros(2*n)
    prevs = dict()
    succs = dict()
    for idx in range(n):
        e = E[idx]
        i, j = e
        succs.setdefault(j, []).append(idx)
        prevs.setdefault(i, []).append(idx)

    for idx in range(n):
        e = E[idx] # e=(i,j)  i<j , idx = i<-j , idx+n = j<-i
        i = e[0]
        j = e[1]
        d_j = G.degree[j]
        if d_j >= 2:
            m_j = np.floor(d_j*theta)
            v_diag[idx] = weight(eta,d_j,m_j)
        d_i = G.degree[i]
        if d_i >= 2:
            m_i = np.floor(d_i*theta)       
            v_diag[idx+n] = weight(eta,d_i,m_i)

        list1 = prevs.get(j, [])
        list2 = list(set(succs.get(j, [])) - set(prevs.get(i, [])))
        list3 = list(set(prevs.get(i, [])) - set(succs.get(j, [])))
        # update entries from idx lists to obtain NB
        for i1 in list1:
            NB[idx,i1] = 1
            NB[i1+n,idx+n] = 1
        for i2 in list2:
            NB[idx,i2+n] = 1
        for i3 in list3:
            NB[idx+n,i3] = 1

    D = lil_matrix((2*n,2*n))
    D.setdiag(v_diag)
    return np.multiply(np.multiply(D,NB), 1-eta)


# def Jacobian_h(G,eta,theta):
#     E = np.array(G.edges())
#     n = len(E)
#     NB = lil_matrix((2*n,2*n))
#     v_diag = np.zeros(2*n)
#     for idx in range(n):
#         e = E[idx] # e=(i,j)  i<j , idx = i<-j , idx+n = j<-i
#         i = e[0]
#         j = e[1]
#         d_j = G.degree[j]
#         if d_j >= 2:
#             m_j = np.floor(d_j*theta)
#             v_diag[idx] = weight(eta,d_j,m_j)
#         d_i = G.degree[i]
#         if d_i >= 2:
#             m_i = np.floor(d_i*theta)       
#             v_diag[idx+n] = weight(eta,d_i,m_i)
#         list1 = []    # idx with i<j<k
#         list2 = []    # idx with i<j>k
#         list3 = []    # idx with j>i<k
#         for idx_k in range(n): 
#             if E[idx_k][0] == j:
#                 list1.append(idx_k)
#             if E[idx_k][1] == j and E[idx_k][0] != i:
#                 list2.append(idx_k)
#             if E[idx_k][0] == i and E[idx_k][1] != j:
#                 list3.append(idx_k)                
#         # update entries from idx lists to obtain NB
#         for i1 in list1:
#             NB[idx,i1] = 1
#             NB[i1+n,idx+n] = 1
#         for i2 in list2:
#             NB[idx,i2+n] = 1
#         for i3 in list3:
#             NB[idx+n,i3] = 1

#     D = lil_matrix((2*n,2*n))
#     D.setdiag(v_diag)
#     return np.multiply(np.multiply(D,NB), 1-eta)


# def theta_eigen(g_i,rho_0,theta_i):
#     #db = HM(g_i, rho_0, theta_i)
#     #dh_m_i = np.multiply(db,1-rho_0)
#     dh_m_i = Jacobian_h(g_i,rho_0,theta_i)
#     try:
#         vals_i, vecs_i = eigs(dh_m_i, k=6)
#     except:
#         vals_i = [0]
#     eigen_i = {}
#     vals_abs = [abs(i) for i in vals_i]
#     eigen_i['eigen_abs'] = max(vals_abs)
#     #eigen_i['eigen_diff'] = abs(max(vals_abs)-1)
#     return {theta_i : eigen_i}


def theta_eigen(g_i,rho_0,theta_i):
    #db = HM(g_i, rho_0, theta_i)
    #dh_m_i = np.multiply(db,1-rho_0)
    dh_m_i = Jacobian_h(g_i,rho_0,theta_i)
    try:
        vals_i, vecs_i = eigs(dh_m_i, k=6)
    except:
        vals_i = [0]
    eigen_i = []
    vals_abs = [abs(i) for i in vals_i]
    eigen_i = max(vals_abs)
    #eigen_i['eigen_diff'] = abs(max(vals_abs)-1)
    return {theta_i : eigen_i}


def find_critical_search(eta, g, g_name):
    #theta_list0 = list(np.arange(0.01,1,0.01))
    theta_list0 = [i/20 for i in range(20)]
    p = mp.Pool(mp.cpu_count())
    #(worker, (i, 4), callback=callback)
    result_0 = {}
    
    for x in theta_list0:
        p.apply_async(theta_eigen, (g,eta,x), callback = result_0.update) 
    p.close()
    p.join() 
    
    #diff_to_1 = dict((k, v['eigen_diff']) for k, v in result_0.items())
    #theta_c = min(diff_to_1, key = diff_to_1.get)

    with open('../results/'+'%s_%s.pickle' %(g_name, eta), 'wb') as handle:
        pickle.dump(result_0, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return eta, result_0


