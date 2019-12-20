#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
from scipy.special import comb
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import pickle
import copy
import multiprocess as mp
from heapq import nsmallest
from numba import jit



def assign_theta(G,theta):
    for n in G.nodes():
        #G.node[n]['activated'] = np.random.uniform(0,theta)
        G.node[n]['activated'] = theta
    return G


def w(x,d,m):
    factor = comb(d-2, m-1, exact=True)
    dh_j = x**(m-1)*(1-x)**(d - m - 1)*factor
    return dh_j



def theta_eigen(g_i,theta_i,rho_0):
    #g_i = copy.deepcopy(g0)
    #g_i = assign_theta(g_i,theta_i)
    dh_m_i = Jacobian_h(g_i, rho_0, theta_i)
    try:
        vals_i, vecs_i = eigs(dh_m_i, k=6)
    except:
        vals_i = [0]
    eigen_i = {}
    vals_abs = [abs(i) for i in vals_i]
    eigen_i['eigen_abs'] = max(vals_abs)*(1-rho_0)
    eigen_i['eigen_diff'] = abs(max(vals_abs)*(1-rho_0)-1)
    return theta_i, eigen_i


def small_step_list(k1,k2,step):
    if k1>k2:
        theta_list_2 = np.arange(k2,k1,step)
    else :
        theta_list_2 = np.arange(k1,k2,step)
    return theta_list_2


def narrow_window(theta_eigen_dict, upper, step_i):
    diff_to_1 = dict((k,v['eigen_diff']) for k, v in theta_eigen_dict.items())
    if min(diff_to_1, key = diff_to_1.get) == upper:
        k1_i = upper
        k2_i = 1
    else:
        k1_i, k2_i = nsmallest(2, diff_to_1, key=diff_to_1.get)
    theta_list_s = small_step_list(k1_i,k2_i,step_i)
    
    return theta_list_s


def find_critical_search(eta, g, g_name):
    theta_list0 = list(np.arange(0.01,1,0.01))
    p = mp.Pool(5)
    result_0 = dict(p.map(lambda x: theta_eigen(g,x,eta), theta_list0))
    result_i = result_0

    #theta_list_s1 = narrow_window(result_0, 0.91,0.01)
    #result_1 = dict(p.map(lambda x: theta_eigen(g,x,eta), theta_list_s1))
    #result_i.update(result_1)
    
    diff_to_1 = dict((k, v['eigen_diff']) for k, v in result_0.items())
    theta_c = min(diff_to_1, key = diff_to_1.get)

    with open('../results/'+'theta_result_'+'%s_%s.pickle' %(g_name, eta), 'wb') as handle:
        pickle.dump(result_i, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return eta, theta_c


# graph G, initial seed /eta, fixed fractional threshold /theta
def Jacobian_h(G,eta,theta):
    E = np.array(G.edges())
    n = len(E)
    NB = np.zeros((2*n,2*n))
    v_diag = np.zeros(2*n)
    for idx in range(n):
        e = E[idx] # e=(i,j)  i<j , idx = i<-j , idx+n = j<-i
        i = e[0]
        j = e[1]
        d_j = G.degree[j]
        if d_j >= 2:
            m_j = np.floor(d_j*theta)
            #v_diag[idx] = w(eta,d_j,m_j)
            v_diag[idx] = 1/(theta*d_j)
        d_i = G.degree[j]
        if d_j >= 2:
            m_i = np.floor(d_i*theta)       
            #v_diag[idx+n] = w(eta,d_i,m_i)
            v_diag[idx+n] = 1/(theta*d_i)
        # find indices with e[0] = j,   
        x = []
        y = []
        for idx_k in range(n): 
            if E[idx_k][0] == j:
                x.append(idx_k)
            if E[idx_k][1] == j and E[idx_k][0]!=i:
                y.append(idx_k)
        for idx_x in x:
            NB[idx][idx_x] = 1
            NB[idx+n][idx_x+n] = 1
        for idx_y in y:
            NB[idx][idx_y+n] = 1
            NB[idx+n][idx_y] = 1   
    D = np.diag(v_diag)
    H = np.matmul(D,NB)
    return np.multiply(H, 1-eta)