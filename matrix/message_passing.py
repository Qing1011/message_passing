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


def NBM(g,edges):
    l_edges = []
    for e in edges:
        neighbours = list(set(g.edges(e[1]))-set((e,(e[1],e[0]))))
        temp = zip([e]*len(neighbours),neighbours)
        l_edges.extend(list(temp))
    g_l = nx.DiGraph()
    g_l.add_edges_from(l_edges)
    A = nx.adjacency_matrix(g_l)
    l_nodes = list(g_l.nodes())
    return A, l_nodes

def DM(g,eta,theta,edges):
    n = len(edges)
    v_diag = np.zeros(n)
    for idx in range(n):
        e = edges[idx]
        j = e[1]
        d_j = g.degree[j]
        if d_j >= 2:
            m_j = np.floor(d_j*theta)
            v_diag[idx] = w(eta,d_j,m_j)
            #v_diag[idx] = 1/(theta*d_j)
    D = lil_matrix((n,n))
    D.setdiag(v_diag)
    return D


def w(x,d,m):
    factor = comb(d-2, m-1, exact=True)
    dh_j = x**(m-1)*(1-x)**(d - m - 1)*factor
    return dh_j


def HM(g0,eta0,theta0):
    edges = list(g0.edges())
    edges_re = [(e[1],e[0]) for e in edges]
    edges.extend(edges_re)
    b,l_nodes = NBM(g0,edges)
    d = DM(g0,eta0,theta0,l_nodes)
    H = np.multiply(d,b)
    #H = np.multiply(db,1-eta0)
    return H


def theta_eigen(g_i,rho_0,theta_i):
    db = HM(g_i, rho_0, theta_i)
    dh_m_i = np.multiply(db,1-rho_0)
    try:
        vals_i, vecs_i = eigs(dh_m_i, k=6)
    except:
        vals_i = [0]
    eigen_i = {}
    vals_abs = [abs(i) for i in vals_i]
    eigen_i['eigen_abs'] = max(vals_abs)
    #eigen_i['eigen_diff'] = abs(max(vals_abs)-1)
    return {theta_i : eigen_i}


def find_critical_search(eta, g, g_name):
    theta_list0 = list(np.arange(0.01,1,0.01))
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




