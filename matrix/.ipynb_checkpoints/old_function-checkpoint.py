#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import pickle
import copy


def assign_theta(G,theta):
    theta_list = np.random.uniform(0,theta,nx.number_of_nodes(G))
    i = 0
    for n in G.nodes():
        G.node[n]['defect'] = theta_list[i]
        i += 1


def propogation(g_p, h_i_p, rho_p):
    i,j,k,l = h_i_p[0][0], h_i_p[0][1], h_i_p[1][0], h_i_p[1][1]
    dh_j = 0
    if j==k and i!=l:
        #print (i,j,k,l)
        #print ('calculate')
        theta_j = nx.get_node_attributes(g_p,'defect')[j]
        k_j = g_p.degree[j]
        m_j = k_j*theta_j/2  
        #m_j = k_j*theta_j

        if m_j  >= 1 and k_j > 2:
            factor = comb(k_j-2, m_j-1, exact=True)
            dh_j = rho_p**(m_j-1)*(1-rho_p)**(k_j - m_j - 1)*factor
            #print (theta_j, m_j, factor, dh_j)
        else :
            #dh_j = 1
            pass 
            #print (dh_j)
        #print ('__________')
        
    return dh_j


def DH(G, rho):
    edges = list(G.edges())
    edges_re = []
    for e in edges:
        edges_re.append((e[1],e[0]))
    
    edges.extend(edges_re)
    h = list(itertools.product(edges, repeat = 2))
    #dh_dict = {h_i: propogation(g_test,h_i,rho) for h_i in h}
    #dh_dict = dict(zip(h, map(lambda x: propogation(g_test,x,0.0001), h)))
    
    #print ('finish propogation')
    
    n_edge = nx.number_of_edges(G)
    n_col = 2*n_edge
    
    row = []
    for i in range(n_col):
        row.extend(n_col*[i])
    col = n_col*list(range(n_col))
    
    #entries = list(dh_dict.values())
    entries = list(map(lambda x: propogation(G,x,rho), h))
    dh_matrix = csr_matrix((entries, (row, col)), shape=(n_col, n_col)).toarray()
    
    return dh_matrix


def theta_eigen(g_i,theta_i,rho_0):
    #g_i = copy.deepcopy(g0)
    #assign_theta(g_i,theta_i)
    dh_m_i = DH(g_i, rho_0)
    #print (rho_0, theta_i)
    vals_i, vecs_i = eigs(dh_m_i, k=6)
    eigen_max = max(vals_i)
    return eigen_max