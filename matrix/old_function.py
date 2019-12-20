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


def find_critical(eta, g, g_name, theta_list):
    p = Pool(10)
    result_1 = dict(p.map(lambda x: theta_eigen(g,x,eta), theta_list))
    result_i = result_1
    diff_to_1 = dict((k,v['eigen_diff']) for k, v in result_1.items())
    
    k1, k2 = nsmallest(2, diff_to_1, key=diff_to_1.get)
    theta_list_small = small_step_list(k1,k2,0.01)
    
    result_2 = dict(p.map(lambda x: theta_eigen(g,x,eta), theta_list_small))
    result_i.update(result_2)
    diff_to_2 = dict((k, v['eigen_diff']) for k, v in result_2.items())
    
    theta_c = min(diff_to_2, key = diff_to_2.get)

    with open('../results/'+'theta_result_'+'%s_%s.pickle' %(g_name, eta), 'wb') as handle:
        pickle.dump(result_i, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return eta, theta_c


def DH(G, eta):
    edges = list(G.edges())
    edges_re = []
    for e in edges:
        edges_re.append((e[1],e[0]))
    
    edges.extend(edges_re)
    #h = list(itertools.product(edges, repeat = 2))
    h_dict = {h_i: 0 for h_i in itertools.product(edges, repeat = 2)}
    
    for e in edges:
        entries_i = propogation_simple2(G,e[1],eta) 
        neighbours = list(set(G.edges(e[1]))-set([e,(e[1],e[0])]))
        temp = zip([e]*len(neighbours),neighbours)
        
        for temp_i in temp:
            h_dict[temp_i] = entries_i

    #entries = dict(zip(keys, map(lambda x: propogation_simple(G,x,rho), keys)))

    #print (h_dict)
    n_edge = nx.number_of_edges(G)
    n_col = 2*n_edge
    
    row = []
    for i in range(n_col):
        row.extend(n_col*[i])
    col = n_col*list(range(n_col))
    
    dh_matrix = csr_matrix((list(h_dict.values()), (row, col)), shape=(n_col, n_col)).toarray()
    
    return dh_matrix


def hm_i(e, g, eta):
    h_dict_temp ={}
    entries_i = propogation_simple2(g,e[1],eta)
    neighbours = list(set(g.edges(e[1]))-set([e,(e[1],e[0])]))
    temp = zip([e]*len(neighbours),neighbours)
    for temp_i in temp:
        h_dict_temp[temp_i] = entries_i
    return h_dict_temp

def DH(G, eta):
    edges = list(G.edges())
    edges_re = []
    for e in edges:
        edges_re.append((e[1],e[0]))
    
    edges.extend(edges_re)
    
    #keys = {k for k in itertools.product(edges, repeat = 2)}
    #h_dict = dict.fromkeys(keys, 0)
    
    h_dict = {h_i: 0 for h_i in itertools.product(edges, repeat = 2)}
    #diagonal = []
#     for e in edges:
#         entries_i = propogation_simple2(G,e[1],eta) 
#         neighbours = list(set(G.edges(e[1]))-set([e,(e[1],e[0])]))
#         temp = zip([e]*len(neighbours),neighbours)
        
#         for temp_i in temp:
#             h_dict[temp_i] = entries_i
    p = mp.Pool(processes=6)
    results = [p.apply_async(hm_i, args = (x, G, eta)) for x in edges]
    for p in results:
        h_dict.update(p.get())
        
    
    n_edge = nx.number_of_edges(G)
    n_col = 2*n_edge
    
    row = []
    for i in range(n_col):
        row.extend(n_col*[i])
    col = n_col*list(range(n_col))
    
    #B_matrix = csr_matrix((list(h_dict.values()), (row, col)), shape=(n_col, n_col))
    #D = lil_matrix(np.zeros([n_col, n_col]), dtype = int)
    #D.setdiag(diagonal)
    dh_matrix = csr_matrix((list(h_dict.values()), (row, col)), shape=(n_col, n_col)).toarray()
    
    return dh_matrix


def create_dic(edges_g):
    dict_e = {h_i: 0 for h_i in itertools.product(edges_g, repeat = 2)}
    return dict_e

def propogation_simple2(g_p, j, q):
    theta_j = nx.get_node_attributes(g_p,'activated')[j]
    #if the theta_j is from uniform distribution 
    k_j = g_p.degree[j]
    #m_j = k_j*theta_j/2
    m_j = k_j*theta_j
    dh_j = 0

    if m_j  >= 1 and k_j > 2:
        factor = comb(k_j-2, m_j-1, exact=True)
        dh_j = q**(m_j-1)*(1-q)**(k_j - m_j - 1)*factor
        #dh_j = (1/k_j) * (1/theta_j) #（for uniform theta and eta = 0 ）
    return dh_j