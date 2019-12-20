#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from scipy.special import comb
import itertools
import pickle
import multiprocessing as mp
import scipy.stats as st
from sklearn.model_selection import ParameterGrid
import random
import sys


def assign_theta(G,theta):
    #theta_list = np.random.uniform(0,theta,nx.number_of_nodes(G))
    i = 0
    for n in G.nodes():
        #G.node[n]['th'] = theta_list[i]
        G.node[n]['th'] = theta
        i += 1

        
def cascade(grph,mu):
    if mu < 0:
        return 0
    else:
        g_size = len(grph) 
        g_nodes = list(grph.nodes())     
        seed_num = np.floor(g_size*mu).astype(int)  
        if g_size > seed_num:
            seeds = random.sample(g_nodes, seed_num)
            nodes_csd = seeds
        #    nodes_live = list(set(g_nodes) - set(nodes_csd))
            nodes_csd_t = seeds
            while nodes_csd_t:
                nodes_vnr = []
                for i in nodes_csd_t:
                    nodes_vnr += [j for j in grph.neighbors(i) if j not in nodes_csd and j not in nodes_vnr] 
                nodes_csd_t = []
                for j in nodes_vnr:  
                    nbr_csd = [x for x in grph.neighbors(j) if x in nodes_csd]
                    if len(nbr_csd) > grph.degree[j]*grph.node[j]['th']:
                        nodes_csd_t.append(j) 
                nodes_csd += nodes_csd_t
        else:
            nodes_csd = g_nodes
        return len(nodes_csd)/g_size


def rho(z_all,x_step=None):
    z_i = z_all['z']
    theta_i = z_all['theta']
    x_i  = z_all['x']
    g_i = nx.read_graphml('../results/er/z_rho/graph_z/g_{}.graphml'.format(z_i))
    assign_theta(g_i,theta_i)
    #d_rho_i = cascade(g0,x_i+x_step) - cascade(g0,x_i-x_step)
    rho_i = cascade(g_i,x_i)
    return (theta_i,z_i,x_i,rho_i)


def main():
    s = sys.argv[1]
    z_list = list(np.arange(2,16,8))
    #z_list = [4,5]
    N = 100
    g_folder = '../results/er/z_rho/graph_z/'
    p_list = [z_i/N for z_i in z_list]
    for z_i in z_list:
        p_i = z_i/N
        g_i = nx.fast_gnp_random_graph(N, p_i)
        nx.write_graphml(g_i, g_folder+'g_{}.graphml'.format(z_i))
    
    #g_z_list = (folder_name, )
    theta_list = np.arange(0.01,1,0.5)
    theta_list = theta_list.tolist()
    theta_list.reverse()
    x_list = list(np.arange(0.001,1,0.5))
    param_grid = {'z':z_list, 'theta': theta_list, 'x' : x_list}
    grid = ParameterGrid(param_grid)
    
    #x_step = 0.002
    #for s in range(100):
    p = mp.Pool(mp.cpu_count())
    results = p.map_async(rho, grid).get()
    p.close()
    p.join() 
    pickle.dump(results, open('../results/er/z_rho/z_theta_x_rho_{}.p' .format(s), 'wb'))

if __name__== "__main__":
    main()