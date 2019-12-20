#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import copy
from heapq import nsmallest
import random
import multiprocessing as mp
from multiprocess import Pool
import string
from mp_nx import find_critical_search, theta_eigen
#from message_passing import find_critical_search

# def critical_multi_i(eta_i, output):
#     g_i = nx.read_graphml('../data/g_gowalla.graphml')
#     g_name_i = 'real_data/gowalla'
#     x, y = find_critical_search(eta_i, g_i,g_name_i)
#     temp = {x:y}
#     output.put(temp)


# def main():
#     eta_list = [0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]
#     for eta_i in eta_list:
#         g_i = nx.read_graphml('../data/g_facebook.graphml')
#         g_name_i = 'real_data/facebook'
#         x, y = find_critical_search(eta_i, g_i,g_name_i)

  
# if __name__== "__main__":
#     main()


def main():
    eta_list = [0.001,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    for eta_i in eta_list:
        g_i = nx.read_graphml('../data/g_config_fb.graphml')
        g_name_i = 'real_data_2/config_fb'
        theta_list0 = [i/20 for i in range(20)]
        result = {}
        for x in theta_list0:
            result_i = theta_eigen(g_i,eta_i,x)
            result.update(result_i)
        
        with open('../results/'+'%s_%s.pickle' %(g_name_i, eta_i), 'wb') as handle:
            pickle.dump(result, handle,protocol=pickle.HIGHEST_PROTOCOL)
 
if __name__== "__main__":
    main()
