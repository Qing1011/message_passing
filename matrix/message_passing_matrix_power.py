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
#from message_passing import find_critical_search
from mp_nx import find_critical_search


def critical_eta(eta, output):
    g_i = nx.read_graphml('../data/g_power.graphml')
    g_folder_name = 'real_data_new/power'
    x, y = find_critical_search(eta, g_i, g_folder_name)
    temp = {x:y}
    output.put(temp)


def main():
    output = mp.Queue()
    results = {}
    eta_list = [0.001,0.1,0.2,0.3,0.4,0.5]
    processes = [mp.Process(target=critical_eta, args=(eta_i, output)) for eta_i in eta_list]  
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    for p in processes:
        results.update(output.get())

    with open('../results/real_data_new/power.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

  
if __name__== "__main__":
    main()