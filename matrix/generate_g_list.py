#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# generating er graph with z in 1,16 30 steps
import numpy as np
import networkx as nx
import multiprocessing as mp

def gen_g_i(z):
    n = int(1E2)
    p_er = z/n
    g_er = nx.fast_gnp_random_graph(n,p_er)
    nx.write_graphml(g_er, '../data/g_er_test/g_er_%s.graphml' %z)
    

def main():
    z_list = np.arange(1,10,1)
    p = mp.Pool(10)
    p.map(gen_g_i, z_list)

if __name__== "__main__":
    main()