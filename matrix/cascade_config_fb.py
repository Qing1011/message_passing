#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import networkx as nx
import numpy as np
import itertools
import pickle
import scipy.stats as st
import random
import sys
from rho import assign_theta, cascade
import multiprocessing as mp

def cascade_theta(g, theta, x):
    assign_theta(g,theta)
    rho_f = cascade(g,x)
    return rho_f


def rho_theta(g_i, theta_i, no_sim):
    p = mp.Pool(30)
    x_list = np.arange(0,1,0.01)
    rho_list = [p.apply(cascade_theta,args=(g_i, theta_i,x_i)) for x_i in np.repeat(x_list,no_sim)]
    pickle.dump(rho_list, open('../results/real_data_3/cascade_configfb_{}.pickle' .format(theta_i), 'wb'))
 


def main():
    g_i = nx.read_graphml('../data/g_config_fb.graphml') 
    for theta_i in [i/100 for i in list(range(20,65,5))]:
        rho_theta(g_i, theta_i, 1000)
        #pickle.dump(rho_theta_list, open('../results/real_data_2/cascade_power_{}.p' .format(theta_i), 'wb'))


if __name__== "__main__":
    main()


# def main():
#     g_i = nx.read_graphml('../data/g_power.graphml') 
    
#     p = mp.Pool(30)
#     for theta_i in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]:
#         p.apply(rho_theta,(g_i, theta_i, 1000))
#     p.close()
#     #p.join() 

# if __name__== "__main__":
#     main()