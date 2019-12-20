#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import comb
import itertools
from scipy.sparse.linalg import eigs
from scipy.io import mmread
import pickle
import multiprocessing as mp
import scipy.stats as st
from sklearn.model_selection import ParameterGrid
import sys


def thrshTransExtd(nu,k,th):
    m=np.arange(0.,k)
    hvsArray=(np.sign(m/k-th)+1)//2
    rv=st.binom(k-1,nu)
    s=np.sum(np.multiply(hvsArray,rv.pmf(m)))
    return s


def H(z, par):
    theta = par['theta']
    nu = par['x']
    #z = par['z']
    k_max = 1000
    h=0   
    for k in range(1,k_max+1):   
        p_k = st.poisson.pmf(k,z)              
        h += float(k)/z*p_k*thrshTransExtd(nu,k,theta)            
    return [z,theta,nu,h]


# def main():
#     z_list = list(np.arange(1,16,0.1))
#     theta_list = np.arange(0.01,1,0.01)
#     theta_list = theta_list.tolist()
#     theta_list.reverse()
#     x_list = list(np.arange(0.001,1,0.002))
#     p = mp.Pool(mp.cpu_count())
#     param_grid = {'z':z_list, 'theta': theta_list, 'x' : x_list}
#     grid = ParameterGrid(param_grid)

#     results = p.map(H, grid)
#     p.close()
#     p.join() 
#     with open('../results/er/z_theta_nu_16', 'wb') as fp:
#         pickle.dump(results, fp)


def main():
    job_index = sys.argv[1]
    z_index = int(job_index ) - 1
    z_list = list(np.arange(1,16.1,0.1))
    theta_list = np.arange(0.01,1,0.01)
    theta_list = theta_list.tolist()
    theta_list.reverse()
    x_list = list(np.arange(0.001,1,0.002))
    param_grid = {'theta': theta_list, 'x' : x_list}
    grid = ParameterGrid(param_grid)
    
    z_i = z_list[z_index]
    p = mp.Pool(30)
    results = [p.apply_async(H, args=(z_i, params)) for params in grid]
    output = [pi.get() for pi in results]
    pickle.dump(output, open('../results/er/z_nu_16/z_theta_nu_{}' .format(z_index), 'wb'))

if __name__== "__main__":
    main()
