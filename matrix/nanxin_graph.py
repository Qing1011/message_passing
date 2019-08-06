#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#from numba import jit
import networkx as nx
#import matplotlib.pyplot as plt
import multiprocessing as mp

# d=5              #random d-regular graph
# nList = [4000]
# zetaList = list(np.linspace(0.60,0.66,num=31))

# # finite size scaling w.r.t end time T
# #rList = [2,4,8,16]  # T=r*N, T0=(r-1)*T, activity [T0:T-1]
# rList = [2]          
# # S1 = 100         # num of random graphs and initial config
# # S2 = 10000         # num of realisations
# #S = 100000         # num of realisations
# S = 2

def rd_reg(d,N):
    G=np.zeros((N,d+1),dtype=np.int32)
    grph = nx.random_regular_graph(d,N)
    for i in range(N):
        G[i][:-1] = list(grph[i])[:]
    return (G)


# input: network topology G[:][:-1] and density rho
# output: time series of active site number aTS[0:T-1]
#@jit(nopython=True)
def avl_surv(G,density,T,d):
    G[:,-1] = 0         # initialize
    N = len(G)
    N0 = np.int(np.floor(N*density))
    occ_sites = np.random.choice(N,N0,replace=False)
    trg = occ_sites[0] 
    G[trg][-1] = 2      # trigger site with two particles
    for j in occ_sites[1:]: 
        G[j][-1] = 1    # occupied substrate sites with one particle
#    print(G)
    aNumTS=[0]*T
    aList=[trg]    
    for t in range(T):
        if aList: # if aList not empty
            aNumTS[t] = len(aList)
            tList=aList[:]  # copy list by value
            # toppling; tList = aList + [receiving sites]
            for k in aList:
                G[k][-1] -= 2
                rcv_sites = G[k][np.random.choice(d,2)]
                for i in rcv_sites:
                    G[i][-1] += 1     
                tList += list(rcv_sites)                          
            # update aList from tList, on condition that ative after toppling
            tList = list(set(tList)) 
            aList = [idx for idx in tList if G[idx][-1]>=2]
        else:
            break
    
    if aNumTS[-1]>=1:
        return 1
    else:
        return 0


def zeta_re(g_i,zeta_i,endt_i,S_i,d_i):
    p = mp.Pool(mp.cpu_count())
    res = [p.apply_async(avl_surv, args = (g_i,zeta_i,endt_i,d_i)) for i in range(S_i)]
    surv_run = [pi.get() for pi in res]
    #print (len(surv_run))
    num_surv = np.sum(surv_run)
    Psurv_i = num_surv/S_i
    return Psurv_i


def main():
    param = ''
    d = 5              #random d-regular graph
    n = 4000
    zetaList = list(np.linspace(0.60,0.64,num=9))
    rList = [16]          
    S = 10000
    g = rd_reg(d,n)
    for r in rList:
        endt=r*n
        param='N'+str(n)+'T'+str(endt)
        data=[]
        for zeta in zetaList:
            Psurv = zeta_re(g,zeta,endt,S,d)
            data.append((zeta,Psurv))
        
        np.save('../results/nanxin_new/'+param,data)


if __name__ == "__main__":
    main()
