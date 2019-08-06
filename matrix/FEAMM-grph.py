# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:50:52 2019

@author: N.WEI
"""

import numpy as np
from numba import jit
#import random as rd
## since numba is not compatible with networkx, we use networkx to generate networks 
## and play with the exported data structure directly without networkx methods
import networkx as nx
#import random as rd
#import math
import matplotlib.pyplot as plt
import time

#th=1             #threshold
d=5              #random d-regular graph
Nlist = [4000]
rhoList = [0.629]
#[0.60,0.62] + list(np.linspace(0.630,0.680,num=11)) + [0.70,0.72]    #d=5
#[0.50] + list(np.linspace(0.51,0.65,num=15)) + [0.70]     #d=10
#[0.66,0.67,0.68,0.69,0.71,0.72]
#list(np.linspace(0.755,0.820,num=14)) + [0.85]            #d=3
#list(np.linspace(0.758,0.770,num=13))
#[0.65,0.70,0.72,0.74,0.75] + list(np.linspace(0.751,0.770,num=20))

r = 5          # T=N*r, T0=N*(r-1), relaxing to stationarity, activity data [T0:T-1]
S1 = 10         # num of random graphs and initial config
S2 = 1000         # num of realisations


# generate random d-regular graph of size N
# input: (d,N)
# output: G=[N]*[d+1] list, G[i][:] for node i, G[i][:-1]={Nbr(i)}, G[i][-1]=0
def rd_reg(d,N):
    G=np.zeros((N,d+1),dtype=np.int32)
    grph = nx.random_regular_graph(d,N)
    for i in range(N):
        G[i][:-1] = list(grph[i])[:]
    return (G)

# input: network topology G[:][:-1] and density rho
# output: time series of active site number aTS[0:T-1]
@jit(nopython=True)
def avalanche(G,N,rho,T):
    G[:,-1] = 0         # initialize
    N0 = np.int(np.floor(N*rho))
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
    return aNumTS



## main()

#ax = plt.axes()
#ax.set_color_cycle([plt.cm.cool(i) for i in np.linspace(0, 1, len(Nlist))])

plt.figure(1, figsize=(6,6), dpi=120)
plt.xlabel(r'$\zeta$')
plt.ylabel(r'$\langle a \rangle$')

plt.figure(2, figsize=(6,6), dpi=120)
plt.xlabel(r'$\zeta$')
plt.ylabel(r'$P_{surv}$')


#plt.ylabel(r'$1-\frac{\langle a\rangle^{4}}{3\langle a\rangle^{2}}$')

#plt.figure(10, figsize=(6,6), dpi=100)
#plt.xlabel(r'$t$')
#plt.ylabel(r'$a$')

for N in Nlist:    
    T0 = N*(r-1)    # time to assume stationarity
    T = N*r         # activity data in [T0:T-1], if survival
    pr_surv = []
    ac_mean = []
    ac_mom2 = []
    ac_mom4 = []
#    binder_cum = []
    for rho in rhoList:    
        time0=time.time()
        ac_data = []
        surv_counter = 0
#    plt.figure(i, figsize=(5,5), dpi=100)
#    plt.xlabel(r'$t$')
#    plt.ylabel(r'$a(t)$')
#    plt.title(r'$\rho = %.3f$' %rho)
        for j in range(S1):
            G = rd_reg(d,N)
            for k in range(S2):
                aNumTS = avalanche(G,N,rho,T)
                if aNumTS[-1]>=1:
                    surv_counter += 1                   # survival by T ~stationary
                    ac_surv = np.average(aNumTS[T0:])/N   # activity = fraction of active sites
                    ac_data.append(ac_surv)
                else:
                    ac_data.append(0)
#                plt.figure(10)
#                plt.plot(ac_T)                                               
        surv_prob = surv_counter/(S1*S2)          
        pr_surv.append(surv_prob)               # survival probability 
        ac_m1 = np.average(ac_data)
        ac_mean.append(ac_m1)                   # mean activity
        ## random_regular_graph(d,N) & rho determine the state space (allowed configurations) 
        ##   and the transition matrix of the Markov chain (FEAMM dynamics)
        ## The recurrent classes of the MC can be either absorbing or active.
        ## Below rho_c almost all recurrent classes are absorbing (active classes and  
        ##   all the transient configs attractable by them have zero measure in thd limit, 
        ##   w.r.t. initial config distribution); 
        ## Above rho_c the process has a finite prob to survive t->infty. 
        ## Obsvble <a> averages over the contribution of the mean activity in each active class  
        ##  (w.r.t. the invariant measure of the active class)
        ac_m2 = np.average(np.power(ac_data,2))
        ac_mom2.append(ac_m2)
        ac_m4 = np.average(np.power(ac_data,4))
        ac_mom4.append(ac_m4)
#        binder_cum.append(1 - ac_m4/(3*(ac_m2)**2))   
        print ('------------------------------------------------------------------------')
        print ('d=%d N=%d, rho=%.3f  P_surv = %.6f  <a> = %.9f' %(d, N, rho, surv_prob, ac_m1))
        print ('time = %f' %(time.time()-time0)) 
    sv_data = [(rhoList[i],pr_surv[i],ac_mean[i],ac_mom2[i],ac_mom4[i]) for i in range(len(rhoList))]
    fname = '-rho-surv-ac_d'+str(d)+'N'+str(N)+'T'+str(T0)+'-'+str(T)+'S'+str(S1)+'x'+str(S2)+'.txt'
    np.savetxt(fname, sv_data, fmt='%.3f %.6f %.9f %.12f %.15f')
    plt.figure(1)
    plt.plot(rhoList,ac_mean)
    plt.figure(2)
    plt.plot(rhoList,pr_surv)

plt.show()



