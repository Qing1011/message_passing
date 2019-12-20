# -*- coding: utf-8 -*-
"""
Created on 1 Sep 2016

@author: blaine

threshold dynamics on undirected graphs
fixed fractional threshold
"""

import random
#import math
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
# from networkx.utils import powerlaw_sequence
# from networkx.generators.classic import empty_graph
# from networkx.generators.random_graphs import _random_subset
# from scipy.stats import gaussian_kde
# from itertools import groupby

path='D:/src/Python Scripts/csd/' 
axis_font = {'fontname':'Arial', 'size':'25'}


S=5            # num of realizations (ensemble size)
T=800         # time steps     
T0=400         # assumed time to ensure stationary state
ntSize=2000    # networks size
phi0=0.2       # fraction of nodes alive to keep the network 'alive' 
# init_d=0.0         
lambda0=0.0025       # rate of spontaneous death


gtype='rd'      # rd (poisson), sf (power law), BA (Barabasi-Albert) 
th=0.50001      #threshold for death spreading
survTh=1-th     #threshold for survival
k_max=500       #cutoff deg
k_mean=30       #mean deg                (rd)
a=3.0           #exponent                (sf)       
m=5             #num of added edges      (BA)
k_min=m         #min deg                 (sf)



if gtype=='rd':   
    print ('P_k = Pois(k,z)')
    print ('<k> = ' + str(k_mean) )
    gparam=gtype+'_th'+str(round(th,2))+'z'+str(k_mean)
    
#==============================================================================
# continuous power law distr f(x)=(a-1)*x**-a  x>=1
# z=(a-1)/(a-2), or a=2+1./(z-1)   2<a<3
# for deg distr, exponent a can be smaller than 2  
#==============================================================================
elif gtype=='sf':
    deg=np.arange(k_min,k_max+1)
    c=1./np.sum(deg**-a)
    deg_avg=np.sum(c*deg**(-a+1))
    print ('P_k = c*k^(-a)')
    print ('a = ' + str(a))
    print ('c = ' + str(c))
    print ('k_min = ' + str(k_min))
    print ('<k> = ' + str(deg_avg))  
    gparam=gtype+'_th'+str(round(th,2))+'a'+str(a)+'m'+str(m)
    seq_powerlaw=[]
    ntSize_a=0
    for k in range(k_min,k_max):
        n_k = int(ntSize*c*k**-a)
        seq_powerlaw += n_k*[k]
        ntSize_a += n_k
    ntSize = ntSize_a
    if sum(seq_powerlaw)%2 != 0:
        seq_powerlaw[-1] += 1
elif gtype=='BA':
    print ('P_k = 2*m*(m+1)/[k*(k+1)*(k+2)]')
    print ('<k> = ' + str(2*m)) 
    gparam=gtype+'_th'+str(round(th,2))+'m'+str(m)

print ('------------------------------------------')
print ('threshold = ' + str(th))
          



'''
random_degree_sequence_graph(sequence[, ...])	
       simple random graph with the given degree sequence

    create_degree_sequence(n[, sfunction, max_tries])
    powerlaw_sequence(n, exponent=2.0)   
    discrete_sequence(n, distribution=None, cdistribution=None)        

expected_degree_graph(w, seed=None, selfloops=True)
       random graph with given expected degrees.
 
fast_gnp_random_graph(n, p[, seed, directed])
gnp_random_graph(n, p[, seed, directed])
       G_{n,p} random graph, Erdős-Rényi 

barabasi_albert_graph(n, m[, seed])
       growing preferential attachment graph, power law degree distribution 
 
//random_powerlaw_tree(n[, gamma, seed, tries])
       tree with a power law degree distribution

In general, should draw degree sequence from certain distribution
     and then use random_degree_sequence_graph;
For simplicity, use fast_gnp_random_graph for Poisson
'''


##################################
# input current graph (unstable after initial deaths), 
#       original graph (for threshold check), 
#       and initial deaths
# output current graph after cascade (stable)
# Cascade deaths independent of update order, irrelevant of update scheme
def cascade(cgraph,ograph,nodes_d1):            
    nodes_d=nodes_d1      
    while nodes_d:  
        # find vunerable nodes, i.e. nodes affected by deaths in recent microstep         
        nodes_vnr=[]
        for x in nodes_d:
            nodes_vnr += [y for y in ograph.neighbors(x) 
            if y not in nodes_vnr and y in cgraph.nodes()]    
#            for y in ograph.neighbors(x):
#                if y not in nodes_vnr and y in cgraph.nodes():
#                    nodes_vnr.append(y)       
        # check survivability of vunerable nodes, record cascade deaths if any            
        nodes_d=[]
        for x in nodes_vnr:          
            # threshold check
            if len(cgraph.neighbors(x)) < survTh*cgraph.node[x]['sup_num']:  
                cgraph.remove_node(x)
                nodes_d.append(x)
                
    return cgraph
 
   
##################################
# input current graph, original graph, and recovery rate
# output current graph after recovery
# For recovery to be independent of update order, use synchronous update: 
#   Admit all potential recovery, then cascade.     
def recover(cgraph,ograph,lambda1):
    nodes_dead = [x for x in ograph.nodes() if x not in cgraph.nodes()]    
    num_r=np.random.binomial(len(nodes_dead),lambda1)
    nodes_r=random.sample(nodes_dead, num_r) 
#    nodes_r = [u for u in nodes_dead if random.random() <= lambda1]
    ### recover nodes and attached edges
    for x in nodes_r:             
        cgraph.add_node(x)
        cgraph.node[x]['sup_num']=len(ograph.neighbors(x))
        cneighbor = [y for y in ograph.neighbors(x) if y in cgraph.nodes()]
        cgraph.add_edges_from(zip(cneighbor,[x]*len(cneighbor)))
    ### check recovered nodes, cascade to kill unsurvivable ones
    nodes_d=[]
    # first microstep of cascade, assuming all recovered nodes vunerable
    for x in nodes_r:   
        if len(cgraph.neighbors(x)) < survTh*cgraph.node[x]['sup_num']:    
            cgraph.remove_node(x)
            nodes_d.append(x)
    # ensuing microsteps of cascade        
    nodes_d1=nodes_d
    cgraph=cascade(cgraph,ograph,nodes_d1)
    
    return cgraph


##################################
# threshold dynamics on 'grph' for T steps
def thrshd_dynam(grph,lambda_ratio,rate_d=lambda0,N=ntSize):
    phi_data=[]  
#    csd_data=[]
    rate_r=lambda_ratio*rate_d

    grph_live=grph.copy()
    nodes_live=grph_live.nodes()
    num_live=len(nodes_live)
    
    for t in range(0,T+1):
        phi=float(num_live)/N
        phi_data.append(phi)
        if (phi<phi0):
            break

        ### spontaneous deaths        
        num_d1=np.random.binomial(num_live,rate_d)  # np.random.poisson(rate_d*num_live) 
        nodes_d1=random.sample(nodes_live, num_d1)

#        nodes_d1=[u for u in nodes_live if random.random() <= rate_d]
        grph_live.remove_nodes_from(nodes_d1) 
        num_live -= num_d1                 
        ### cascade deaths & recovery
        grph_live=cascade(grph_live,grph,nodes_d1)   
        
#==============================================================================
#         nodes_live=grph_live.nodes()
#         num_d2=num_live-len(nodes_live)     # cascade size
#         num_live -= num_d2
#         # collect cascade stats when stationary
#         if t>T0 and num_d2<0.3*N:          
#             csd_data.append(num_d2)
# #        print num_d2    
#==============================================================================
            
        grph_live=recover(grph_live,grph,rate_r)
        nodes_live=grph_live.nodes() 
        num_live=len(nodes_live)
        
    return (phi_data,t)    # data_csd
    

##################################
# run threshold dynamics for S realizations
def dynam_stats(R,rate_d=lambda0,N=ntSize,phi_e=0):
    simParam=gparam+'_R'+str(R)[:6]+'d'+str(rate_d)+'N'+str(N)+'T'+str(T)+'S'+str(S)
    scounter=0      # num of steady realizations     
    phi_avg=np.zeros(T+1)
    phi_eq=0
    phi_eq_std=0
    t_c_sim=[]
    t_c=0
    t_c_std=0
    phi_c_sim=[]
    phi_c=0
    phi_c_std=0
    print ('R = ' + str(R))
    
    plt.figure(1,figsize=(15,15),dpi=600)  
    plt.title(r'$Poisson, z = 30; \, R = %.4f$' %R, fontsize = 30) #manual set of fig title
    plt.xlabel(r'$t$', fontsize=30) #**axis_font
    plt.ylabel(r'$\phi$', fontsize=30)
    plt.tick_params(axis = 'x', top ='off', labelsize = 25)
    plt.tick_params(axis = 'y', right = 'off', labelsize = 25)
    ## with comparison to theoretical phi(t)
    plt.figure(2,figsize=(15,15),dpi=600) 
    plt.title(r'$Poisson, z = 30; \, R = %.4f$' %R, fontsize = 30) 
    plt.xlabel(r'$t$', fontsize=30)
    plt.ylabel(r'$\phi$', fontsize=30)
    plt.tick_params(axis = 'x', top ='off', labelsize = 25)
    plt.tick_params(axis = 'y', right = 'off', labelsize = 25)
    
    for s in range(S):
        time0=time.time()
        ### generate grph
        if gtype == 'rd':
            p=float(k_mean)/N
            grph=nx.fast_gnp_random_graph(N,p)   
        elif gtype == 'sf': 
            ## simple graph powerlaw sequence configuration model
            grph=nx.random_degree_sequence_graph(seq_powerlaw,tries=1000)            
#            powerlaw_a = lambda x: powerlaw_sequence(x, exponent=a)
#            seq=nx.utils.create_degree_sequence(N,powerlaw_a,max_tries=1000)
#            grph=nx.random_degree_sequence_graph(seq,tries=1000)
            ## approximate powerlaw sequence configuration model
#            seq_avg=powerlaw_sequence(N, exponent=a)
#            grph=nx.expected_degree_graph(seq_avg)            
        elif gtype == 'BA':
            m=k_mean//2
            grph=nx.barabasi_albert_graph(N,m)   
            
#        print '------------------------------------------'
#        print 'graph generated'  
#        print 'time = %.2f' %(time.time()-time0)           
        for i in range(N):
            grph.node[i]['sup_num']=len(grph.neighbors(i)) 
        
        phi_data, endt = thrshd_dynam(grph,R)   
        plt.figure(1)
        plt.plot(phi_data,linewidth=3)
        plt.figure(2)
        plt.plot(phi_data,linewidth=3)
        print ('------------------------------------------')
        print ('realization %d complete' %(s+1))
        print ('time = %.2f' %(time.time()-time0))  
        ### sum of phi for steady realizations, phi_c & t_c for collapsing ones
        if endt==T:
            scounter+=1
            phi_array=np.asarray(phi_data)
            phi_avg=np.add(phi_avg,phi_array)  # sum over realizations
        else:
            t_c_sim.append(endt)
            print ('t_c = %d' %endt)
            phi_c_sim.append(phi_data[-2])
            print ('phi_c = %.4f' %phi_data[-2])   
  
    ## extract theoretical estimation of phi(t), phi_thry
    fn_phithry = 'phiT_'+gparam+'_R'+str(R)+'.txt'
    fp_phithry = path + fn_phithry
    with open(fp_phithry) as f:
        lines = f.read().splitlines()
    dt_line=[x for x in lines]
    dt=[[x for x in line.rstrip().split()] for line in dt_line]
    phi_thry=[float(x[1]) for x in dt]    
    
    ## if a realization is steady, consider active phase, otherwise collapsing phase
    if scounter>0:
        phi_avg=phi_avg/scounter  # average over realizations
        phi_eq=np.average(phi_avg[T0:T])
        phi_eq_std=np.std(phi_avg[T0:T])
        np.savetxt('phi_'+simParam+'.txt', phi_avg, fmt='%.4f')
        # adjust axis and save plot for simulations
        plt.figure(1)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0.5,1))
        plt.savefig('phi_'+simParam+'.jpg')
        # plot phi_thry as comparison, add legend and save
        plt.figure(2)           
        plt.axis((x1,x2,0.5,1))
        lineThry,=plt.plot(phi_thry,'k--',linewidth=6,label='theory')
        plt.legend(handles=[lineThry], loc=1, fontsize=30)
        plt.savefig('phi_'+simParam+'_compare.jpg')
    else:
        t_c=np.average(t_c_sim)
        t_c_std=np.std(t_c_sim)
        phi_c=np.average(phi_c_sim)
        phi_c_std=np.std(phi_c_sim)
        # save plot for simulations
        plt.figure(1)
        plt.savefig('phi_'+simParam+'.jpg')
        # plot phi_thry as comparison, add legend and save
        plt.figure(2)   
        phi_c_thry=phi_thry[-1]
        t_c_thry=len(phi_thry)
        lineThry,=plt.plot(phi_thry,'k--',linewidth=6,label='theory')
        plt.plot(np.ones(10)*t_c_thry,np.linspace(0,phi_c_thry,num=10),'k--')
        plt.legend(handles=[lineThry], loc=1, fontsize=30)        
        plt.savefig('phi_'+simParam+'_compare.jpg')
        
        
#    fig1=plt.figure(1)    
#    ax=fi1g.add_subplot(111)    
#    plt.text(1.02, 1,r'$\bar{\phi} = %.4f$' %phi_eq, transform=ax.transAxes)
#    plt.text(1.02, 0.9,r'$\phi_{e} = %.4f$' %phi_e, transform=ax.transAxes)
         
    
    if t_c>0:
        print ('------------------------------------------')
        print ('<t_c> = %.2f' %t_c)
        print ('std(t_c) = %.2f' %t_c_std)
        print ('<phi_c> = %.4f' %phi_c)
        print ('std(t_c) = %.4f' %phi_c_std)
        print ('------------------------------------------')
        # read data from file and add theoretical phi(t)
           
    plt.show()    
        
    return phi_avg, phi_eq, phi_eq_std, t_c, t_c_std, phi_c, phi_c_std
    

    
################################## main ##################################

###### simulaton with designated R (active or collapsing phase)
fp = gparam+'N'+str(ntSize)+'S'+str(S)+'_R'
Rlist=[1.0]
sv_data_c=[]


for R in Rlist:
    fp = fp+'_'+str(R)[:3]
    phi_avg, phi_eq, phi_eq_std, t_c, t_c_std, phi_c, phi_c_std \
    = dynam_stats(R,phi_e=0) 
    sv_data_c.append((R, t_c, t_c_std, phi_c, phi_c_std))    

np.savetxt('Tc_Sim_'+fp+'.txt',sv_data_c,fmt='%.4f %.2f %.2f %.4f %.4f')

###### active phase with R calculated theoretically given eta 
#==============================================================================
# fp = gparam+'N'+str(ntSize)+'T'+str(T)+'S'+str(S)
# RidxList=[29] 
# Rlist=[]
# etaList=[]
# rhoList=[]
# rhoSimList=[]
# #rhoTrivList=[]
# phiList=[]
# phiSimList=[]
# sv_R_rho_rhoSim=[]    
# 
# ## get R list (with respect to eta-rho points before cidx) to use for simulation 
# fn_Rdata = 'lambdaRatio_'+gparam+'.txt'
# fp_Rdata = path + fn_Rdata
# with open(fp_Rdata) as f:
#     lines = f.read().splitlines()
# data=[x for x in lines]
# L=len(data)             # L=cidx, idx \in [1,cidx]
# dt=[[x for x in line.rstrip().split()] for line in data]
# dt_R=[float(x[1]) for x in dt]      
#  
# R_array=np.asarray(dt_R)                 
# 
# ## get theoretical rho before cidx for comparison, and extend at cidx 
# fn_eta_rho = 'eta-rho_'+gparam+'.txt'
# fp_eta_rho = path + fn_eta_rho
# with open(fp_eta_rho) as f1:
#     lines1 = f1.read().splitlines()
# data1=[x for x in lines1]
# dt1=[[x for x in line.rstrip().split()] for line in data1]
# dt_eta_rho=[(int(x[0]),float(x[1]),float(x[2])) for x in dt1] 
# 
# rhoArray=np.asarray([y[2] for y in dt_eta_rho])[:L]
# rho_extend=np.linspace(rhoArray[-1],1.0,num=10)
# R_extend=np.ones(10)*R_array[-1]
# 
# 
# ##
# for idx in RidxList:
#     R=dt_R[idx-1]
#     Rlist.append(R)
# #    fp = fp+'_R'+str(R)[:4]
#     print '------------------------------------------'
#     eta_eff=dt_eta_rho[idx][1]  
#     rho_eff=dt_eta_rho[idx][2]
#     phi_eff=1-rho_eff
#     etaList.append(eta_eff)
#     rhoList.append(rho_eff)
#     phiList.append(phi_eff)
#         
#     ## run simulation using lambda ratio R 
#     phi_avg, phi_eq, phi_eq_std, t_c, t_c_std, phi_c, phi_c_std \
#     = dynam_stats(R,phi_e=phi_eff) 
#     
#     rho_eq=1-phi_eq
#     rhoSimList.append(rho_eq) 
#     phiSimList.append(phi_eq)    
#     print 'rho = '+str(rho_eff)
#     print 'rhoSim = '+str(rho_eq) 
#     
# #    rho_trv=1./(1+R)
# #    rhoTrivList.append(rho_trv)
# #    print 'rho_trv = ' + str(rho_trv)
# #    print '------------------------------------------'   
# #    sv_R_rho_rhoSim.append((R,rho_eff,rho_eq,rho_trv))
#        
# sv_R_rho_rhoSim.append((R,rho_eff,rho_eq))
# np.savetxt('compareRho_'+fp+'.txt',sv_R_rho_rhoSim,fmt='%.4f %.4f %.4f') 
# 
# 
# 
# 
# plt.figure(3,figsize=(10,10), dpi=100)
# #plt.set_title('axes title')
# plt.xlabel(r'$R$', **axis_font)
# plt.ylabel(r'$\rho^{*}$', **axis_font)
# line1,=plt.plot(R_array,rhoArray,'b-',label='theory')
# line2,=plt.plot(R_extend,rho_extend,'b--')
# lineSim,=plt.plot(Rlist,rhoSimList,'ro',label='simulation')
# plt.legend(handles=[line1,lineSim], loc=1)
# plt.savefig('R-rho_'+fp+'_.jpg') 
# plt.show()
#==============================================================================

#plt.figure(3)
#plt.xlabel(r'$\eta$')
#plt.ylabel(r'$\rho$')
#plt.plot(etaList,rhoList,'b-',etaList,rhoSimList,'ro',etaList,rhoTrivList,'g--')
#plt.savefig('compareRho_'+fp+'.jpg') 
#plt.show()










