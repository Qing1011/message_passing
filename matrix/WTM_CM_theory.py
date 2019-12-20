# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:10:09 2019

@author: NW
"""

import numpy as np
import scipy.stats as st
#from sympy.functions.special.delta_functions import Heaviside
import matplotlib.pyplot as plt
#import time
#import networkx as nx


gtype='ER'      # ER (poisson), SF (power law), BA (Barabasi-Albert) 
k_max=100       #max deg
z=5           #mean deg                (rd)
a=3.0           #exponent                (sf)       
d=5             #num of added edges      (BA)
k_min=d         #min deg                (BA,sf)
th=0.8
'''
(Standard) power law distr f(x)=(a-1)*x**-a  x>=1
            <x>=(a-1)/(a-2)
For power law degree distr f(k)=c*k**-a     1<=k<=k_max       
take a=2+1./(z-1)    (2<a<3)
and adjusted normalizing factor c will lead to <k> ~ z 
To be comparable with BA network, here we fix a=3.0
'''
if gtype == 'ER':   
    print ('P_k = Pois(k,z)')
#    print ('<k> = ' + str(z) )
    g_fparam = gtype+'_z'+str(z)+'th0'+"{:.3f}".format(th)[2:]
    g_param = [gtype,z]
elif gtype == 'SF':   
    print ('P_k = c*k^(-a)')
#    print ('a = ' + str(a))
#    print ('c = ' + str(c))
#    print ('<k> = ' + str(k_mean) )  
    g_fparam = gtype+'_a'+str(a)+'th0'+"{:.3f}".format(th)[2:]
    g_param = [gtype,a]
elif gtype == 'BA':
#    c=1./np.sum(np.divide(2.*d*(d+1),deg*(deg+1)*(deg+2)))
    print ('P_k = 2*d*(d+1)/[k*(k+1)*(k+2)]' )       # *c
#    print ('<k> = ' + str(2*deg_add) )
    g_fparam = gtype+'_d'+str(d)+'th0'+"{:.3f}".format(th)[2:]
    g_param = [gtype,d]


  
def degDistr(g_par):
    if g_par[0] == 'ER':
        k_mean = g_par[1]
        p_k_vec = st.poisson(k_mean).pmf(range(k_max+1))
    elif g_par[0] == 'SF':
        power = g_par[1]
        deg = np.arange(k_min,k_max+1)
        c = 1/np.sum(deg**-power)
        p_k_v1 = c*deg**-power
        p_k_v0 = np.zeros(k_min)
        p_k_vec = np.concatenate([p_k_v0,p_k_v1])
        k_mean = np.sum(deg*p_k_vec)
    elif gtype == 'BA':
        deg_add = g_par[1] 
        deg = np.arange(k_min,k_max+1)
        p_k_v1 = 2*deg_add*(deg_add+1)/(deg*(deg+1)*(deg+2)) 
        p_k_v0 = np.zeros(k_min)
        p_k_vec = np.concatenate([p_k_v0,p_k_v1])
        k_mean = 2*deg_add
    return p_k_vec, k_mean
    
def thrshTransExtd(x,k,th_par):
#    s=0
#    for m in range(2,k):
#        s += Heaviside(float(m)/k-th)*st.binom.pmf(m,k-1,nu)
    m = np.arange(0,k)
    theta = th_par # fixed theta
    R_mk = (np.sign(m/k-theta)+1)//2
    trans = np.sum(R_mk*st.binom(k-1,x).pmf(m))
    return trans

def thrshTrans(x,k,th_par):
    m = np.arange(0,k+1)
    theta = th_par # fixed theta
    R_mk = (np.sign(m/k-theta)+1)//2
    trans = np.sum(R_mk*st.binom(k,x).pmf(m))
    return trans

def h(x,deg_distr,deg_mean,th_par):
    h_x=0  
    for k in range(1,k_max+1):              
        h_x += k/deg_mean*deg_distr[k]*thrshTransExtd(x,k,th_par)            
    return h_x     

def g(x,deg_distr,th_par):
    g_x=0
    for k in range(1,k_max+1):
        g_x += deg_distr[k]*thrshTrans(x,k,th_par)            
    return g_x  

# when jump exists, mu(nu) increase first, turn at idx_c1, then jump to idx_c2
def nu_crit(mu_vec): 
    i_max = len(mu_vec)-1
    i_c1 = i_max
    i_c2 = i_max
    for i in range(i_max):
        if mu_vec[i+1] <= mu_vec[i]:
            i_c1 = i
            break
    if i_c1 <= i_max-2:
        for j in range(i_c1+1,i_max):
            if mu_vec[j] >= mu_vec[i_c1]:
                i_c2 = j
                break
    return i_c1, i_c2
    
################################## main #################################

n=100    # number of points 
deg_distr,deg_mean = degDistr(g_param)
th_par = th

nu_vec = list(np.linspace(0,1,num=n,endpoint=False))
mu_vec = [(nu-h(nu,deg_distr,deg_mean,th_par))/(1-h(nu,deg_distr,deg_mean,th_par)) for nu in nu_vec]
i_c1, i_c2 = nu_crit(mu_vec)
# [0,idx_c1],[idx_c2,n] -> (mu(nu), nu) 'b-' ; 
## [idx_c1+1, idx_c2-1] -> (mu(nu), nu) 'r-' ;
## [idx_c1] to [idx_c2] 'b--'

fig = plt.figure(1, figsize=(6,6), dpi=120)
plt.xlabel(r'$\mu$',fontsize=16)
plt.ylabel(r'$\nu$',fontsize=16)
if i_c1 <= n-2:
    mu_vec1 = mu_vec[:i_c1+1]
    nu_vec1 = nu_vec[:i_c1+1]
    x1, y1 = mu_vec[i_c1], nu_vec[i_c1]
    plt.plot(mu_vec1,nu_vec1,'b-')
    if i_c2 <= n-2:
        x2, y2 = mu_vec[i_c1], (nu_vec[i_c2-1] + nu_vec[i_c2])/2
        mu_vec2 = [x2] + mu_vec[i_c2:]
        nu_vec2 = [y2] + nu_vec[i_c2:]        
        plt.plot([x1,x2],[y1,y2],'b--')
        plt.plot(mu_vec2,nu_vec2,'b-')
    else:
        x2, y2 = mu_vec[i_c1], 1
        plt.plot([x1,x2],[y1,y2],'b--')
        plt.plot([x2,1],[y2,1],'b-')
else:
    plt.plot(mu_vec,nu_vec,'b-')
plt.savefig('mu-nu_'+g_fparam+'.pdf') 
plt.show()


 