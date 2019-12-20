# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 18:09:35 2016

@author: blaine
"""


import numpy as np
import scipy.stats as st
#from sympy.functions.special.delta_functions import Heaviside
import matplotlib.pyplot as plt
#import time
#import networkx as nx


gtype='sf'      # rd (poisson), sf (power law), BA (Barabasi-Albert) 
th=0.50001      #threshold
k_max=100       #cutoff deg
k_mean=30       #mean deg                (rd)
a=3.0           #exponent                (sf)       
m=5             #num of added edges      (BA)
k_min=m         #min deg                (BA,sf)
'''
(Standard) power law distr f(x)=(a-1)*x**-a  x>=1
            <x>=(a-1)/(a-2)
For power law degree distr f(k)=c*k**-a     1<=k<=k_max       
take a=2+1./(z-1)    (2<a<3)
and adjusted normalizing factor c will lead to <k> ~ z 
To be comparable with BA network, here we fix a=3.0
'''
if gtype=='rd':   
    print ('P_k = Pois(k,z)')
    print ('<k> = ' + str(k_mean) )
    gparam=gtype+'_th'+str(round(th,2))+'z'+str(k_mean)
elif gtype=='sf':
    deg=np.arange(k_min,k_max+1)
    c=1./np.sum(deg**-a)
    deg_avg=np.sum(c*deg**(-a+1))
    print ('P_k = c*k^(-a)')
    print ('a = ' + str(a))
    print ('c = ' + str(c))
    print ('<k> = ' + str(deg_avg) )  
    gparam=gtype+'_th'+str(round(th,2))+'a'+str(a)+'m'+str(m)
elif gtype=='BA':
    deg=np.arange(m,k_max+1)
#    c=1./np.sum(np.divide(2.*m*(m+1),deg*(deg+1)*(deg+2)))
    print ('P_k = 2*m*(m+1)/[k*(k+1)*(k+2)]' )       # *c
    print ('<k> = ' + str(2*m) )
    gparam=gtype+'_th'+str(round(th,2))+'m'+str(m)
    
print ('------------------------------------------')
print ('threshold = ' + str(th))

    
def deg_distr(k):
    if gtype=='rd':
        return st.poisson.pmf(k,k_mean), k_mean
    elif gtype == 'sf':
        if k>=k_min:
            p_k=c*k**-a
        else: 
            p_k=0
        return p_k, deg_avg
    elif gtype=='BA':
        if k>=m:
            p_k=2.*m*(m+1)/(k*(k+1)*(k+2))      # *c
        else:
            p_k=0.
        return p_k, 2*m
    
          

def thrshTransExtd(nu,k):
#    s=0
#    for m in range(2,k):
#        s += Heaviside(float(m)/k-th)*st.binom.pmf(m,k-1,nu)
    m=np.arange(0.,k)
    hvsArray=(np.sign(m/k-th)+1)//2
    rv=st.binom(k-1,nu)
    s=np.sum(np.multiply(hvsArray,rv.pmf(m)))
    return s

def thrshTrans(nu,k):
    m=np.arange(0.,k+1)
    hvsArray=(np.sign(m/k-th)+1)//2
    rv=st.binom(k,nu)
    s=np.sum(np.multiply(hvsArray,rv.pmf(m)))
    return s

def H(nu):
    h=0   
    for k in range(1,k_max+1):   
        p_k, z = deg_distr(k)                 
        h += float(k)/z*p_k*thrshTransExtd(nu,k)            
    return h     

def G(nu):
    g=0
    for k in range(1,k_max+1):  
        p_k, z = deg_distr(k)                   
        g += p_k*thrshTrans(nu,k)            
    return g  

def nuSol(eta,nu_max):
    nPt=50
    nuArray=np.linspace(0,nu_max,num=nPt)
    for nu in nuArray:
        y=H(nu)
        eta_n=(nu-y)/(1-y)
        if eta_n > eta:
            break
    return nu-nu_max/float(2*nPt)    

def etaEvol(eta,nu_max,r0,r1):
    nu=nuSol(eta,nu_max)
    return r0*(1-eta)*(1-H(nu))-r1*eta, nu
    
    
################################## main #################################

numPt=100    # number of points 

nuPoints=np.linspace(0,1,num=numPt,endpoint=False)
etaPoints=[]
# eta as function of nu, according to self-consistent Eq.
for nu in nuPoints:     
    y=H(nu)
    eta=(nu-y)/(1-y)
    etaPoints.append(eta)
#    print nu,eta
    

idx=0
cidx=numPt
etaPoints_adj=[0]*(numPt)

# adjusted eta
while idx<numPt-1:
    # if normal growth, keep going
    if etaPoints[idx+1]-etaPoints[idx] > 0:
#    if (etaPoints[idx+1]-etaPoints[idx])/(nuPoints[idx+1]-nuPoints[idx])>0.001:
        etaPoints_adj[idx+1]=etaPoints[idx+1]
        idx=idx+1
    # if abnormal growth, break and record critical index
    else:
        cidx=idx
        break

print ('cidx = ' + str(cidx))

# critical index smaller than end index means jump
if cidx<numPt-1:
    idx=cidx+1
    while idx<numPt:
        # abnormal nu, with no eta solution
        if etaPoints[idx]>1 or etaPoints[idx]<0:
            etaPoints_adj[idx]=etaPoints[cidx]
            nuPoints[idx]=1
        # abnormal growth turns into jump
        elif etaPoints[idx] <= etaPoints[cidx]:
#        if etaPoints[idx]<etaPoints[cidx]+0.001:
            etaPoints_adj[idx]=etaPoints[cidx];
        # resume normal growth
        else:
            etaPoints_adj[idx]=etaPoints[idx];
        idx=idx+1

#plt.figure(1)
#plt.xlabel(r'$\eta$')
#plt.ylabel(r'$\nu$')
#plt.plot(etaPoints,nuPoints,'r--')
#plt.plot(etaPointsA,nuPoints,'b-')
#plt.savefig('eta-nu_'+fparam+'.jpg') 
#plt.show()


rhoPoints=[0]*numPt
rhoPoints_adj=[0]*numPt  

for i in range(numPt):
#    rhoPoints[i]=etaPoints[i]+(1-etaPoints[i])*G(nuPoints[i])
    rhoPoints_adj[i]=etaPoints_adj[i]+(1-etaPoints_adj[i])*G(nuPoints[i])
    
eta_c=etaPoints_adj[cidx]   # record critical eta 
nu_c=nuPoints[cidx]
phi_c=(1-eta_c)*(1-H(nu_c))


plt.figure(2,figsize=(10,10), dpi=100)
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\rho$')
plt.axis([0,1,0,1])
#plt.plot(etaPoints,rhoPoints,'r--')
plt.plot(etaPoints_adj,rhoPoints_adj,'b-')
plt.savefig('eta-rho_'+gparam+'_.jpg') 
plt.show()

effData=[(i,etaPoints_adj[i],rhoPoints_adj[i]) for i in range(numPt-1)]
np.savetxt('eta-rho_'+gparam+'.txt',effData,fmt='%d %.4f %.4f')  

ratioPoints=[(i,(1-rhoPoints_adj[i])/etaPoints_adj[i]) for i in range(1,cidx+1)]    
print (ratioPoints)   
print ('-----------------------------------' )
print ('r_c = ' + str(ratioPoints[-1][1]))
np.savetxt('lambdaRatio_'+gparam+'.txt',ratioPoints,fmt='%d %.4f')  


############### numeric solution for t_c #################


#==============================================================================
# print '-----------------------------------'
# t_c_list=[]
# phi_c_list=[]
# 
# Rlist=[1.0,1.1,1.2,1.3,1.4,1.5]
# 
# for R in Rlist:
#     lambda0=0.0025 
#     lambda1=lambda0*R
#     
#     T=[0]
#     eta_T=[0]
#     phi_T=[1]
#     
#     t=0
#     t_max=800
#     eta_t=0
#     
#     while eta_t < eta_c and t < t_max:
#         t=t+1
#         T.append(t)
#         eta_dif, nu_t = etaEvol(eta_t,nu_c,lambda0,lambda1)
#         eta_t = eta_t + eta_dif
#         eta_T.append(eta_t)
#         phi_t = (1-eta_t)*(1-H(nu_t))
#         phi_T.append(phi_t)
#         print "eta(%d) = %.4f    phi(%d) = %.4f" %(t,eta_t,t,phi_t)
#     
#     print '------------------------'  
#     print 'R = %.4f' %R
#     print 't_c = %d' %t         # only for collapsing phase
#     t_c_list.append(t) 
#     print 'eta_c = %.4f' %eta_c
#     print 'nu_c = %.4f' %nu_c
#     print 'phi_c = %.4f' %phi_c      
#     phi_c_list.append(phi_c)
#     
#     plt.figure(3,figsize=(10,10),dpi=100)
#     plt.xlabel('t')
#     plt.ylabel(r'$\phi(t)$')
#     plt.axis([0, t+20, 0, 1])
#     plt.plot(T,phi_T,'b-')
#     if t < t_max:
#         plt.plot(np.ones(10)*t,np.linspace(0,phi_t,num=10),'b--')
#     plt.savefig('phiT_'+gparam+'_R'+str(R)[:4]+'.jpg') 
#     plt.show()
#     
#     dt_phiT = [(T[i],phi_T[i]) for i in range(t+1)]
#     np.savetxt('phiT_'+gparam+'_R'+str(R)[:4]+'.txt',dt_phiT,fmt='%d %.4f') 
# 
# dt_Tc = [(Rlist[i],t_c_list[i],phi_c_list[i]) for i in range(len(Rlist))]
# np.savetxt('Tc_'+gparam+'.txt',dt_Tc,fmt='%.4f %d %.4f')
#==============================================================================









