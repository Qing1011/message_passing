#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
xx = pd.read_csv('../results/er/results.txt',header=None,names = ['theta', 'x', 'z', 'h'], sep=' ')
aaa = xx.values.tolist()


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
    return i_c1, i_c2, i   


for z_index in range(10,len(z_list)):
    z_i = z_list[z_index]
    h_z_key = aaa[50000*(z_index):50000*(z_index+1)]
    mu_dict = {}
    c1c2_dict = {}
    for theta_index in range(19,len(theta_list)):
        theta_i = theta_list[theta_index]
        h_list_i = h_z_key[500*theta_index:500*(theta_index+1)]
        
        mu_vec = [(x_list[i]- h_list_i[i][3])/(1-h_list_i[i][3]) for i in range(len(x_list))]
        mu_dict[(z_i,theta_i)] = mu_vec
        i_c1, i_c2, x_pos = nu_crit(mu_vec)
        x = x_list[x_pos]
        c1c2_dict[(z_i,theta_i)] = (i_c1, i_c2, x)
    z_all = {'mu':mu_dict,'c1c2':c1c2_dict}
    pickle.dump(z_all, open('../results/er/z_new/z_{}.p' .format(z_i),'wb'))


    z_list = list(np.arange(1,16,0.1))
    z_index = 20
    z_i = z_list[z_index]
    h_z_key = aaa[50000*(z_index):50000*(z_index+1)]
    theta_list = np.arange(0.01,1,0.01)
    theta_index = 29
    theta_i = theta_list[29]
    h_list_i = h_z_key[500*theta_index:500*(theta_index+1)]