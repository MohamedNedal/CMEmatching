# -*- coding: utf-8 -*-
"""
This code to evaluate all the NN architectures and find the best model 

""" 
from pandas import read_excel, DataFrame
import matplotlib.pyplot as plt
from os.path import join

dir_path = r"D:\Study\Academic\Research\Master Degree\Master Work\Plan A [standby]\ANN Codes\[THAT'S IT]_Trying with FFNN\Final Dataset & Apply on SOHO_HCMEs\NN_CME_transitTime\FINAL_EXP_WORK\FINAL_TRY"

# In[]: --- 
report_single_HL = read_excel(dir_path+'/MLP_Outputs/report_single_HL.xlsx')
report_double_HL = read_excel(dir_path+'/MLP_Outputs/report_double_HL.xlsx')
report_triple_HL = read_excel(dir_path+'/MLP_Outputs/report_triple_HL.xlsx')

# In[]: Collect the last MSE value for all NN topologies - SINGLE 
last_error_values_single = DataFrame(columns=['NN_topology','Min_MSE','Min_MAE'])
for i in range(len(report_single_HL['loss'])):
    loss = report_single_HL['loss'][i].split()
    loss = float(loss[-1][:-1])
    MAE = report_single_HL['MAE'][i].split()
    MAE = float(MAE[-1][:-1])    
    last_error_values_single = last_error_values_single.append({
        'NN_topology': str(report_single_HL['num_nodes'][i]), 
        'Min_MSE': loss, 
        'Min_MAE': MAE
        }, ignore_index=True)

min_MSE_idx = last_error_values_single.index[last_error_values_single['Min_MSE'] == min(last_error_values_single['Min_MSE'].values)]
min_MAE_idx = last_error_values_single.index[last_error_values_single['Min_MAE'] == min(last_error_values_single['Min_MAE'].values)]

print('Min MSE index =', last_error_values_single['NN_topology'][min_MSE_idx[0]])
print('Min MSE =', round(min(last_error_values_single['Min_MSE'].values), 2))

print('Min MAE index =', last_error_values_single['NN_topology'][min_MAE_idx[0]])
print('Min MAE =', round(min(last_error_values_single['Min_MAE'].values), 2))

# Plot min MSE values vs NN topologies 
plt.figure(figsize=(10,5))
plt.plot(last_error_values_single['NN_topology'], last_error_values_single['Min_MSE'])
plt.axvline(min_MSE_idx, color='red', alpha=0.5, linewidth=2, linestyle='--', label='MSE = '+str(round(min(last_error_values_single['Min_MSE'].values), 2)))
plt.xlabel('NN topology')
plt.ylabel('Min MSE')
plt.legend(loc='upper right', frameon=False)
plt.xlim(left=min_MSE_idx[0]-3, right=min_MSE_idx[0]+3)
plt.tight_layout(pad=0.01, h_pad=None)
# plt.savefig(dir_path + '/MLP_Outputs/' + 'last_mse_values_single.png', dpi=200)
plt.show()

last_error_values_single.to_excel(join(dir_path+'/MLP_Outputs/', 'last_error_values_single.xlsx'))

# In[]: Collect the last MSE value for all NN topologies - DOUBLE 
last_error_values_double = DataFrame(columns=['NN_topology','Min_MSE','Min_MAE'])
for i in range(len(report_double_HL['loss'])):
    loss = report_double_HL['loss'][i].split()
    loss = float(loss[-1][:-1])
    MAE = report_double_HL['MAE'][i].split()
    MAE = float(MAE[-1][:-1])  
    last_error_values_double = last_error_values_double.append({
        'NN_topology': str(report_double_HL['num_nodes'][i]), 
        'Min_MSE': loss, 
        'Min_MAE': MAE
        }, ignore_index=True)

min_MSE_idx = last_error_values_double.index[last_error_values_double['Min_MSE'] == min(last_error_values_double['Min_MSE'].values)]
min_MAE_idx = last_error_values_double.index[last_error_values_double['Min_MAE'] == min(last_error_values_double['Min_MAE'].values)]

print('Min MSE index =', last_error_values_double['NN_topology'][min_MSE_idx[0]])
print('Min MSE =', round(min(last_error_values_double['Min_MSE'].values), 2))

print('Min MAE index =', last_error_values_double['NN_topology'][min_MAE_idx[0]])
print('Min MAE =', round(min(last_error_values_double['Min_MAE'].values), 2))

# Plot min MSE values vs NN topologies 
plt.figure(figsize=(10,5))
plt.plot(last_error_values_double['NN_topology'], last_error_values_double['Min_MSE'])
plt.axvline(min_MSE_idx, color='red', alpha=0.5, linewidth=2, linestyle='--', label='MSE = '+str(round(min(last_error_values_double['Min_MSE'].values), 2)))
plt.xlabel('NN topology')
plt.ylabel('Min MSE')
plt.legend(loc='upper right', frameon=False)
plt.xticks(rotation=90)
plt.xlim(left=min_MSE_idx[0]-5, right=min_MSE_idx[0]+5)
plt.tight_layout(pad=0.01, h_pad=None)
# plt.savefig(dir_path + '/MLP_Outputs/' + 'last_mse_values_double.png', dpi=200)
plt.show()

last_error_values_double.to_excel(join(dir_path+'/MLP_Outputs/', 'last_error_values_double.xlsx'))

# In[]: Collect the last MSE value for all NN topologies - TRIPLE 
last_error_values_triple = DataFrame(columns=['NN_topology','Min_MSE','Min_MAE'])
for i in range(len(report_triple_HL['loss'])):
    loss = report_triple_HL['loss'][i].split()
    loss = float(loss[-1][:-1])
    MAE = report_triple_HL['MAE'][i].split()
    MAE = float(MAE[-1][:-1])
    last_error_values_triple = last_error_values_triple.append({
        'NN_topology': str(report_triple_HL['num_nodes'][i]), 
        'Min_MSE': loss, 
        'Min_MAE': MAE
        }, ignore_index=True)

min_MSE_idx = last_error_values_triple.index[last_error_values_triple['Min_MSE'] == min(last_error_values_triple['Min_MSE'].values)]
min_MAE_idx = last_error_values_triple.index[last_error_values_triple['Min_MAE'] == min(last_error_values_triple['Min_MAE'].values)]

print('Min MSE index =', last_error_values_triple['NN_topology'][min_MSE_idx[0]])
print('Min MSE =', round(min(last_error_values_triple['Min_MSE'].values), 2))

print('Min MAE index =', last_error_values_triple['NN_topology'][min_MAE_idx[0]])
print('Min MAE =', round(min(last_error_values_triple['Min_MAE'].values), 2))

# Plot min MSE values vs NN topologies 
plt.figure(figsize=(10,5), dpi=100)
plt.plot(last_error_values_triple['NN_topology'], last_error_values_triple['Min_MSE'])
for i in range(len(min_MSE_idx)):
    plt.axvline(min_MSE_idx[i], color='red', alpha=0.5, linewidth=2, linestyle='--', label='MSE = '+str(round(min(last_error_values_triple['Min_MSE'].values), 2)))
plt.xlabel('NN topology')
plt.ylabel('Min MSE')
plt.legend(loc='upper right', frameon=False)
plt.xticks(rotation=90)
plt.xlim(left=min_MSE_idx[0]-5, right=min_MSE_idx[0]+5)
plt.tight_layout(pad=0.01, h_pad=None)
# plt.savefig(dir_path + '/MLP_Outputs/' + 'last_mse_values_triple.png')
plt.show()

last_error_values_triple.to_excel(join(dir_path+'/MLP_Outputs/', 'last_error_values_triple.xlsx'))
