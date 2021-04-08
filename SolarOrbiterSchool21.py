# -*- coding: utf-8 -*-
""" 
This code is for matching CME-ICME pairs from SOHO/LASCO CME catalog and OMNI database of solar and geomagnetic indices 
Created on Sat Apr  3 16:04:44 2021 
@author: Mohamed Nedal 
""" 
from get_omni_hr import get_omni_hr
from G2001 import G2001
from PA import PA
from ChP import ChP
from trendet_hr import trendet_hr

import os.path
from os.path import join
import numpy as np
from datetime import datetime, timedelta
from pandas import read_excel, DataFrame
# import heliopy.data.omni as omni
from heliopy.data import helper as heliohelper
heliohelper.listdata()
from math import sqrt
from statistics import mean
# import trendet
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# In[]: Establishing the output folder 
save_path = 'D:/Study/Academic/Research/Master Degree/Master Work/Software/Codes/Python/Heliopy Examples/auto_examples_python/'
try: os.mkdir(save_path + 'Output_plots')
except OSError as error: print(error)
try: os.mkdir(save_path + 'omni_data_for_events_in_paper')
except OSError as error: print(error)

# In[]: Import CME dataset 
data = read_excel('List_from_Interplanetary shocks lacking type II radio bursts_paper_178_CME-ICME_pairs.xlsx')

sample = data.copy()

# Take a sample 
# sample = data.iloc[16:17]
# sample = sample.reset_index()
# sample.drop('index', axis=1, inplace=True)
# print(sample)

# Create an empty table to be filled with the CME info and its estimated transit time 
final_table = []

# Finalize the output  
cols = sample.columns.insert(sample.shape[1]+1, 'PA_trans_time_hrs')
cols = cols.insert(len(cols)+1, 'PA_est_ICME_datetime')
cols = cols.insert(len(cols)+1, 'ChP_trans_time_hrs')
cols = cols.insert(len(cols)+1, 'ChP_est_ICME_datetime')
cols = cols.insert(len(cols)+1, 'trendet_est_ICME_datetime')

final_table = DataFrame(columns=cols)

# In[]: --- 
# To prevent plots from showing up for more speed 
# import matplotlib
# matplotlib.use('Agg')
# plt.ioff()

event = 6

tt = G2001(sample['CME_Datetime'][event], sample['CME_Speed'][event])

start_window = tt - timedelta(hours=24) # hours=11.04 
end_window = tt + timedelta(hours=24) # hours=11.04 

start_datetime = datetime(start_window.year,
                          start_window.month,
                          start_window.day,
                          start_window.hour,
                          start_window.minute,
                          start_window.second)

end_datetime = datetime(end_window.year,
                        end_window.month,
                        end_window.day,
                        end_window.hour,
                        end_window.minute,
                        end_window.second)

omni_data = get_omni_hr(start_datetime, end_datetime)

omni_data = omni_data.filter(omni_data[['F1800', 'BX_GSE1800', 'BY_GSE1800', 
                                        'BZ_GSE1800', 'V1800', 'N1800', 
                                        'Pressure1800', 'T1800', 'Ratio1800', 
                                        'Beta1800', 'Mach_num1800', 
                                        'Mgs_mach_num1800', 'DST1800']]).astype('float64')

# using 'parametric' approach 
PA_datetime, pa_tt = PA(omni_data, sample, event, tt)

# using the 'change-point' method 
chp_ICME_est_arrival_time, arrival_time_ChP = ChP(omni_data, sample, event)

# using 'trendet' package 
trendet_trans_time = trendet_hr(omni_data)

dt_G2001 = sample['ICME_Datetime'][event] - tt

print('\nCME launched on:', sample['CME_Datetime'][event])
print('Real arrival time is:', sample['ICME_Datetime'][event])
print('G2001 arrival time is:', tt)
print('PA arrival time is:', PA_datetime)
print('G2001 Error is:', 
      abs(dt_G2001.components.days), 'days,', 
      abs(dt_G2001.components.hours), 'hours,', 
      abs(dt_G2001.components.minutes), 'minutes.')

if PA_datetime == []:
    print('\nPA did not find arrival time')
    dt_PA = "'---"
else:
    dt_PA = sample['ICME_Datetime'][event] - PA_datetime
    print('PA Error is:', 
          abs(dt_PA.components.days), 'days,', 
          abs(dt_PA.components.hours), 'hours,', 
          abs(dt_PA.components.minutes), 'minutes.')

final_table = final_table.append({'CME_Datetime': sample['CME_Datetime'][event], 
                              'W': sample['W'][event], 
                              'CME_Speed': sample['CME_Speed'][event], 
                              'a': sample['a'][event], 
                              'Trans_Time': sample['Trans_Time'][event], 
                              'ICME_Datetime': sample['ICME_Datetime'][event], 
                              'PA_trans_time': pa_tt, 
                              'PA_est_ICME_datetime': PA_datetime, 
                              'ChP_trans_time': arrival_time_ChP, 
                              'ChP_est_ICME_datetime': chp_ICME_est_arrival_time, 
                              'trendet_trans_time': trendet_trans_time}, 
                              ignore_index=True)


fig, axs = plt.subplots(7, 1, figsize=(10,15), sharex=True)

axs[0].plot(omni_data['F1800'])
axs[0].set_ylabel(r'$B_t (nT)$', fontsize=14)
axs[0].tick_params(labelsize=14)

# axs[1].plot(omni_data['BX_GSE1800'], label='B_x GSE')
# axs[1].plot(omni_data['BY_GSE1800'], label='B_y GSE')
axs[1].plot(omni_data['BZ_GSE1800'], label='B_z GSE')
axs[1].tick_params(labelsize=14)

axs[2].plot(omni_data['V1800'])
axs[2].set_ylabel(r'$V (km.s^{-1})$', fontsize=14)
axs[2].tick_params(labelsize=14)

axs[3].plot(omni_data['N1800'])
axs[3].set_ylabel(r'$n (cm^{-3})$', fontsize=14)
axs[3].tick_params(labelsize=14)

axs[4].plot(omni_data['Pressure1800'])
axs[4].set_ylabel('P (nPa)', fontsize=14)
axs[4].tick_params(labelsize=14)

axs[5].plot(omni_data['T1800'], label='T_P')
# Calculating half the expected solar wind temperature (0.5Texp) 
''' 
define the Texp as mentioned in: 
    Lopez, R. E., & Freeman, J. W. (1986). 
    Solar wind proton temperature‐velocity relationship. 
    Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
''' 
if mean(omni_data['V1800']) > 500:
    # for the high-speed wind 
    Texp = ((0.031 * omni_data['V1800']) - 4.39)**2
else:
    # for the high-speed wind 
    Texp = ((0.77 * omni_data['V1800']) - 265)**2
Texp.rename('Texp', inplace=True)
axs[5].plot(Texp, label='T_{exp}')
axs[5].set_yscale('log')
axs[5].tick_params(labelsize=14)

axs[6].plot(omni_data['DST1800'])
axs[6].set_ylabel('Dst (nT)', fontsize=14)
axs[6].tick_params(labelsize=14)
axs[6].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H:%M'))
fig.autofmt_xdate(rotation=45)

for ax in axs:
    ax.set_xlim([start_datetime, end_datetime])
    ax.axvline(sample['ICME_Datetime'][event], color='green', alpha=0.5, linewidth=2, linestyle='--', label='Real')
    ax.axvline(tt, color='tomato', alpha=0.5, linewidth=2, linestyle='--', label='G2001')
    try: ax.axvline(PA_datetime, color='black', alpha=0.5, linewidth=2, linestyle='--', label='PA')
    except ValueError as verr: print(verr)
    try: ax.axvline(chp_ICME_est_arrival_time, color='blue', alpha=0.5, linewidth=2, linestyle='--', label='ChP')
    except ValueError as verr: print(verr)
    ax.legend(loc='upper right', frameon=False, prop={'size': 12})
    # ax.grid()

plt.xlabel('Date')
fig.tight_layout()
# sta = str(start_datetime.year)+str(start_datetime.month)+str(start_datetime.day)+str(start_datetime.hour)+str(start_datetime.minute)+str(start_datetime.second)
# end = str(end_datetime.year)+str(end_datetime.month)+str(end_datetime.day)+str(end_datetime.hour)+str(end_datetime.minute)+str(end_datetime.second)
# plt.savefig(os.path.join(save_path, 'CME_num_'+str(event)+'_omni_data_for_events_in_paper' + '/', 'OMNI_Data_'+sta+'--'+end+'.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'CME_num_'+str(event)+'_omni_PA.png'), dpi=300)
plt.show()
    
# final_table.to_excel(os.path.join(save_path, 'final_table.xlsx'))

# In[]: Plot Histograms for the Residuals 
df = read_excel('final_table.xlsx')

df['dE_ChP'] = ''
for i in range(len(df)):
    df['dE_ChP'][i] = df['Trans_Time'][i] - df['ChP_trans_time'][i]
    
pa_df = df[['CME_Speed','Trans_Time','PA_trans_time']]
pa_df = pa_df.replace('[]', np.nan)
pa_df.dropna(inplace=True)
pa_df = pa_df.reset_index()

pa_df['dE_PA'] = ''
for i in range(len(pa_df)):
    pa_df['dE_PA'][i] = pa_df['Trans_Time'][i] - pa_df['PA_trans_time'][i]

from sklearn.metrics import mean_absolute_error, mean_squared_error
print('\nMAPE for PA =', round(np.mean(np.abs((pa_df['Trans_Time'] - pa_df['PA_trans_time']) / pa_df['Trans_Time'])) * 100, 2), ' %.')
print('MAPE for ChP =', round(np.mean(np.abs((df['Trans_Time'] - df['ChP_trans_time']) / df['Trans_Time'])) * 100, 2), ' %.\n')
print('\nMAE for PA =', round(mean_absolute_error(pa_df['Trans_Time'], pa_df['PA_trans_time']), 2), ' hours.')
print('MAE for ChP =', round(mean_absolute_error(df['Trans_Time'], df['ChP_trans_time']), 2), ' hours.\n')
print('\nMSE for PA =', round(mean_squared_error(pa_df['Trans_Time'], pa_df['PA_trans_time']), 2), ' hours.')
print('MSE for ChP =', round(mean_squared_error(df['Trans_Time'], df['ChP_trans_time']), 2), ' hours.\n')
print('\nRMSE for PA =', round(sqrt(mean_squared_error(pa_df['Trans_Time'], pa_df['PA_trans_time'])), 2), ' hours.')
print('RMSE for ChP =', round(sqrt(mean_squared_error(df['Trans_Time'], df['ChP_trans_time'])), 2), ' hours.\n')

# for PA 
fig, ax = plt.subplots(figsize=(12,12))
ax.hist(pa_df['dE_PA'], bins=10)
ax.set_xlabel(r'$\Delta E$ (hours)', fontsize=20)
ax.set_ylabel('Number of events', fontsize=20)
ax.axvline(pa_df['dE_PA'].mean(), color='k', linestyle='dashed', linewidth=2)
ax.tick_params(labelsize=20)
min_xlim, max_xlim = plt.xlim()
min_ylim, max_ylim = plt.ylim()
ax.text(0.06, 0.95, 'Mean = {:.2f}'.format(pa_df['dE_PA'].mean()), 
              ha='left', va='top', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.text(0.06, 0.90, 'Max = {:.2f}'.format(pa_df['dE_PA'].max()), 
              ha='left', va='top', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.text(0.97, 0.95, '(PA)', horizontalalignment='right', verticalalignment='top', fontsize=20, fontweight='bold', transform=ax.transAxes)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(join(save_path, 'hist_PA_dE.png'), dpi=700, format='png', bbox_inches='tight')
plt.show()

# for ChP 
fig, ax = plt.subplots(figsize=(12,12))
ax.hist(df['dE_ChP'], bins=10)
ax.set_xlabel(r'$\Delta E$ (hours)', fontsize=20)
ax.set_ylabel('Number of events', fontsize=20)
ax.axvline(df['dE_ChP'].mean(), color='k', linestyle='dashed', linewidth=2)
ax.tick_params(labelsize=20)
min_xlim, max_xlim = plt.xlim()
min_ylim, max_ylim = plt.ylim()
ax.text(0.06, 0.95, 'Mean = {:.2f}'.format(df['dE_ChP'].mean()), 
              ha='left', va='top', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.text(0.06, 0.90, 'Max = {:.2f}'.format(df['dE_ChP'].max()), 
              ha='left', va='top', fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.text(0.97, 0.95, '(ChP)', horizontalalignment='right', verticalalignment='top', fontsize=20, fontweight='bold', transform=ax.transAxes)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(join(save_path, 'hist_ChP_dE.png'), dpi=700, format='png', bbox_inches='tight')
plt.show()

# In[]: 


# In[]: --- 








# In[]: --- 









# In[]: --- 








# In[]: --- 









# In[]: --- 









# In[]: --- 











# In[]: --- 










# In[]: --- 










# In[]: --- 











# In[]: --- 







# In[]: --- 








# In[]: --- 





# In[]: --- 








