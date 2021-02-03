# -*- coding: utf-8 -*-
"""
Plotting OMNI Data
=================== 

Importing and plotting data from OMNI Web Interface.
OMNI provides interspersed data from various spacecrafts.

"""
# import numpy as np
from pandas import read_excel, DataFrame
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os.path
from necessary_functions import G2001, get_omni_hr, using_trendet_hr, PA, ChP
from statistics import mean
# Print out the data directory path and their sizes 
from heliopy.data import helper as heliohelper
heliohelper.listdata()
# import ruptures as rpt
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# In[]: Establishing the output folder 
save_path = 'D:/Study/Academic/Research/Master Degree/Master Work/Software/Codes/Python/Heliopy Examples/auto_examples_python/'
try:
    os.mkdir(save_path + 'Output_plots')
except OSError as error:
    print(error)

# In[]: --- 
print('\nFinding the ICMEs in OMNI data and match them with the CMEs from SOHO-LASCO')
print('For more info about the method, check the paper:\nGopalswamy et al. (2010), Interplanetary shocks lacking type II radio bursts,\nThe Astrophysical Journal, 710(2), 1111.')

print('\nPredicting the transit time of the CMEs using the G2001 model\nwith mean error of:', 11.04, 'hours, according to Gopalswamy et al. 2001 .. \n')

print('For more info about the G2001 model, check this paper:\nGopalswamy, N., Lara, A., Yashiro, S., Kaiser, M. L., and Howard,\nR. A.: Predicting the 1-AU arrival times of coronal mass ejections,\nJ. Geophys. Res., 106, 29 207, 2001a.\n')
print('And this paper:\nOwens, M., & Cargill, P. (2004, January).\nPredictions of the arrival time of Coronal Mass Ejections at 1AU:\nan analysis of the causes of errors.\nIn Annales Geophysicae (Vol. 22, No. 2, pp. 661-671). Copernicus GmbH.\n')
print('=========================================================')

# In[]: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# ------- APPLYING 3 METHODS ------- 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# In[]: 
sample = read_excel('List_from_Interplanetary shocks lacking type II radio bursts_paper_178_CME-ICME_pairs.xlsx')

# For testing purposes 
sample = sample.head(3)

# Create an empty table to be filled with the CME info and its estimated transit time 
final_table = []

# Finalize the output  
cols = sample.columns.insert(sample.shape[1]+1, 'PA_trans_time_hrs')
cols = cols.insert(len(cols)+1, 'PA_est_ICME_datetime')
cols = cols.insert(len(cols)+1, 'ChP_trans_time_hrs')
cols = cols.insert(len(cols)+1, 'ChP_est_ICME_datetime')

final_table = DataFrame(columns=cols)

# To prevent plots from showing up for more speed 
# import matplotlib
# matplotlib.use('Agg')
# plt.ioff()

for event_num in range(len(sample) ):
        
    arrival_datetime = G2001(sample.CME_Datetime[event_num], sample.CME_Speed[event_num])
    
    dt = arrival_datetime - sample.CME_Datetime[event_num]
    print('CME launced on:', sample.CME_Datetime[event_num])
    print('Estimated arrival time is:', arrival_datetime)
    print('Estimated Transit time is: {} days, {} hours, {} minutes' .format(dt.components.days, 
                                                                             dt.components.hours, 
                                                                             dt.components.minutes))
    print('-------------------------------------------------------')
    
    # ------------------------------------------------------ 
    start_window = arrival_datetime - timedelta(hours=12) # hours=11.04 
    end_window = arrival_datetime + timedelta(hours=12) # hours=11.04 
    
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
    
    # ------------------------------------------------------ 
    
    # using the 'parametric approach' 
    AVG, arrival_time_PA = PA(omni_data, sample, event_num, arrival_datetime)
    
    # using the 'change-point' method 
    chp_ICME_est_arrival_time, arrival_time_ChP = ChP(omni_data, sample, event_num)
    
    # using the 'trendet' package 
    '''
        ADD EXCEPTIONS FOR NOT FINDING UP_TREND OR DOWN_TREND COLUMNS ... 
    '''
    # arrival_time_trendet = using_trendet_hr(omni_data)

    # ---------------------------------------------------------------------------- 
    
    final_table = final_table.append({'ICME_Datetime': sample.ICME_Datetime[event_num], 
                                      'CME_Datetime': sample.CME_Datetime[event_num], 
                                      'W': sample['W'][event_num], 
                                      'CME_Speed': sample['CME_Speed'][event_num], 
                                      'Shock_Speed': sample['Shock_Speed'][event_num], 
                                      'a': sample['a'][event_num], 
                                      'ICME_Speed': sample.ICME_Speed[event_num], 
                                      'Trans_Time': sample['Trans_Time'][event_num], 
                                      'PA_trans_time_hrs': arrival_time_PA, 
                                      'PA_est_ICME_datetime': AVG, 
                                      'ChP_trans_time_hrs': arrival_time_ChP, 
                                      'ChP_est_ICME_datetime': chp_ICME_est_arrival_time}, 
                                     ignore_index=True)
                        

print('Final Report:\n===============')
print('Total number of CMEs:', len(sample))
print('Number of CMEs with Dst index =< -40 nT:', len(final_table))
print('Number of skipped CMEs during the PA:', len(sample) - len(final_table))

# final_table.to_excel('Matched_List_'+str(len(final_table))+'_CME-ICME_pairs.xlsx')

# PLOT SPEED VS TRANSIT TIME 
plt.figure()
plt.scatter(final_table['CME_Speed'], final_table['PA_trans_time_hrs'], label='Model')
plt.scatter(final_table['CME_Speed'], final_table['Trans_Time'], label='Actual')
plt.legend(loc=0, frameon=False)
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel(r'$Transit$ $time$ $(hrs)$')
plt.title('Estimated CMEs transit times using parametric approach')
# plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'PA_V_vs_T.png'))
plt.show()

plt.figure()
plt.scatter(final_table['CME_Speed'], final_table['ChP_trans_time_hrs'], label='Model')
plt.scatter(final_table['CME_Speed'], final_table['Trans_Time'], label='Actual')
plt.legend(loc=0, frameon=False)
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel(r'$Transit$ $time$ $(hrs)$')
plt.title('Estimated CMEs transit times using dynamic programming search method')
# plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'ChP_V_vs_T.png'))
plt.show()

# Calculation of Root Mean Squared Error (RMSE) 
PA_rmse = sqrt(mean_squared_error(final_table['Trans_Time'], final_table['PA_trans_time_hrs']))
ChP_rmse = sqrt(mean_squared_error(final_table['Trans_Time'], final_table['ChP_trans_time_hrs']))

# Calculation of absolute percentage error 
PA_abs_err = abs((final_table['Trans_Time']-final_table['PA_trans_time_hrs'])/final_table['Trans_Time']) * 100
ChP_abs_err = abs((final_table['Trans_Time']-final_table['ChP_trans_time_hrs'])/final_table['Trans_Time']) * 100

print('\nFor Parametric Approach:\n-----------------------')
print('RMSE:', round(PA_rmse,2), 'hours')
print('MAPE:', round(mean(PA_abs_err),2), '%')

print('\nFor Dynamic Programming Search Method:\n----------------------------------')
print('RMSE:', round(ChP_rmse,2), 'hours')
print('MAPE:', round(mean(ChP_abs_err),2), '%')

# Distribution of Error 
plt.figure()
plt.hist2d(final_table['CME_Speed'], PA_abs_err, bins=10)
plt.colorbar()
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel('MAPE (%)')
plt.tight_layout()
# plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'PA_hist_V_vs_Err.png'))
plt.show()

plt.figure()
plt.hist2d(final_table['CME_Speed'], ChP_abs_err, bins=10)
plt.colorbar()
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel('MAPE (%)')
plt.tight_layout()
# plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'ChP_hist_V_vs_Err.png'))
plt.show()

plt.figure()
ymin, ymax = plt.ylim()
plt.hist(PA_abs_err, bins=10, alpha=0.7)
plt.axvline(PA_abs_err.mean(), color='k', linestyle='dashed', 
            linewidth=1, label='Mean = '+str(round(PA_abs_err.mean(),2))+'%')
plt.legend(loc='best', frameon=False)
plt.xlabel('MAPE (%)')
plt.ylabel('Frequency')
plt.tight_layout()
# plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'PA_hist_err.png'))
plt.show()

plt.figure()
ymin, ymax = plt.ylim()
plt.hist(ChP_abs_err, bins=10, alpha=0.7)
plt.axvline(ChP_abs_err.mean(), color='k', linestyle='dashed', 
            linewidth=1, label='Mean = '+str(round(ChP_abs_err.mean(),2))+'%')
plt.legend(loc='best', frameon=False)
plt.xlabel('MAPE (%)')
plt.ylabel('Frequency')
plt.tight_layout()
# plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'ChP_hist_err.png'))
plt.show()

# In[]: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

# sample = read_excel('List_from_Interplanetary shocks lacking type II radio bursts_paper_178_CME-ICME_pairs.xlsx')

# arrival_datetime = G2001(sample.CME_Datetime[event_num], sample.CME_Speed[event_num])

# dt = arrival_datetime - sample.CME_Datetime[event_num]
# print('\nCME launced on:', sample.CME_Datetime[event_num])
# print('Estimated arrival time is:', arrival_datetime)
# print('Estimated Transit time is: {} days, {} hours, {} minutes' .format(dt.components.days, 
#                                                                          dt.components.hours, 
#                                                                          dt.components.minutes))
# print('-------------------------------------------------------')

# start_window = arrival_datetime - timedelta(hours=12) # hours=11.04 
# end_window = arrival_datetime + timedelta(hours=12) # hours=11.04 



# start_datetime = datetime(start_window.year,
#                           start_window.month,
#                           start_window.day,
#                           start_window.hour,
#                           start_window.minute,
#                           start_window.second)

# end_datetime = datetime(end_window.year,
#                         end_window.month,
#                         end_window.day,
#                         end_window.hour,
#                         end_window.minute,
#                         end_window.second)


# event_num = 20

# resolution = 'hourly' # 1-minute' or 'hourly' 

# # start_datetime = datetime(2013, 7, 12)
# # end_datetime = datetime(2013, 7, 20)

# if resolution == '1-minute':
#     omni_data = get_omni_min(start_datetime, end_datetime)
#     # using the 'trendet' package 
#     arrival_time_trendet = using_trendet_min(omni_data)
    
# elif resolution == 'hourly':
#     omni_data = get_omni_hr(start_datetime, end_datetime)
    
#     # using the 'parametric approach' 
#     arrival_time_PA = PA(omni_data, sample, event_num, arrival_datetime)
    
#     # using the 'change-point' method 
#     arrival_time_ChP = ChP(omni_data, sample, event_num)
    
#     # using the 'trendet' package 
#     arrival_time_trendet = using_trendet_hr(omni_data)


























