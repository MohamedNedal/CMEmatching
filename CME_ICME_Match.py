# -*- coding: utf-8 -*-
"""
Run the snippets one-by-one 

"""
import os
from pandas import read_excel, DataFrame
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os.path
from necessary_functions import G2001, get_omni_hr, get_omni_min, using_trendet_min, PA, PA2_min, ChP, ChP_min
from statistics import mean
# Print out the data directory path and their sizes 
from heliopy.data import helper as heliohelper
heliohelper.listdata()
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# In[]: Establishing the output folder 
save_path = os.path.dirname(os.path.abspath('CME_ICME_Match.py'))
try: os.mkdir(save_path + 'Output_plots')
except OSError as error: print(error)
try: os.mkdir(save_path + 'omni_data_for_events_in_paper')
except OSError as error: print(error)

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
cols = cols.insert(len(cols)+1, 'trendet_est_ICME_datetime')

final_table = DataFrame(columns=cols)

# Uncomment this part to prevent plots from showing up for more speed 
# import matplotlib
# matplotlib.use('Agg')
# plt.ioff()

for event_num in range(len(sample)):
        
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
    
    # Example 
    # start_datetime = datetime(2017, 9, 7)
    # end_datetime = datetime(2017, 9, 14)
    omni_data = get_omni_hr(start_datetime, end_datetime)
    # ------------------------------------------------------ 
    
    # using the 'parametric approach' 
    AVG, arrival_time_PA = PA(omni_data, sample, event_num, arrival_datetime)
    
    # using the 'change-point' method 
    chp_ICME_est_arrival_time, arrival_time_ChP = ChP(omni_data, sample, event_num)
    
    # using the 'trendet' package 
    arrival_time_trendet = using_trendet_min(get_omni_min(start_datetime, end_datetime))

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
                                      'ChP_est_ICME_datetime': chp_ICME_est_arrival_time, 
                                      'trendet_est_ICME_datetime': arrival_time_trendet}, 
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
# >>>>>>>>>>>>>>> EVENTS LIST >>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
df = read_excel('train_test_final_dataset.xlsx', sheet_name='Filtered')

# Create an empty table to be filled with the CME info and its estimated transit time 
final_table = []

# Finalize the output  
cols = df.columns.insert(df.shape[1]+1, 'PA_trans_time')
cols = cols.insert(len(cols)+1, 'PA_est_ICME_datetime')
cols = cols.insert(len(cols)+1, 'ChP_trans_time')
cols = cols.insert(len(cols)+1, 'ChP_est_ICME_datetime')

final_table = DataFrame(columns=cols)

# To prevent plots from showing up for more speed 
import matplotlib
matplotlib.use('Agg')
plt.ioff()

for event in range(len(df)):

    tt = G2001(df['CME_Datetime'][event], df['CME_Speed'][event])
    
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
    
    omni_data = get_omni_min(start_datetime, end_datetime)
    
    omni_data = omni_data.filter(omni_data[['F', 'BX_GSE', 'BY_GSM', 'BZ_GSM', 'flow_speed', 'proton_density', 'Pressure', 'T', 'NaNp_Ratio', 'Beta', 'Mach_num', 'Mgs_mach_num', 'SYM_H']])
    omni_data = omni_data.astype('float64')
    
    # using 'parametric' approach 
    avg, pa_tt = PA2_min(omni_data, df['CME_Datetime'][event], tt, -10)
    
    # using the 'change-point' method 
    chp_ICME_est_arrival_time, arrival_time_ChP = ChP_min(omni_data, df['CME_Datetime'][event])
    
    dt_G2001 = df['ICME_Datetime'][event] - tt
    dt_PA = df['ICME_Datetime'][event] - avg
    
    print('\nCME launched on:', df['CME_Datetime'][event])
    print('Real arrival time is:', df['ICME_Datetime'][event])
    print('G2001 arrival time is:', tt)
    print('PA arrival time is:', avg)
    print('G2001 Error is:', 
          abs(dt_G2001.components.days), 'days,', 
          abs(dt_G2001.components.hours), 'hours,', 
          abs(dt_G2001.components.minutes), 'minutes.')
    print('PA Error is:', 
          abs(dt_PA.components.days), 'days,', 
          abs(dt_PA.components.hours), 'hours,', 
          abs(dt_PA.components.minutes), 'minutes.')
    
    final_table = final_table.append({'CME_Datetime': df['CME_Datetime'][event], 
                                  'W': df['W'][event], 
                                  'CME_Speed': df['CME_Speed'][event], 
                                  'a': df['a'][event], 
                                  'MPA': df['MPA'][event], 
                                  'Lat': df['Lat'][event], 
                                  'Lon': df['Lon'][event], 
                                  'Trans_Time': df['Trans_Time'][event], 
                                  'ICME_Datetime': df['ICME_Datetime'][event], 
                                  'PA_trans_time': pa_tt, 
                                  'PA_est_ICME_datetime': avg, 
                                  'ChP_trans_time': arrival_time_ChP, 
                                  'ChP_est_ICME_datetime': chp_ICME_est_arrival_time}, 
                                  ignore_index=True)
    
    fig, axs = plt.subplots(11, 1, figsize=(10,30), sharex=True)
    
    axs[0].plot(omni_data['F'])
    axs[0].set_ylabel('B_t (nT)')
    
    axs[1].plot(omni_data['BX_GSE'], label='Bx GSE')
    axs[1].plot(omni_data['BY_GSM'], label='By GSM')
    axs[1].plot(omni_data['BZ_GSM'], label='Bz GSM')
    
    axs[2].plot(omni_data['flow_speed'])
    axs[2].set_ylabel('V (km/s)')
    
    axs[3].plot(omni_data['proton_density'])
    axs[3].set_ylabel('n (/cm3)')
    
    axs[4].plot(omni_data['Pressure'])
    axs[4].set_ylabel('P (nPa)')
    
    axs[5].plot(omni_data['T'], label='T_P')
    # To check this condition Tp/Texp < 0.5 --> signature of ICME (MC) 
    if mean(omni_data['flow_speed']) > 500:
        # for the high-speed wind 
        Texp = ((0.031 * omni_data['flow_speed']) - 4.39)**2
    else:
        # for the high-speed wind 
        Texp = ((0.77 * omni_data['flow_speed']) - 265)**2
    Texp.rename('Texp', inplace=True)
    axs[5].plot(Texp, label='T_{exp}')
    axs[5].set_yscale('log')
    
    axs[6].plot(omni_data['NaNp_Ratio'])
    axs[6].set_ylabel('Na/Np\nratio')
    axs[6].set_yscale('log')
    
    axs[7].plot(omni_data['Beta'])
    axs[7].set_ylabel('Plasma\nBeta')
    axs[7].set_yscale('log')
    
    axs[8].plot(omni_data['Mach_num'])
    axs[8].set_ylabel('Mach num')
    
    axs[9].plot(omni_data['Mgs_mach_num'])
    axs[9].set_ylabel('Magnetonic\nMach num')
    
    axs[10].plot(omni_data['SYM_H'])
    axs[10].set_ylabel('Dst (nT)')
    
    for ax in axs:
        ax.axvline(df['ICME_Datetime'][event], color='green', alpha=0.5, linewidth=2, linestyle='--', label='Real')
        ax.axvline(tt, color='tomato', alpha=0.5, linewidth=2, linestyle='--', label='G2001')
        ax.axvline(avg, color='black', alpha=0.5, linewidth=2, linestyle='--', label='PA')
        ax.legend(loc='upper right', frameon=False, prop={'size': 10})
        ax.set_xlim([start_datetime, end_datetime])
        # ax.grid()
    
    plt.xlabel('Date')
    fig.tight_layout()
    sta = str(start_datetime.year)+str(start_datetime.month)+str(start_datetime.day)+str(start_datetime.hour)+str(start_datetime.minute)+str(start_datetime.second)
    end = str(end_datetime.year)+str(end_datetime.month)+str(end_datetime.day)+str(end_datetime.hour)+str(end_datetime.minute)+str(end_datetime.second)
    # plt.savefig(os.path.join(save_path, 'omni_data_for_events_in_paper' + '/', 'OMNI_Data_'+sta+'--'+end+'.png'), dpi=300)
    plt.show()

# In[]: --- plot hourly data 
omni_data = omni_data.filter(omni_data[['F1800', 'BX_GSE1800', 'BY_GSE1800', 'BZ_GSE1800', 'V1800', 'N1800', 'Pressure1800', 'T1800', 'Ratio1800', 'Beta1800', 'Mach_num1800', 'Mgs_mach_num1800', 'DST1800']])
omni_data = omni_data.astype('float64')

fig, axs = plt.subplots(11, 1, figsize=(10,30), dpi=100, sharex=True)

axs[0].plot(omni_data['F1800'])
axs[0].set_ylabel('B_t (nT)')

axs[1].plot(omni_data['BX_GSE1800'], label='Bx GSE')
axs[1].plot(omni_data['BY_GSE1800'], label='By GSM')
axs[1].plot(omni_data['BZ_GSE1800'], label='Bz GSM')

axs[2].plot(omni_data['V1800'])
axs[2].set_ylabel('V (km/s)')

axs[3].plot(omni_data['N1800'])
axs[3].set_ylabel('n (/cm3)')

axs[4].plot(omni_data['Pressure1800'])
axs[4].set_ylabel('P (nPa)')

axs[5].plot(omni_data['T1800'], label='T_P')
# To check this condition Tp/Texp < 0.5 --> signature of ICME (MC) 
if mean(omni_data['V1800']) > 500:
    # for the high-speed wind 
    Texp = ((0.031 * omni_data['V1800']) - 4.39)**2
else:
    # for the high-speed wind 
    Texp = ((0.77 * omni_data['V1800']) - 265)**2
Texp.rename('Texp', inplace=True)
axs[5].plot(Texp, label='T_{exp}')
axs[5].set_yscale('log')

axs[6].plot(omni_data['Ratio1800'])
axs[6].set_ylabel('Na/Np\nratio')
axs[6].set_yscale('log')

axs[7].plot(omni_data['Beta1800'])
axs[7].set_ylabel('Plasma\nBeta')
axs[7].set_yscale('log')

axs[8].plot(omni_data['Mach_num1800'])
axs[8].set_ylabel('Mach num')

axs[9].plot(omni_data['Mgs_mach_num1800'])
axs[9].set_ylabel('Magnetonic\nMach num')

axs[10].plot(omni_data['DST1800'])
axs[10].set_ylabel('Dst (nT)')

for ax in axs:
    ax.axvline(df['ICME_Datetime'][event], color='green', alpha=0.5, linewidth=2, linestyle='--', label='Real')
    ax.axvline(tt, color='tomato', alpha=0.5, linewidth=2, linestyle='--', label='G2001')
    ax.axvline(avg, color='black', alpha=0.5, linewidth=2, linestyle='--', label='PA')
    ax.legend(loc='upper right', frameon=False, prop={'size': 10})
    ax.set_xlim([start_datetime, end_datetime])
    # ax.grid()

plt.xlabel('Date')
fig.tight_layout()
plt.show()

# In[]: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>> INDIVIDUAL EVENT >>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
df = read_excel('train_test_final_dataset.xlsx', sheet_name='Filtered')

# Create an empty table to be filled with the CME info and its estimated transit time 
final_table = []

# Finalize the output  
cols = df.columns.insert(df.shape[1]+1, 'PA_trans_time')
cols = cols.insert(len(cols)+1, 'PA_est_ICME_datetime')
cols = cols.insert(len(cols)+1, 'ChP_trans_time')
cols = cols.insert(len(cols)+1, 'ChP_est_ICME_datetime')

final_table = DataFrame(columns=cols)

# To prevent plots from showing up for more speed 
import matplotlib
matplotlib.use('Agg')
plt.ioff()

# event = 184

for event in range(len(df)):

    tt = G2001(df['CME_Datetime'][event], df['CME_Speed'][event])
    
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
    
    omni_data = get_omni_min(start_datetime, end_datetime)
    
    omni_data = omni_data.filter(omni_data[['F', 'BX_GSE', 'BY_GSM', 'BZ_GSM', 'flow_speed', 'proton_density', 'Pressure', 'T', 'NaNp_Ratio', 'Beta', 'Mach_num', 'Mgs_mach_num', 'SYM_H']])
    omni_data = omni_data.astype('float64')
    
    # using 'parametric' approach 
    avg, pa_tt = PA2_min(omni_data, df['CME_Datetime'][event], tt, -10)
    
    # using the 'change-point' method 
    chp_ICME_est_arrival_time, arrival_time_ChP = ChP_min(omni_data, df['CME_Datetime'][event], event)
    
    dt_G2001 = df['ICME_Datetime'][event] - tt
    
    print('\nCME launched on:', df['CME_Datetime'][event])
    print('Real arrival time is:', df['ICME_Datetime'][event])
    print('G2001 arrival time is:', tt)
    print('PA arrival time is:', avg)
    print('G2001 Error is:', 
          abs(dt_G2001.components.days), 'days,', 
          abs(dt_G2001.components.hours), 'hours,', 
          abs(dt_G2001.components.minutes), 'minutes.')
    
    if avg == []:
        print('\nPA didnot find arrival time')
        dt_PA = "'---"
    else:
        dt_PA = df['ICME_Datetime'][event] - avg
        print('PA Error is:', 
              abs(dt_PA.components.days), 'days,', 
              abs(dt_PA.components.hours), 'hours,', 
              abs(dt_PA.components.minutes), 'minutes.')
    
    final_table = final_table.append({'CME_Datetime': df['CME_Datetime'][event], 
                                  'W': df['W'][event], 
                                  'CME_Speed': df['CME_Speed'][event], 
                                  'a': df['a'][event], 
                                  'MPA': df['MPA'][event], 
                                  'Lat': df['Lat'][event], 
                                  'Lon': df['Lon'][event], 
                                  'Trans_Time': df['Trans_Time'][event], 
                                  'ICME_Datetime': df['ICME_Datetime'][event], 
                                  'PA_trans_time': pa_tt, 
                                  'PA_est_ICME_datetime': avg, 
                                  'ChP_trans_time': arrival_time_ChP, 
                                  'ChP_est_ICME_datetime': chp_ICME_est_arrival_time}, 
                                  ignore_index=True)

final_table.to_excel(os.path.join(save_path, 'final_table.xlsx'))

# In[]: --- 
fig, axs = plt.subplots(3, 1, figsize=(15,6), sharex=True)
axs[0].plot(omni_data['NaNp_Ratio'])
axs[0].set_ylabel('Na/Np')

omni_data['Na'] = omni_data['NaNp_Ratio'] * omni_data['proton_density']
axs[1].plot(omni_data['Na'])
axs[1].set_ylabel('Na')

axs[2].plot(omni_data['proton_density'])
axs[2].set_ylabel('Np')

for ax in axs:
    ax.set_yscale('log')
    ax.set_xlim([start_datetime, end_datetime])

plt.tight_layout()
plt.show()

# In[]: --- 
fig, ax1 = plt.subplots(2, 1, figsize=(15,4), sharex=True)
ax1[0].plot(omni_data['NaNp_Ratio'], 'k-')

ax2 = ax1[1].twinx()
omni_data['Na'] = omni_data['NaNp_Ratio'] * omni_data['proton_density']
ax1[1].plot(omni_data['Na'], 'r-')
ax2.plot(omni_data['proton_density'], 'b-')

ax1[1].set_xlabel('Date')
ax1[1].set_ylabel('Nalpha', color='r')
ax2.set_ylabel('Nproton', color='b')

for ax in ax1:
    ax.set_yscale('log')
    ax.set_xlim([start_datetime, end_datetime])

plt.tight_layout()
plt.show()

# In[]: --- 
omni_data['Na'] = omni_data['NaNp_Ratio'] * omni_data['proton_density']
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(omni_data['NaNp_Ratio'], 'k-', label='Na/Np')
ax.plot(omni_data['Na'], 'r-', label='Na')
ax.plot(omni_data['proton_density'], 'b-', label='Np')
ax.set_xlabel('Date')
ax.set_yscale('log')
ax.set_xlim([start_datetime, end_datetime])
plt.legend(loc='best', frameon=False)
plt.tight_layout()
plt.show()