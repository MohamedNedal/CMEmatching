# -*- coding: utf-8 -*-
"""
Plotting OMNI Data
=================== 

Importing and plotting data from OMNI Web Interface.
OMNI provides interspersed data from various spacecrafts.

"""
import numpy as np
from pandas import read_excel, DataFrame, Timestamp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os.path
from necessary_functions import get_omni_hr, G2001
from statistics import mean
# Print out the data directory path and their sizes 
from heliopy.data import helper as heliohelper
heliohelper.listdata()
import ruptures as rpt
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
# ------- Parametric Approach Based On G2001 Semi-Empirical Model ------- 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# In[]: 
sample = read_excel('List_from_Interplanetary shocks lacking type II radio bursts_paper_178_CME-ICME_pairs.xlsx')

# For testing purposes 
# sample = sample.head(3)

# In[]: --- 
# Create an empty table to be filled with the CME info and its estimated transit time 
final_table = []

# Finalize the output  
cols = sample.columns.insert(sample.shape[1]+1, 'PA_trans_time_hrs')
cols = cols.insert(len(cols)+1, 'PA_est_ICME_datetime')
cols = cols.insert(len(cols)+1, 'ChP_trans_time_hrs')
cols = cols.insert(len(cols)+1, 'ChP_est_ICME_datetime')

final_table = DataFrame(columns=cols)

import matplotlib
matplotlib.use('Agg')
plt.ioff()

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
    start_window = arrival_datetime - timedelta(hours=13) # hours=11.04 
    end_window = arrival_datetime + timedelta(hours=13) # hours=11.04 
    
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
    
    '''
    Change Point Detection: Dynamic Programming Search Method 
    
    '''
    # Filter the OMNI columns 
    omni_col_lst = omni_data.columns.values.tolist()
    omni_data = omni_data.filter(omni_data.columns[[1,6,7,8,9,13,25]])
    
    col_lst = omni_data.columns
    # number of subplots
    n_subpl = len(omni_data.columns) # or omni_data.shape[1] 
    # give figure a name to refer to it later 
    figname = '1-hr OMNI data: ' + str(start_datetime) +' - ' + str(end_datetime) + ' .. Change Point Detection: Dynamic Programming Search Method'
    
    fig = plt.figure(num=figname, dpi=300, figsize=(15,12))
    # define grid of nrows x ncols
    gs = fig.add_gridspec(n_subpl, 1)
    # convert the dataframe to numpy array 
    values = np.array(omni_data)
    
    # Detect changing points over the OMNI variables 
    chp_indices = []
    for i in range(n_subpl):
        points = values[:,i]
        algo = rpt.Dynp(model='l2').fit(points)
        result = algo.predict(n_bkps=2)
        _, curr_ax = rpt.display(points, result, result, num=figname)
        # position current subplot within grid
        curr_ax[0].set_position(gs[i].get_position(fig))
        curr_ax[0].set_subplotspec(gs[i])
        curr_ax[0].set_xlim([0, len(points)-1])
        curr_ax[0].set_ylabel(col_lst[i])
        # getting the timestamps of the change points
        bkps_timestamps = omni_data[col_lst[i]].iloc[[0] + result[:-1] + [-1]].index
        st = bkps_timestamps[1]
        et = bkps_timestamps[2]
        # shade the time span  
        curr_ax[0].axvspan(st, et, facecolor='#FFCC66', alpha=0.5)
        # get a sub-dataframe for the time span 
        window = omni_data['DST1800'].loc[st:et]
        # get the timestamp index at the min value within the window time span 
        min_idx = window[window.values==min(window)].index
        # plot vertical line at the index of the min value 
        curr_ax[0].axvline(min_idx.values[0], color='r', linewidth=2, linestyle='--')
        # export the required values for the other subplot 
        chp_indices.append(bkps_timestamps)
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_chp_for_CME_'+str(event_num)+'.png'))            
        
        # fig.clf()
        # plt.clf()
        # plt.cla()
        # plt.close()
        # plt.close(fig)
        
    # Plot the changing-points indices over OMNI variables 
    timestamps_lst = []
    fig2, axs = plt.subplots(omni_data.shape[1], 1, dpi=300, figsize=(15,12), sharex=True)
    
    # Bt 
    fig1_idx = 0
    chp_fig1 = chp_indices[fig1_idx]
    st1 = chp_fig1[1]
    et1 = chp_fig1[2]
    window1 = omni_data['F1800'].loc[st1:et1]
    min_idx1 = window1[window1.values==max(window1)].index
    try:
        timestamps_lst.append(min_idx1[0])
        axs[fig1_idx].axvline(min_idx1.values[0], color='r', linewidth=2, linestyle='--')
    except IndexError as er:
        print(er)
    axs[fig1_idx].plot(omni_data['F1800'])
    axs[fig1_idx].axvspan(st1, et1, facecolor='#FFCC66', alpha=0.5)
    # axs[fig1_idx].axvline(min_idx1.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig1_idx].set_ylabel(r'$B_{t}$ $(nT)$')
    
    # Bz 
    fig2_idx = 1
    chp_fig2 = chp_indices[fig2_idx]
    st2 = chp_fig2[1]
    et2 = chp_fig2[2]
    window2 = omni_data['BZ_GSE1800'].loc[st2:et2]
    min_idx2 = window2[window2.values==min(window2)].index
    try:
        timestamps_lst.append(min_idx2[0])
        axs[fig2_idx].axvline(min_idx2.values[0], color='r', linewidth=2, linestyle='--')
    except IndexError as er:
        print(er)
    axs[fig2_idx].plot(omni_data['BZ_GSE1800'])
    axs[fig2_idx].axvspan(st2, et2, facecolor='#FFCC66', alpha=0.5)
    # axs[fig2_idx].axvline(min_idx2.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig2_idx].set_ylabel(r'$B_{z}$ $(nT)$')
    
    # Temp 
    fig3_idx = 2
    axs[fig3_idx].set_yscale('log')
    chp_fig3 = chp_indices[fig3_idx]
    st3 = chp_fig3[1]
    et3 = chp_fig3[2]
    window3 = omni_data['T1800'].loc[st3:et3]
    min_idx3 = window3[window3.values==max(window3)].index
    try:
        timestamps_lst.append(min_idx3[0])
        axs[fig3_idx].axvline(min_idx3.values[0], color='r', linewidth=2, linestyle='--')
    except IndexError as er:
        print(er)
    axs[fig3_idx].plot(omni_data['T1800'])
    axs[fig3_idx].axvspan(st3, et3, facecolor='#FFCC66', alpha=0.5)
    # axs[fig3_idx].axvline(min_idx3.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig3_idx].set_ylabel(r'$T_{p}$ $(K)$')
    
    # Density 
    fig4_idx = 3
    chp_fig4 = chp_indices[fig4_idx]
    st4 = chp_fig4[1]
    et4 = chp_fig4[2]
    window4 = omni_data['N1800'].loc[st4:et4]
    min_idx4 = window4[window4.values==max(window4)].index
    try:
        timestamps_lst.append(min_idx4[0])
        axs[fig4_idx].axvline(min_idx4.values[0], color='r', linewidth=2, linestyle='--')
    except IndexError as er:
        print(er)
    axs[fig4_idx].plot(omni_data['N1800'])
    axs[fig4_idx].axvspan(st4, et4, facecolor='#FFCC66', alpha=0.5)
    # axs[fig4_idx].axvline(min_idx4.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig4_idx].set_ylabel(r'$n_{p}$ $(cm^{-3})$')
    
    # V 
    fig5_idx = 4
    chp_fig5 = chp_indices[fig5_idx]
    st5 = chp_fig5[1]
    et5 = chp_fig5[2]
    window5 = omni_data['V1800'].loc[st5:et5]
    min_idx5 = window5[window5.values==max(window5)].index
    try:
        timestamps_lst.append(min_idx5[0])
        axs[fig5_idx].axvline(min_idx5.values[0], color='r', linewidth=2, linestyle='--')
    except IndexError as er:
        print(er)
    axs[fig5_idx].plot(omni_data['V1800'])
    axs[fig5_idx].axvspan(st5, et5, facecolor='#FFCC66', alpha=0.5)
    # axs[fig5_idx].axvline(min_idx5.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig5_idx].set_ylabel('$V_{sw}$\n$(km.s^{-1})$')
    
    # P 
    fig6_idx = 5
    chp_fig6 = chp_indices[fig6_idx]
    st6 = chp_fig6[1]
    et6 = chp_fig6[2]
    window6 = omni_data['Pressure1800'].loc[st6:et6]
    min_idx6 = window6[window6.values==max(window6)].index
    try:
        timestamps_lst.append(min_idx6[0])
        axs[fig6_idx].axvline(min_idx6.values[0], color='r', linewidth=2, linestyle='--')
    except IndexError as er:
        print(er)
    axs[fig6_idx].plot(omni_data['Pressure1800'])
    axs[fig6_idx].axvspan(st6, et6, facecolor='#FFCC66', alpha=0.5)
    # axs[fig6_idx].axvline(min_idx6.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig6_idx].set_ylabel('P (nPa)')
    
    # Dst 
    fig7_idx = 6
    chp_fig7 = chp_indices[fig7_idx]
    st7 = chp_fig7[1]
    et7 = chp_fig7[2]
    window7 = omni_data['DST1800'].loc[st7:et7]
    min_idx7 = window7[window7.values==min(window7)].index
    try:
        timestamps_lst.append(min_idx7[0])
        axs[fig7_idx].axvline(min_idx7.values[0], color='r', linewidth=2, linestyle='--')
    except IndexError as er:
        print(er)
    axs[fig7_idx].plot(omni_data['DST1800'])
    axs[fig7_idx].axvspan(st7, et7, facecolor='#FFCC66', alpha=0.5)
    axs[fig7_idx].set_ylabel('Dst (nT)')
    
    # Taking the average of those timestamps 
    timestamps_lst = DataFrame(timestamps_lst, columns={'timestamps'})
    chp_ICME_est_arrival_time = Timestamp(np.nanmean([tsp.value for tsp in timestamps_lst['timestamps']]))
    
    for ax in axs:
        ax.set_xlim([omni_data.index[0], omni_data.index[-1]])
        ax.axvline(chp_ICME_est_arrival_time, color='k', linewidth=2, linestyle='--')
    
    plt.xlabel('Datetime')
    fig2.tight_layout()
    fig2.autofmt_xdate(rotation=45)
    plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_for_CME_'+str(event_num)+'.png'))
    # plt.show()
    
    # fig2.clf()
    # plt.clf()
    # plt.cla()
    # plt.close()
    plt.close(fig2)
    
    # Calculate the time delta between the CME and the ICME 
    est_tran_time_chp_method = chp_ICME_est_arrival_time - sample.CME_Datetime[event_num]
     
    print('\nReport:\n========')
    print('> CME launch datetime:', sample.CME_Datetime[event_num])
    print('> ICME estimated datetime,\nusing Dynamic Programming Search Method:', chp_ICME_est_arrival_time)
    print('> Estimated transit time is: {} days, {} hours, {} minutes' .format(est_tran_time_chp_method.components.days, 
                                                                             est_tran_time_chp_method.components.hours, 
                                                                             est_tran_time_chp_method.components.minutes))
    print('----------------------------------------------------------------')

    est_tran_time_chp_method = (est_tran_time_chp_method.components.days * 24) + (est_tran_time_chp_method.components.minutes / 60) + (est_tran_time_chp_method.components.seconds / 3600)

# ---------------------------------------------------------------------------- 
    
    ''' 
    Parametric Approach (PA) 
    
    ''' 
    # Assign a threshold of the Dst (nT) to look for geomagnetic storms 
    threshold = -40.0
    
    print('\nDefine timestamps where Dst =<', round(threshold,2), 'nT')
    print('-------------------------------------------------------')

    ''' 
    Select all the rows which satisfies the criteria, and 
    Convert the collection of the index labels of those rows to list 
    
    ''' 
    
    Index_label_Dst = omni_data[omni_data['DST1800'] <= threshold].index.tolist()
    
    if Index_label_Dst == []:
        print('No value of Dst-index =< ', threshold, ' is found\nwithin the specified time interval in OMNI data.')
        print('Skip the analysis for the CME number:', sample.index[event_num])
        print('-------------------------------------------------------')
    
    else:
        if min(omni_data['DST1800']) <= threshold:
            
            # Calculating half the expected solar wind temperature (0.5Texp) 
            ''' 
            define the Texp as mentioned in: 
                Lopez, R. E., & Freeman, J. W. (1986). 
                Solar wind proton temperature‐velocity relationship. 
                Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
                
            ''' 
            AVG = []
            
            if mean(omni_data['V1800']) > 500:
                # for the high-speed wind 
                Texp = (0.5*((0.031*omni_data['V1800']) - 4.39)**2)
            else:
                # for the high-speed wind 
                Texp = (0.5*((0.77*omni_data['V1800']) - 265)**2)
            Texp.rename('Texp', inplace=True)
            
            # Find Geomagnetic Storms in Data 
            for i in range(1, len(omni_data)):
                
                if omni_data['DST1800'][i] <= threshold:
                    
                    ''' 
                     RUN A FOR-LOOP THAT CHECK THE MIN. DIFF. BETWEEN 
                     'arrival_datetime' & 'Index_label_<ANY_SW_PARAM.>' 
                     & STORE THE 'idx' VALUES FOR EACH SW PARAM. AS A LIST 
                     FINALLY, TAKE THE AVG. OF THAT LIST, TO BE 
                     THE PROPABLE ARRIVAL TIME OF THE EVENT. 
                     
                    ''' 
                    # Find the local min Dst value within the window of 'Index_label_Dst'           
                    min_Dst_window = min(omni_data['DST1800'].loc[Index_label_Dst[0]:Index_label_Dst[-1]])
                    
                    dt_G2001_idxLabel = []
                    for idx in range(len(Index_label_Dst)):
                        dt_G2001_idxLabel.append(abs(arrival_datetime - Index_label_Dst[idx]))
                    
                    T_RED = omni_data[omni_data['DST1800']==min_Dst_window].index[0]
                    T_BLACK = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))]
                    T_GREEN = arrival_datetime
                    
                    AVG.append(Timestamp((T_RED.value + T_BLACK.value + T_GREEN.value)/3.0))

            # APPEND THE OUTPUT TRANSIT TIME WITH THE CME INFO 
            try:
                PA_est_trans_time = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))] - sample.CME_Datetime[event_num]
                PA_est_trans_time
            except IndexError as idx_err:
                print(idx_err)

            PA_tran_time_hours = (PA_est_trans_time.components.days * 24) + (PA_est_trans_time.components.minutes / 60) + (PA_est_trans_time.components.seconds / 3600)
            
            
            # BUGS FOUND:
                # PA_est_ICME_datetime: it produces several timestamps per event 
                # ICME_Datetime: missing 
                # ICME_Speed: missing 
            final_table = final_table.append({'CME_Datetime': sample.CME_Datetime[event_num], 
                                              'W': sample['W'][event_num], 
                                              'CME_Speed': sample['CME_Speed'][event_num], 
                                              'Shock_Speed': sample['Shock_Speed'][event_num], 
                                              'a': sample['a'][event_num], 
                                              'Trans_Time': sample['Trans_Time'][event_num], 
                                              'PA_trans_time_hrs': PA_tran_time_hours, 
                                              'PA_est_ICME_datetime': AVG, # PA_est_trans_time 
                                              'ChP_trans_time_hrs': est_tran_time_chp_method, 
                                              'ChP_est_ICME_datetime': chp_ICME_est_arrival_time}, 
                                             ignore_index=True)
                        
        else:
            print('The OMNI data from '+str(start_datetime)+' to '+str(end_datetime)+' has no Dst value below '+str(threshold)+' nT.')
            print('-------------------------------------------------------\n')

print('Final Report:\n===============')
print('Total number of CMEs:', len(sample))
print('Number of CMEs with Dst index =< -40 nT:', len(final_table))
print('Number of skipped CMEs during the PA:', len(sample) - len(final_table))

final_table.to_excel('Matched_List_'+str(len(final_table))+'_CME-ICME_pairs.xlsx')

# PLOT SPEED VS TRANSIT TIME 
plt.figure()
plt.scatter(final_table['CME_Speed'], final_table['PA_trans_time_hrs'], label='Model')
plt.scatter(final_table['CME_Speed'], final_table['Trans_Time'], label='Actual')
plt.legend(loc=0, frameon=False)
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel(r'$Transit$ $time$ $(hrs)$')
plt.title('Estimated CMEs transit times using parametric approach')
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'PA_V_vs_T.png'))
# plt.show()

plt.figure()
plt.scatter(final_table['CME_Speed'], final_table['ChP_trans_time_hrs'], label='Model')
plt.scatter(final_table['CME_Speed'], final_table['Trans_Time'], label='Actual')
plt.legend(loc=0, frameon=False)
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel(r'$Transit$ $time$ $(hrs)$')
plt.title('Estimated CMEs transit times using dynamic programming search method')
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'ChP_V_vs_T.png'))
# plt.show()

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
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'PA_hist_V_vs_Err.png'))
# plt.show()

plt.figure()
plt.hist2d(final_table['CME_Speed'], ChP_abs_err, bins=10)
plt.colorbar()
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel('MAPE (%)')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'ChP_hist_V_vs_Err.png'))
# plt.show()

plt.figure()
ymin, ymax = plt.ylim()
plt.hist(PA_abs_err, bins=10, alpha=0.7)
plt.axvline(PA_abs_err.mean(), color='k', linestyle='dashed', 
            linewidth=1, label='Mean = '+str(round(PA_abs_err.mean(),2))+'%')
plt.legend(loc='best', frameon=False)
plt.xlabel('MAPE (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'PA_hist_err.png'))
# plt.show()

plt.figure()
ymin, ymax = plt.ylim()
plt.hist(ChP_abs_err, bins=10, alpha=0.7)
plt.axvline(ChP_abs_err.mean(), color='k', linestyle='dashed', 
            linewidth=1, label='Mean = '+str(round(ChP_abs_err.mean(),2))+'%')
plt.legend(loc='best', frameon=False)
plt.xlabel('MAPE (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'ChP_hist_err.png'))
# plt.show()

# In[]: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

# In[]: Try the 'trendet' package 
start_datetime = datetime(2015, 6, 22)
end_datetime = datetime(2015, 6, 29)
omni_data_raw = get_omni_hr(start_datetime, end_datetime)
# In[]: Plot the Dst index 
plt.figure(figsize=(15,3))
plt.plot(omni_data_raw['DST1800'])
plt.xlabel('Datetime')
plt.ylabel('Dst (nT)')
plt.xlim(omni_data_raw.index[0], omni_data_raw.index[-1])
plt.show()
# In[]: --- 
omni_data = omni_data_raw.filter(omni_data_raw.columns[[1,6,7,8,9,13,25]])
omni_data = omni_data.astype('float64')

import trendet
import seaborn as sns
sns.set()

trend_type = 'both'

if trend_type == 'up':
    
    res = trendet.identify_df_trends(df=omni_data, column='DST1800', window_size=5, identify=trend_type)
    res.reset_index(inplace=True)
    
    fig = plt.figure(figsize=(15,3))

    ax = sns.lineplot(x=res['Time'], y=res['DST1800'])
    
    labels = res['Up Trend'].dropna().unique().tolist()
    
    for label in labels:
        sns.lineplot(x=res[res['Up Trend'] == label]['Time'],
                     y=res[res['Up Trend'] == label]['DST1800'],
                     color='green')
    
        ax.axvspan(res[res['Up Trend'] == label]['Time'].iloc[0],
                   res[res['Up Trend'] == label]['Time'].iloc[-1],
                   alpha=0.2,
                   color='green')
    
    ax.set(xlim=(omni_data['Time'].iloc[0], omni_data['Time'].iloc[-1]))
    plt.gcf().autofmt_xdate()
    plt.show()


elif trend_type == 'down':
    
    res = trendet.identify_df_trends(df=omni_data, column='DST1800', window_size=5, identify=trend_type)
    res.reset_index(inplace=True)
    
    fig = plt.figure(figsize=(15,3))

    ax = sns.lineplot(x=res['Time'], y=res['DST1800'])
    
    labels = res['Down Trend'].dropna().unique().tolist()
    
    for label in labels:
        sns.lineplot(x=res[res['Down Trend'] == label]['Time'],
                      y=res[res['Down Trend'] == label]['DST1800'],
                      color='red')
    
        ax.axvspan(res[res['Down Trend'] == label]['Time'].iloc[0],
                    res[res['Down Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2,
                    color='red')
    
    ax.set(xlim=(omni_data['Time'].iloc[0], omni_data['Time'].iloc[-1]))
    plt.gcf().autofmt_xdate()
    plt.show()

elif trend_type == 'both':

    res = trendet.identify_df_trends(df=omni_data, column='DST1800', window_size=5, identify=trend_type)
    res.reset_index(inplace=True)
    
    fig = plt.figure(figsize=(15,3))

    ax = sns.lineplot(x=res['Time'], y=res['DST1800'])
    
    labels = res['Up Trend'].dropna().unique().tolist()
    
    for label in labels:
        sns.lineplot(x=res[res['Up Trend'] == label]['Time'],
                     y=res[res['Up Trend'] == label]['DST1800'],
                     color='green')
    
        ax.axvspan(res[res['Up Trend'] == label]['Time'].iloc[0],
                   res[res['Up Trend'] == label]['Time'].iloc[-1],
                   alpha=0.2,
                   color='green')
    
    labels = res['Down Trend'].dropna().unique().tolist()
    
    for label in labels:
        sns.lineplot(x=res[res['Down Trend'] == label]['Time'],
                      y=res[res['Down Trend'] == label]['DST1800'],
                      color='red')
    
        ax.axvspan(res[res['Down Trend'] == label]['Time'].iloc[0],
                    res[res['Down Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2,
                    color='red')
    
    ax.set(xlim=(omni_data['Time'].iloc[0], omni_data['Time'].iloc[-1]))
    ax.set_ylabel('Dst (nT)')
    ax.set_xlabel('Date')
    fig.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

# In[]: Subplot for comparison between up and down trends 
omni_data = omni_data_raw.filter(omni_data_raw.columns[[1,6,7,8,9,13,25]])
omni_data = omni_data.astype('float64')

sns.set()
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15,7))
fig.suptitle('Comparison between trends for the Dst index')

res = trendet.identify_df_trends(df=omni_data, column='DST1800', window_size=3, identify='both')
res.reset_index(inplace=True)

# Up trend 
sns.lineplot(ax=axes[0], x=res['Time'], y=res['DST1800'])
labels = res['Up Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[0], x=res[res['Up Trend'] == label]['Time'],
                 y=res[res['Up Trend'] == label]['DST1800'], color='green')
    axes[0].axvspan(res[res['Up Trend'] == label]['Time'].iloc[0],
               res[res['Up Trend'] == label]['Time'].iloc[-1],
               alpha=0.2, color='green')
axes[0].set(xlim=(omni_data['Time'].iloc[0], omni_data['Time'].iloc[-1]))
axes[0].set_title('Up Trend')
axes[0].set_ylabel('Dst (nT)')

# Down trend 
sns.lineplot(ax=axes[1], x=res['Time'], y=res['DST1800'])
labels = res['Down Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[1], x=res[res['Down Trend'] == label]['Time'],
                  y=res[res['Down Trend'] == label]['DST1800'], color='red')
    axes[1].axvspan(res[res['Down Trend'] == label]['Time'].iloc[0],
                res[res['Down Trend'] == label]['Time'].iloc[-1],
                alpha=0.2, color='red')
axes[1].set(xlim=(omni_data['Time'].iloc[0], omni_data['Time'].iloc[-1]))
axes[1].set_title('Down Trend')
axes[1].set_ylabel('Dst (nT)')

# Both trends 
sns.lineplot(ax=axes[2], x=res['Time'], y=res['DST1800'])
labels = res['Up Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[2], x=res[res['Up Trend'] == label]['Time'],
                 y=res[res['Up Trend'] == label]['DST1800'], color='green')
    axes[2].axvspan(res[res['Up Trend'] == label]['Time'].iloc[0],
               res[res['Up Trend'] == label]['Time'].iloc[-1],
               alpha=0.2, color='green')
    
labels = res['Down Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[2], x=res[res['Down Trend'] == label]['Time'],
                  y=res[res['Down Trend'] == label]['DST1800'], color='red')
    axes[2].axvspan(res[res['Down Trend'] == label]['Time'].iloc[0],
                res[res['Down Trend'] == label]['Time'].iloc[-1],
                alpha=0.2, color='red')
    
axes[2].set(xlim=(omni_data['Time'].iloc[0], omni_data['Time'].iloc[-1]))
axes[2].set_title('Both Trends')
axes[2].set_ylabel('Dst (nT)')
axes[2].set_xlabel('Date')
fig.tight_layout()
plt.show()

# In[]: Subplot for V, n, T, Dst
sns.set(style='darkgrid')
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15,7))
fig.suptitle('Comparison between trends for OMNI hourly data')

# Plasma speed 
omni_data = omni_data_raw.filter(omni_data_raw.columns[[1,6,7,8,9,13,25]]).astype('float64')
v = trendet.identify_df_trends(df=omni_data, column='V1800', window_size=5, identify='both')
v.reset_index(inplace=True)
sns.lineplot(ax=axes[0], x=v['Time'], y=v['V1800'])
labels = v['Up Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[0], x=v[v['Up Trend'] == label]['Time'], 
                 y=v[v['Up Trend'] == label]['V1800'], color='green')
    axes[0].axvspan(v[v['Up Trend'] == label]['Time'].iloc[0], 
                    v[v['Up Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='green')   
labels = v['Down Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[0], x=v[v['Down Trend'] == label]['Time'], 
                 y=v[v['Down Trend'] == label]['V1800'], color='red')
    axes[0].axvspan(v[v['Down Trend'] == label]['Time'].iloc[0], 
                    v[v['Down Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='red')
axes[0].set(xlim=(v['Time'].iloc[0], v['Time'].iloc[-1]))
axes[0].set_ylabel('V (km/s)')

# Plasma density 
omni_data = omni_data_raw.filter(omni_data_raw.columns[[1,6,7,8,9,13,25]]).astype('float64')
n = trendet.identify_df_trends(df=omni_data, column='N1800', window_size=3, identify='both')
n.reset_index(inplace=True)
sns.lineplot(ax=axes[1], x=n['Time'], y=n['N1800'])
labels = n['Up Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[1], x=n[n['Up Trend'] == label]['Time'], 
                 y=n[n['Up Trend'] == label]['N1800'], color='green')
    axes[1].axvspan(n[n['Up Trend'] == label]['Time'].iloc[0], 
                    n[n['Up Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='green')   
labels = n['Down Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[1], x=n[n['Down Trend'] == label]['Time'], 
                 y=n[n['Down Trend'] == label]['N1800'], color='red')
    axes[1].axvspan(n[n['Down Trend'] == label]['Time'].iloc[0], 
                    n[n['Down Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='red')
axes[1].set(xlim=(n['Time'].iloc[0], n['Time'].iloc[-1]))
axes[1].set_ylabel('n (#/cm3)')

# Plasma temperature 
omni_data = omni_data_raw.filter(omni_data_raw.columns[[1,6,7,8,9,13,25]]).astype('float64')
T = trendet.identify_df_trends(df=omni_data, column='T1800', window_size=5, identify='both')
T.reset_index(inplace=True)
sns.lineplot(ax=axes[2], x=T['Time'], y=T['T1800'])
labels = T['Up Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[2], x=T[T['Up Trend'] == label]['Time'], 
                 y=T[T['Up Trend'] == label]['T1800'], color='green')
    axes[2].axvspan(T[T['Up Trend'] == label]['Time'].iloc[0], 
                    T[T['Up Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='green')   
labels = T['Down Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[2], x=T[T['Down Trend'] == label]['Time'], 
                 y=T[T['Down Trend'] == label]['T1800'], color='red')
    axes[2].axvspan(T[T['Down Trend'] == label]['Time'].iloc[0], 
                    T[T['Down Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='red')
axes[2].set(xlim=(T['Time'].iloc[0], T['Time'].iloc[-1]))
axes[2].set_yscale('log')
axes[2].set_ylabel('T (K)')

# Dst 
omni_data = omni_data_raw.filter(omni_data_raw.columns[[1,6,7,8,9,13,25]]).astype('float64')
dst = trendet.identify_df_trends(df=omni_data, column='DST1800', window_size=3, identify='both')
dst.reset_index(inplace=True)
sns.lineplot(ax=axes[3], x=dst['Time'], y=dst['DST1800'])
labels = dst['Up Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[3], x=dst[dst['Up Trend'] == label]['Time'], 
                 y=dst[dst['Up Trend'] == label]['DST1800'], color='green')
    axes[3].axvspan(dst[dst['Up Trend'] == label]['Time'].iloc[0], 
                    dst[dst['Up Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='green')   
labels = dst['Down Trend'].dropna().unique().tolist()
for label in labels:
    sns.lineplot(ax=axes[3], x=dst[dst['Down Trend'] == label]['Time'], 
                 y=dst[dst['Down Trend'] == label]['DST1800'], color='red')
    axes[3].axvspan(dst[dst['Down Trend'] == label]['Time'].iloc[0], 
                    dst[dst['Down Trend'] == label]['Time'].iloc[-1],
                    alpha=0.2, color='red')
axes[3].set(xlim=(dst['Time'].iloc[0], dst['Time'].iloc[-1]))
axes[3].set_ylabel('Dst (nT)')

# NEXT: FIND THE TIME INDEX AT WHICH THE D-TREND OF DST AND N = U-TREND OF V AND T 
omni_trends = DataFrame({'Datetime': omni_data['Time'], 
                         'V': omni_data['V1800'], 
                         'n': omni_data['N1800'], 
                         'T': omni_data['T1800'], 
                         'Dst': omni_data['DST1800'], 
                         'U-trend_V': v['Up Trend'], 
                         'D-trend_V': v['Down Trend'], 
                         'U-trend_n': n['Up Trend'], 
                         'D-trend_n': n['Down Trend'], 
                         'U-trend_T': T['Up Trend'], 
                         'D-trend_T': T['Down Trend'], 
                         'U-trend_Dst': dst['Up Trend'], 
                         'D-trend_Dst': dst['Down Trend']})

omni_trends = omni_trends.set_index('Datetime')

# ------------------------------------------------------ 
V_n = omni_trends[['U-trend_V', 'D-trend_n']]
V_n.dropna(inplace=True)
# ------------------------------------------------------ 
V_T = omni_trends[['U-trend_V', 'U-trend_T']]
V_T.dropna(inplace=True)
# ------------------------------------------------------ 
n_T = omni_trends[['D-trend_n', 'U-trend_T']]
n_T.dropna(inplace=True)
# ------------------------------------------------------ 
V_Dst = omni_trends[['U-trend_V', 'U-trend_Dst']]
V_Dst.dropna(inplace=True)
# ------------------------------------------------------ 
n_Dst = omni_trends[['D-trend_n', 'U-trend_Dst']]
n_Dst.dropna(inplace=True)
# ------------------------------------------------------ 
T_Dst = omni_trends[['U-trend_T', 'U-trend_Dst']]
T_Dst.dropna(inplace=True)
# ------------------------------------------------------ 

# Find the intersection timestamp 
intersection = set(set(set(set(set(V_n.index).intersection(V_T.index)).intersection(n_T.index)).intersection(V_Dst.index)).intersection(n_Dst.index)).intersection(T_Dst.index)
intersection = [*intersection,]

# Plot a dashed black line representing the estimated CME arrival time 
for ax in axes:
    ax.axvline(intersection[-1], color='k', linewidth=2, linestyle='--')

fig.tight_layout()
plt.show()

# In[]: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

















