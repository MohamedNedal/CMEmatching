# -*- coding: utf-8 -*-
"""
The Dynamic Programming-based Segmentation (DPS) Method 
    Change Point (ChP) Detection: Dynamic Programming Search Method 
Created on Sat Apr  3 18:50:31 2021
@author: Mohamed Nedal 
""" 
from os.path import join
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame, Timestamp

def ChP(omni_data, sample, event_num):
    '''
    Parameters
    ----------
    omni_data : DataFrame table 
        Hourly-OMNI data. 
    sample : DataFrame table 
        List of CME-ICME pairs from the paper "Interplanetary shocks lacking type II radio bursts". 
    event_num : Integer number 
        CME index in the SOHO/LASCO catalog. 

    Returns
    ------- 
    chp_ICME_est_arrival_time : Timestamp 
        Estimated ICME arrival time. 
    est_tran_time_chp_method : Float number 
        Estimated transit time of the CME in hours. 
    ''' 
    
    # Filter the OMNI columns 
    # omni_data = omni_data.filter(omni_data.columns[[1,6,7,8,9,13,25]])
    omni_data = omni_data.filter(omni_data[['F1800', 'BZ_GSE1800', 
                                            'T1800', 'N1800', 'V1800', 
                                            'Pressure1800', 'DST1800']])
    col_lst = omni_data.columns
    
    # number of subplots
    n_subpl = len(omni_data.columns) # or omni_data.shape[1] 
    
    # give figure a name to refer to it later 
    figname = '1-hr OMNI data'
    
    fig = plt.figure(num=figname, figsize=(15,12))
    
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
        # plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_chp_for_CME_'+str(event_num)+'.png'), dpi=300)
        plt.show()
        
    # Plot the changing-points indices over OMNI variables 
    timestamps_lst = []
    fig2, axs = plt.subplots(omni_data.shape[1], 1, figsize=(17,10), sharex=True)
    
    # Bt 
    fig1_idx = 0
    chp_fig1 = chp_indices[fig1_idx]
    st1 = chp_fig1[1]
    et1 = chp_fig1[2]
    window1 = omni_data['F1800'].loc[st1:et1]
    min_idx1 = window1[window1.values==max(window1)].index
    try:
        timestamps_lst.append(min_idx1[0])
        axs[fig1_idx].axvline(min_idx1.values[0], color='r', linewidth=2, linestyle='--', label='ChP')
    except IndexError as er:
        print(er)
    axs[fig1_idx].plot(omni_data['F1800'])
    axs[fig1_idx].axvspan(st1, et1, facecolor='#FFCC66', alpha=0.5)
    # axs[fig1_idx].axvline(min_idx1.values[0], color='r', linewidth=2, linestyle='--')
    # axs[fig1_idx].set_xlabel(fontsize=16)
    axs[fig1_idx].set_ylabel(r'$B_{t}$ $(nT)$', fontsize=14)
    axs[fig1_idx].tick_params(labelsize=14)
    
    
    # Bz 
    fig2_idx = 1
    chp_fig2 = chp_indices[fig2_idx]
    st2 = chp_fig2[1]
    et2 = chp_fig2[2]
    window2 = omni_data['BZ_GSE1800'].loc[st2:et2]
    min_idx2 = window2[window2.values==min(window2)].index
    try:
        timestamps_lst.append(min_idx2[0])
        axs[fig2_idx].axvline(min_idx2.values[0], color='r', linewidth=2, linestyle='--', label='ChP')
    except IndexError as er:
        print(er)
    axs[fig2_idx].plot(omni_data['BZ_GSE1800'])
    axs[fig2_idx].axvspan(st2, et2, facecolor='#FFCC66', alpha=0.5)
    # axs[fig2_idx].axvline(min_idx2.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig2_idx].set_ylabel(r'$B_{z}$ $(nT)$', fontsize=14)
    axs[fig2_idx].tick_params(labelsize=14)
    
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
        axs[fig3_idx].axvline(min_idx3.values[0], color='r', linewidth=2, linestyle='--', label='ChP')
    except IndexError as er:
        print(er)
    axs[fig3_idx].plot(omni_data['T1800'])
    axs[fig3_idx].axvspan(st3, et3, facecolor='#FFCC66', alpha=0.5)
    # axs[fig3_idx].axvline(min_idx3.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig3_idx].set_ylabel(r'$T_{p}$ $(K)$', fontsize=14)
    axs[fig3_idx].tick_params(labelsize=14)
    
    # Density 
    fig4_idx = 3
    chp_fig4 = chp_indices[fig4_idx]
    st4 = chp_fig4[1]
    et4 = chp_fig4[2]
    window4 = omni_data['N1800'].loc[st4:et4]
    min_idx4 = window4[window4.values==max(window4)].index
    try:
        timestamps_lst.append(min_idx4[0])
        axs[fig4_idx].axvline(min_idx4.values[0], color='r', linewidth=2, linestyle='--', label='ChP')
    except IndexError as er:
        print(er)
    axs[fig4_idx].plot(omni_data['N1800'])
    axs[fig4_idx].axvspan(st4, et4, facecolor='#FFCC66', alpha=0.5)
    # axs[fig4_idx].axvline(min_idx4.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig4_idx].set_ylabel(r'$n_{p}$ $(cm^{-3})$', fontsize=14)
    axs[fig4_idx].tick_params(labelsize=14)
    
    # V 
    fig5_idx = 4
    chp_fig5 = chp_indices[fig5_idx]
    st5 = chp_fig5[1]
    et5 = chp_fig5[2]
    window5 = omni_data['V1800'].loc[st5:et5]
    min_idx5 = window5[window5.values==max(window5)].index
    try:
        timestamps_lst.append(min_idx5[0])
        axs[fig5_idx].axvline(min_idx5.values[0], color='r', linewidth=2, linestyle='--', label='ChP')
    except IndexError as er:
        print(er)
    axs[fig5_idx].plot(omni_data['V1800'])
    axs[fig5_idx].axvspan(st5, et5, facecolor='#FFCC66', alpha=0.5)
    # axs[fig5_idx].axvline(min_idx5.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig5_idx].set_ylabel('$V_{sw}$\n$(km.s^{-1})$', fontsize=14)
    axs[fig5_idx].tick_params(labelsize=14)
    
    # P 
    fig6_idx = 5
    chp_fig6 = chp_indices[fig6_idx]
    st6 = chp_fig6[1]
    et6 = chp_fig6[2]
    window6 = omni_data['Pressure1800'].loc[st6:et6]
    min_idx6 = window6[window6.values==max(window6)].index
    try:
        timestamps_lst.append(min_idx6[0])
        axs[fig6_idx].axvline(min_idx6.values[0], color='r', linewidth=2, linestyle='--', label='ChP')
    except IndexError as er:
        print(er)
    axs[fig6_idx].plot(omni_data['Pressure1800'])
    axs[fig6_idx].axvspan(st6, et6, facecolor='#FFCC66', alpha=0.5)
    # axs[fig6_idx].axvline(min_idx6.values[0], color='r', linewidth=2, linestyle='--')
    axs[fig6_idx].set_ylabel('P (nPa)', fontsize=14)
    axs[fig6_idx].tick_params(labelsize=14)
    
    # Dst 
    fig7_idx = 6
    chp_fig7 = chp_indices[fig7_idx]
    st7 = chp_fig7[1]
    et7 = chp_fig7[2]
    window7 = omni_data['DST1800'].loc[st7:et7]
    min_idx7 = window7[window7.values==min(window7)].index
    try:
        timestamps_lst.append(min_idx7[0])
        axs[fig7_idx].axvline(min_idx7.values[0], color='r', linewidth=2, linestyle='--', label='ChP')
    except IndexError as er:
        print(er)
    axs[fig7_idx].plot(omni_data['DST1800'])
    axs[fig7_idx].axvspan(st7, et7, facecolor='#FFCC66', alpha=0.5)
    axs[fig7_idx].set_ylabel('Dst (nT)', fontsize=14)
    axs[fig7_idx].tick_params(labelsize=14)
    
    # Taking the average of those timestamps 
    timestamps_lst = DataFrame(timestamps_lst, columns={'timestamps'})
    chp_ICME_est_arrival_time = Timestamp(np.nanmean([tsp.value for tsp in timestamps_lst['timestamps']]))
    
    for ax in axs:
        ax.set_xlim([omni_data.index[0], omni_data.index[-1]])
        ax.axvline(chp_ICME_est_arrival_time, color='k', linewidth=2, linestyle='--', label='TT')
        ax.legend(loc='upper right', frameon=False, prop={'size': 14})
    
    axs[fig7_idx].set_xlabel('Date', fontsize=14)
    axs[fig7_idx].tick_params(labelsize=14)
    
    fig2.tight_layout()
    axs[fig7_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H:%M'))
    fig2.autofmt_xdate(rotation=45)
    save_path = 'D:/Study/Academic/Research/Master Degree/Master Work/Software/Codes/Python/Heliopy Examples/auto_examples_python/'
    plt.savefig(join(save_path, 'OMNI_for_CME_'+str(event_num)+'.png'), dpi=300)
    plt.show()
    
    # plt.close(fig2)
    
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
    
    return chp_ICME_est_arrival_time, est_tran_time_chp_method