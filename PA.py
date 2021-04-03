# -*- coding: utf-8 -*-
"""
The Parametric Approach (PA) 
Created on Sat Apr  3 18:53:39 2021
@author: Mohamed Nedal
"""
import numpy as np
from statistics import mean
from pandas import DataFrame, Timestamp

def PA(omni_data, sample, event_num, arrival_datetime):
    '''
    Parameters
    ----------
    omni_data : DataFrame table 
        Hourly-OMNI data. 
    sample : DataFrame table 
        List of CME-ICME pairs from the paper "Interplanetary shocks lacking type II radio bursts". 
    event_num : Integer number 
        CME index in the SOHO/LASCO catalog. 
    arrival_datetime : Datetime object 
        Estimated arrival time using G2001 model. 

    Returns
    ------- 
    PA_datetime : Timestamp 
        Average estimated ICME arrival time. 
    PA_tran_time_hours : Float number 
        Estimated transit time of the CME in hours. 
    ''' 

    # Assign a threshold of the Dst (nT) to look for geomagnetic storms 
    threshold = -35.0
    
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
        # print('Skip the analysis for the CME number:', sample)
        print('-------------------------------------------------------')
        PA_datetime, PA_tran_time_hours = [], []
    
    else:
        if min(omni_data['DST1800']) <= threshold:
            
            # Calculating half the expected solar wind temperature (0.5Texp) 
            ''' 
            define the Texp as mentioned in: 
                Lopez, R. E., & Freeman, J. W. (1986). 
                Solar wind proton temperature‐velocity relationship. 
                Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
            ''' 
            PA_datetime_lst = []
            
            if mean(omni_data['V1800']) > 500:
                # for the high-speed wind 
                Texp = ((0.031*omni_data['V1800']) - 4.39)**2
            else:
                # for the low-speed wind 
                Texp = ((0.77*omni_data['V1800']) - 265)**2
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
                    
                    PA_datetime_lst.append(Timestamp((T_RED.value + T_BLACK.value + T_GREEN.value)/3.0))

            # APPEND THE OUTPUT TRANSIT TIME WITH THE CME INFO 
            try: PA_est_trans_time = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))] - sample.CME_Datetime[event_num]
            except IndexError as idx_err: print(idx_err)
            
            PA_datetime_lst = DataFrame(PA_datetime_lst, columns={'timestamps'})
            
            # Taking the average of those timestamps 
            PA_datetime = Timestamp(np.nanmean([tsp.value for tsp in PA_datetime_lst['timestamps']]))
            PA_tran_time_hours = (PA_est_trans_time.components.days * 24) + (PA_est_trans_time.components.minutes / 60) + (PA_est_trans_time.components.seconds / 3600)
    
    return PA_datetime, PA_tran_time_hours
