# -*- coding: utf-8 -*-
"""
Plotting OMNI Data
==================

Importing and plotting data from OMNI Web Interface.
OMNI provides interspersed data from various spacecrafts.

"""
# import numpy as np
from pandas import read_excel, DataFrame, Timestamp, to_datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os.path
from necessary_functions import get_omni, G2001
from statistics import mean
# Print out the data directory path and their sizes 
from heliopy.data import helper as heliohelper
heliohelper.listdata()
import warnings
warnings.filterwarnings('ignore')

# In[]: Establishing the output folder 
save_path = 'D:/Study/Academic/Research/Master Degree/Master Work/Software/Codes/Python/Heliopy Examples/auto_examples_python/'
try:
    os.mkdir(save_path + 'Output_plots')
except OSError as error:
    print(error)

# In[]: IMPORT THE LIST OF CME EVENTS 
date_columns = ['Shock_Date', 'Shock_Time', 'ICME_Time', 'CME_Date', 'CME_Time']
sample = read_excel('List_from_Interplanetary shocks lacking type II radio bursts_paper.xlsx', 
                    sheet_name='Sheet1', parse_dates=date_columns)

# formatting the datetimes 
for i in range(len(sample)):
    try:
        # Remove the '*' character in 'ICME Time' column 
        if sample['ICME_Time'][i][0] == '*':
            sample['ICME_Time'][i] = sample['ICME_Time'][i][1:]
    except:
        pass

sample['Shock_Date'] = to_datetime(sample['Shock_Date'])
sample['Shock_Time'] = to_datetime(sample['Shock_Time'])
sample['ICME_Time'] = to_datetime(sample['ICME_Time'])
sample['CME_Date'] = to_datetime(sample['CME_Date'])
sample['CME_Time'] = to_datetime(sample['CME_Time'])

new_cme_datetime = []
new_icme_datetime = []
for i in range(len(sample)):
    new_cme_datetime.append(datetime.combine(sample['CME_Date'][i].date(), sample['CME_Time'][i].time()))
    new_icme_datetime.append(datetime.combine(sample['Shock_Date'][i].date(), sample['ICME_Time'][i].time()))

sample.insert(0, 'ICME_Datetime', new_icme_datetime)
sample.insert(1, 'CME_Datetime', new_cme_datetime)

trans_time_timedelta = []
trans_time_hours = []
for i in range(len(sample)):
    trans_time_timedelta.append(new_icme_datetime[i] - new_cme_datetime[i])
    trans_time_hours.append(round((trans_time_timedelta[i].days*24) + (trans_time_timedelta[i].seconds/3600), 2))

sample.insert(sample.shape[1], 'Trans_Time', trans_time_hours)
    
for col in sample.columns:
    try:
        if col in sample.columns:
            sample = sample.drop(columns={'Shock_Date',
                                          'Shock_Time',
                                          'ICME_Time',
                                          'CME_Date',
                                          'CME_Time'})
    except KeyError as err:
        print(err)

sample.to_excel('List_from_Interplanetary shocks lacking type II radio bursts_paper_'+str(len(sample))+'_CME-ICME_pairs.xlsx')

# In[]: Create an empty table to be filled with the CME info and its estimated transit time 
final_table = []

print('\nFinding the ICMEs in OMNI data and match them with the CMEs from SOHO-LASCO')
print('For more info about the method, check the paper:\nGopalswamy et al. (2010), Interplanetary shocks lacking type II radio bursts,\nThe Astrophysical Journal, 710(2), 1111.')

print('\nPredicting the transit time of the CMEs using the G2001 model\nwith mean error of:', 11.04, 'hours, according to Gopalswamy et al. 2001 .. \n')

print('For more info about the G2001 model, check this paper:\nGopalswamy, N., Lara, A., Yashiro, S., Kaiser, M. L., and Howard,\nR. A.: Predicting the 1-AU arrival times of coronal mass ejections,\nJ. Geophys. Res., 106, 29 207, 2001a.\n')
print('And this paper:\nOwens, M., & Cargill, P. (2004, January).\nPredictions of the arrival time of Coronal Mass Ejections at 1AU:\nan analysis of the causes of errors.\nIn Annales Geophysicae (Vol. 22, No. 2, pp. 661-671). Copernicus GmbH.\n')
print('=========================================================')

# In[]: --- 
# FINALIZE THE OUTPUT 
cols = sample.columns

cols = cols.insert(0, 'CME_datetime')
cols = cols.insert(len(cols)+1, 'Transit_time_hrs')
cols = cols.insert(len(cols)+1, 'est_ICME_datetime')

final_table = DataFrame(columns=cols)

# Assign a threshold of the Dst (nT) to look for geomagnetic storms 
threshold = -40.0
print('Define timestamps where Dst =<', round(threshold,2), 'nT')
print('-------------------------------------------------------')

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
    start_window = arrival_datetime - timedelta(hours=11.04)
    end_window = arrival_datetime + timedelta(hours=11.04)
    
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
    
    omni_data = get_omni(start_datetime, end_datetime)
    
    # ------------------------------------------------------ 
    
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
            if mean(omni_data['V1800']) >= 500:
                Texp = 0.5 * (((0.031*omni_data['V1800']) - 5.1)**2) * (10**3)
            else:
                Texp = ((0.0106*omni_data['V1800']) - 0.278)**2
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
                    
                    AVG = Timestamp((T_RED.value + T_BLACK.value + T_GREEN.value)/3.0)

            # APPEND THE OUTPUT TRANSIT TIME WITH THE CME INFO 
            try:
                est_trans_time = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))] - sample.CME_Datetime[event_num]
            except IndexError as idx_err:
                print(idx_err)

            tran_time_hours = (est_trans_time.components.days * 24) + (est_trans_time.components.minutes / 60) + (est_trans_time.components.seconds / 3600)

            final_table = final_table.append({'CME_Datetime': sample.CME_Datetime[event_num], 
                                              'W': sample['W'][event_num], 
                                              'CME_Speed': sample['CME_Speed'][event_num], 
                                              'a': sample['a'][event_num], 
                                              'Model_trans_time_hrs': tran_time_hours, 
                                              'est_ICME_datetime': est_trans_time, 
                                              'Actual_Trans_Time': sample['Trans_Time'][event_num]}, 
                                             ignore_index=True)
                        
        else:
            print('The OMNI data from '+str(start_datetime)+' to '+str(end_datetime)+' has no Dst value below '+str(threshold)+' nT.')
            print('-------------------------------------------------------\n')

print('Final Report:\n===============')
print('Total number of CMEs:', len(sample))
print('Number of CMEs with Dst index =< -40 nT:', len(final_table))
print('Number of skipped CMEs:', len(sample) - len(final_table))


try:
    final_table = final_table.drop(columns={'CME_datetime',
                                            'ICME_Datetime',
                                            'Shock_Speed',
                                            'ICME_Speed',
                                            'Trans_Time', 
                                            'Transit_time_hrs'})
except KeyError as err:
    print(err)

final_table.to_excel('Matched_List_'+str(len(final_table))+'_CME-ICME_pairs.xlsx')

# In[]: PLOT SPEED VS TRANSIT TIME 
plt.figure()
plt.scatter(final_table['CME_Speed'], final_table['Model_trans_time_hrs'], label='Model')
plt.scatter(final_table['CME_Speed'], final_table['Actual_Trans_Time'], label='Actual')
plt.legend(loc='best', frameon=False)
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel(r'$Transit$ $time$ $(hrs)$')
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'V_vs_T.png'))

# In[]: Calculate the Error 

# Calculation of Mean Squared Error (MSE) 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(final_table['Actual_Trans_Time'], final_table['Model_trans_time_hrs'])

# Calculation of absolute percentage error 
abs_err = abs((final_table['Actual_Trans_Time']-final_table['Model_trans_time_hrs'])/final_table['Actual_Trans_Time']) * 100

# In[]: Distribution of Error 
plt.figure()
plt.hist2d(final_table['CME_Speed'], abs_err, bins=10)
plt.colorbar()
plt.xlabel(r'$V_{CME}$ $(km.s^{-1})$')
plt.ylabel('Abs. Error (%)')
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'hist_V_vs_Err.png'))

plt.figure()
ymin, ymax = plt.ylim()
plt.hist(abs_err, bins=10, alpha=0.7)
plt.axvline(abs_err.mean(), color='k', linestyle='dashed', 
            linewidth=1, label='Mean = '+str(round(abs_err.mean(),2))+'%')
plt.legend(loc='best', frameon=False)
plt.xlabel('Abs. Error (%)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'hist_err.png'))


