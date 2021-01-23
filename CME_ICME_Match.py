# -*- coding: utf-8 -*-
"""
Plotting OMNI Data
==================

Importing and plotting data from OMNI Web Interface.
OMNI provides interspersed data from various spacecrafts.

"""
import numpy as np
from pandas import read_excel, DataFrame, Timestamp, to_datetime, date_range
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os.path
from necessary_functions import get_omni_hr, get_omni_min, G2001
from statistics import mean
# Print out the data directory path and their sizes 
from heliopy.data import helper as heliohelper
import ruptures as rpt
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
    
    omni_data = get_omni_hr(start_datetime, end_datetime)
    
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
            ''' 
            define the Texp as mentioned in: 
                Lopez, R. E., & Freeman, J. W. (1986). 
                Solar wind proton temperature‐velocity relationship. 
                Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
                
            ''' 
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

# In[]: --- TEST --- 

start_datetime = datetime(2003, 10, 25, 0, 0, 0)
end_datetime = datetime(2003, 11, 3, 0, 0, 0)

omni_data = get_omni_hr(start_datetime, end_datetime)

''' 
define the Texp as mentioned in: 
    Lopez, R. E., & Freeman, J. W. (1986). 
    Solar wind proton temperature‐velocity relationship. 
    Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
    
''' 
    
if mean(omni_data['V1800']) > 500:
    # for the high-speed wind 
    # Texp = ((0.0106*omni_data['V1800']) - 0.278)**2
    Texp = (0.5*((0.031*omni_data['V1800']) - 4.39)**2)
    
else:
    # for the high-speed wind 
    # Texp = 0.5 * (((0.031*omni_data['V1800']) - 5.1)**2) * (10**3)
    Texp = (0.5*((0.77*omni_data['V1800']) - 265)**2)

Texp.rename('Texp', inplace=True)

# In[]: --- 
x = Texp.index
y1 = omni_data['T1800'].values
y2 = Texp.values

fig, ax = plt.subplots(figsize=(15,5))

ax.plot(x, y1, label='$T_P$')
ax.plot(x, y2, label=r'$0.5T_{exp}$')

ax.fill_between(x, y1, y2, where=(y2 > y1), facecolor='red', alpha=0.5)

# to get the indices of the intersection points between both curves 
idx = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
ax.plot(x[idx], y2[idx], 'ro')
for i in idx:
    ax.axvline(omni_data['T1800'].index[i], color='green', linewidth=1, linestyle='--')

ax.legend(loc='best', frameon=False)
ax.set_xlabel('Datetime')
ax.set_ylabel('T (K)')
ax.set_xlim(Texp.index[0], Texp.index[-1])
plt.show()

# In[]: --- 
fig, axs = plt.subplots(2, figsize=(15,5))

axs[0].plot(x, y1, label='$T_P$')
axs[0].plot(x, y2, label=r'$0.5T_{exp}$')

axs[0].fill_between(x, y1, y2, where=(y2 > y1), facecolor='red', alpha=0.5)

# to get the indices of ICME boundaries 
for i in range(len(y1)):
    if y1[i] < y2[i]:
        lines = ax.axvline(omni_data['T1800'].index[i], color='k', linewidth=1, linestyle='--')

# deine the legend outside the loop to avoid showing multiple legends within the for-loop 
lines.set_label('$T_{P}<T_{exp}$')
axs[0].set_ylabel('T (K)')

axs[1].plot(omni_data['Ratio1800'])

axs[1].set_xlabel('Datetime')
axs[1].set_ylabel(r'$\dfrac{N_{\alpha}}{N_{P}}$')
    
for ax in axs:
    ax.legend(loc='best', frameon=False)
    ax.set_xlim(Texp.index[52], Texp.index[-90])

plt.show()

# In[]: --- FOR PLOTTING 1-MIN OMNI DATA --- 
start_datetime = datetime(2017, 9, 6, 0, 0, 0)
end_datetime = datetime(2017, 9, 17, 0, 0, 0)

omni_data = get_omni_min(start_datetime, end_datetime)

fig, axs = plt.subplots(7, 1, figsize=(15,10), sharex=True)
fig.suptitle('1-min OMNI data: ' + str(start_datetime) +' - ' + str(end_datetime))

axs[0].plot(omni_data['F'], label='$<|B_{T}|>$')
axs[0].set_ylabel('B (nT)')

axs[1].plot(omni_data['BX_GSE'], label='Bx GSE')
axs[1].plot(omni_data['BY_GSM'], label='By GSM')
axs[1].plot(omni_data['BZ_GSM'], label='Bz GSM')
axs[1].set_ylabel('B (nT)')

axs[2].plot(omni_data['flow_speed'])
axs[2].set_ylabel(r'$Flow\;Speed$'
                  '\n'
                  r'$(km.s^{-1})$')

axs[3].plot(omni_data['T'], label='T')
axs[3].set_ylabel('Temperature\n(K)')

if mean(omni_data['flow_speed']) > 500:
    # for the high-speed wind 
    Texp = (0.5*((0.031*omni_data['flow_speed']) - 4.39)**2)
else:
    # for the high-speed wind 
    Texp = (0.5*((0.77*omni_data['flow_speed']) - 265)**2)

Texp.rename('Texp', inplace=True)

axs[3].plot(Texp, label=r'$0.5T_{exp}$')
axs[3].set_yscale('log')

axs[4].plot(omni_data['NaNp_Ratio'])
axs[4].set_ylabel('$\dfrac{N_{alpha}}{N_{p}}$')
# axs[4].set_yscale('log')

axs[5].plot(omni_data['Beta'])
axs[5].set_yscale('log')
axs[5].set_ylabel(r'$Plasma\;\beta$')

axs[6].plot(omni_data['SYM_H'], label='SYM-H')
axs[6].set_ylabel('Dst (nT)')

for ax in axs:
    ax.legend(loc='upper right', frameon=False, prop={'size': 10})
    ax.set_xlim([omni_data.index[0], omni_data.index[-1]])

plt.xlabel('Datetime')
fig.tight_layout()
plt.show()

# In[]: --- FOR PLOTTING 1-HR OMNI DATA --- 
start_datetime = datetime(2017, 9, 6, 0, 0, 0)
end_datetime = datetime(2017, 9, 17, 0, 0, 0)

omni_data = get_omni_hr(start_datetime, end_datetime)

fig, axs = plt.subplots(8, 1, figsize=(15,13), sharex=True)
fig.suptitle('1-hr OMNI data: ' + str(start_datetime) +' - ' + str(end_datetime))

axs[0].plot(omni_data['F1800'])
axs[0].set_ylabel('r$B_t$ $(nT)$')

axs[1].plot(omni_data['BX_GSE1800'], color='dodgerblue', label=r'$B_x$ $GSE$')
axs[1].plot(omni_data['BY_GSE1800'], color='green', label=r'$B_y$ $GSE$')
axs[1].plot(omni_data['BZ_GSE1800'], color='orange', label='r$B_z$ $GSE$')
axs[1].set_ylabel(r'$B_{x,y,z}\;(nT)$')

axs[2].plot(omni_data['V1800'], label='$V_{sw}$')
axs[2].set_ylabel(r'$V_{sw}$ $(km.s^{-1})$')

axs[3].plot(omni_data['T1800'], label='$T_p$')
# Calculating half the expected solar wind temperature (0.5Texp) 
# the method I found in the paper: 
# Lopez, R. E., & Freeman, J. W. (1986). 
# Solar wind proton temperature‐velocity relationship. 
# Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
if mean(omni_data['V1800']) > 500:
    # for the high-speed wind 
    Texp = (0.5*((0.031*omni_data['V1800']) - 4.39)**2)
else:
    # for the high-speed wind 
    Texp = (0.5*((0.77*omni_data['V1800']) - 265)**2)
Texp.rename('Texp', inplace=True)

axs[3].plot(Texp, label='$0.5T_{exp}$')
axs[3].set_ylabel('T (K)')
axs[3].set_yscale('log')

axs[4].plot(omni_data['Ratio1800'])
axs[4].set_ylabel('$\dfrac{N_{alpha}}{N_{p}}$')

axs[5].plot(omni_data['N1800'], label=r'$n_{p}$ $(cm^{-3})$')
axs[5].plot(omni_data['Pressure1800'], label='P (nPa)')

axs[6].plot(omni_data['Beta1800'])
axs[6].set_ylabel(r'$Plasma\;\beta$')

axs[7].plot(omni_data['DST1800'])
axs[7].set_ylabel('Dst (nT)')

for ax in axs:
    ax.legend(loc='upper right', frameon=False, prop={'size': 10})
    ax.set_xlim([omni_data.index[0], omni_data.index[-1]])

plt.xlabel('Datetime')
fig.tight_layout()            
plt.show()

# In[]: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# ------- ANOTHER METHOD OF MATCHING ------- 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# In[]: --- MATCHING TWO LISTS --- 
''' SOHO/LASCO CME Catalog & WIND List of ICMEs ''' 

CMEs = read_excel('CMEs_preprocessed.xlsx')
ICMEs = read_excel('ICME_catalog_from_WIND.xlsx', sheet_name='Sheet1')

# to start logically with the ICMEs list 
CMEs = CMEs.drop([0,1,2,3,4,5,6], axis=0)
ICMEs = ICMEs.drop([0,1,2,3,4,5,6,7,8,9,10], axis=0)

CMEs['Datetime'] = to_datetime(CMEs['Datetime'])
ICMEs['ICME_st'] = to_datetime(ICMEs['ICME_st'])
ICMEs['MO_st'] = to_datetime(ICMEs['MO_st'])
ICMEs['MO_ICME_et'] = to_datetime(ICMEs['MO_ICME_et'])

CMEs = CMEs.set_index('Datetime')
ICMEs = ICMEs.set_index('ICME_st')

# In[]: --- 
# matched_table = DataFrame(columns=CMEs.columns)
matched_table = DataFrame()
trans_time = []

for cme in range(len(CMEs)):
    arrival_datetime = G2001(CMEs.Datetime[cme], CMEs.Linear_Speed[cme])
    trans_time.append(arrival_datetime)
    
trans_time = to_datetime(trans_time)
    
    
    # dt = arrival_datetime - CMEs.Datetime[cme]

for i in range(len(trans_time)):
    
    start_window = trans_time[i] - timedelta(hours=11.04)
    end_window = trans_time[i] + timedelta(hours=11.04)
    
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
    
    t = date_range(start_datetime, end_datetime)
    
    

# min(trans_time, key=lambda s: trans_time[s] - ICMEs['ICME_st'])



# def nearest(items, pivot):
#     return min(items, key=lambda x: abs(x - pivot))
#     # return min(item for item in items if item > pivot)

# def nearest(items, pivot):
#     if pivot in items:
#         return pivot
#     else:
#         return min(items, key=lambda x: abs(x - pivot))

# near = nearest(ICMEs.ICME_st, trans_time)




# ICMEs.truncate(before=est_trans_time[5])
# ICMEs.iloc[ICMEs.index.get_loc(est_trans_time[0], method='backfill')]    
    
    
    # matched_table = matched_table.append({'CME_Datetime': CMEs.Datetime[cme], 
    #                                       'CME_Width': CMEs.Width[cme], 
    #                                       'CME_Linear_Speed': CMEs.Linear_Speed[cme], 
    #                                       'CME_Initial_Speed': CMEs.Initial_Speed[cme], 
    #                                       'CME_Final_Speed': CMEs.Final_Speed[cme], 
    #                                       'CME_Speed_20Rs': CMEs.Speed_20Rs[cme], 
    #                                       'CME_Accel': CMEs.Accel[cme], 
    #                                       'CME_MPA': CMEs.MPA[cme]}, 
    #                                       # 'ICME_Datetime': , 
    #                                       # 'Model_trans_time_hrs': , 
    #                                       ignore_index=True)


# near = []
# for i in range(len(trans_time)):
#     near.append(ICMEs.iloc[ICMEs.index.get_loc(trans_time[i], method='nearest')]) # backfill
# near = DataFrame(near)

# ICMEs.truncate(before=trans_time[20])

# from datetime import timedelta, datetime
# base_date = ICMEs.ICME_st[11]
# # b_d = datetime.strptime(base_date, "%m/%d %I:%M %p")
# def func(x):
#     # d =  datetime.strptime(x[0], "%m/%d %I:%M %p")
#     delta = x - base_date if x > base_date else timedelta.max
#     return delta

# min(trans_time, key = func)



# ICMEs.index.get_loc(trans_time[20], method='nearest')


# In[]: --- 
T = []
for i in range(len(ICMEs)):
    est_t = ICMEs.index[i] - timedelta(days=3)
    for j in range(len(CMEs)):
        T.append(min(est_t - CMEs.index[j]))


# In[]: --- 
# est_t = ICMEs.index[0] - timedelta(days=4)
# T = []
# for j in range(len(CMEs)):
#     T.append({'CME_datetime': CMEs.index[j], 
#               'est_t': abs(est_t - CMEs.index[j])})
    

# T = DataFrame(T)


# In[]: --- 
# >>> <<< 
sample = read_excel('List_from_Interplanetary shocks lacking type II radio bursts_paper_178_CME-ICME_pairs.xlsx')

# In[]: --- forecast the arrival time of a CME event from the sample data 
event_num = 176

arrival_datetime = G2001(sample.CME_Datetime[event_num], sample.CME_Speed[event_num])

dt = arrival_datetime - sample.CME_Datetime[event_num]
print('\nCME launced on:', sample.CME_Datetime[event_num])
print('Estimated arrival time is:', arrival_datetime)
print('Estimated Transit time is: {} days, {} hours, {} minutes' .format(dt.components.days, 
                                                                         dt.components.hours, 
                                                                         dt.components.minutes))
print('-------------------------------------------------------')

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

omni_data_raw = get_omni_hr(start_datetime, end_datetime)

# In[]: --- filter the OMNI columns 
omni_col_lst = omni_data_raw.columns.values.tolist()
omni_data = omni_data_raw.filter(omni_data_raw.columns[[1,6,7,8,9,12,13,15,25]])
print(*omni_data.columns, sep= '\n')
omni_data.plot(subplots=True)
plt.show()

# In[]: --- 
fig, ax = plt.subplots(figsize=(15,4))
omni_data['DST1800'].plot(legend=False, ax=ax)
plt.show()

# In[]: --- 
fig, ax = plt.subplots(figsize=(15,4))
omni_data['T1800'].plot(legend=False, ax=ax)
if mean(omni_data['V1800']) > 500:
    Texp = (0.5*((0.031*omni_data['V1800']) - 4.39)**2)
else:
    Texp = (0.5*((0.77*omni_data['V1800']) - 265)**2)
Texp.rename('Texp', inplace=True)
Texp.plot(legend=False, ax=ax)
plt.show()

# In[]: --- 
#Convert the time series values to a numpy 1D array
# points = np.array(omni_data['DST1800'])
    
#RUPTURES PACKAGE
#Changepoint detection with the Pelt search method
# model="rbf"
# algo = rpt.Pelt(model=model).fit(points)
# result = algo.predict(pen=10)
# rpt.display(points, result, figsize=(10, 6))
# plt.title('Change Point Detection: Pelt Search Method')
# plt.show()  
    
#Changepoint detection with the Binary Segmentation search method
# model = 'l2'
# algo = rpt.Binseg(model=model).fit(points)
# my_bkps = algo.predict(n_bkps=10)
# # show results
# rpt.show.display(points, my_bkps, figsize=(15, 4))
# plt.title('Change Point Detection: Binary Segmentation Search Method')
# plt.show()
    
#Changepoint detection with window-based search method
# model = "l2"  
# algo = rpt.Window(width=40, model=model).fit(points)
# my_bkps = algo.predict(n_bkps=10)
# rpt.show.display(points, my_bkps, figsize=(10, 6))
# plt.title('Change Point Detection: Window-Based Search Method')
# plt.show()
    
#Changepoint detection with dynamic programming search method
# model = "l1"  
# algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(points)
# my_bkps = algo.predict(n_bkps=10)
# rpt.show.display(points, my_bkps, figsize=(10, 6))
# plt.title('Change Point Detection: Dynamic Programming Search Method')
# plt.show()

# In[]: --- 
figs, axs = [], []
for col in omni_data.columns:
    points = np.array(omni_data[col])
    algo = rpt.Dynp(model='l2').fit(points)
    result = algo.predict(n_bkps=2)
    fig, ax = rpt.display(points, result, result, figsize=(15, 3))
    figs.append(fig)
    axs.append(ax)
    plt.title(col + '\nChange Point Detection: Dynamic Programming Search Method')
    plt.tight_layout(pad=1.0)
    plt.xlim([0, len(points)-1])
    plt.show()

# In[]: --- 
points = np.array(omni_data['DST1800'])
algo = rpt.Dynp(model='l2').fit(points)
result = algo.predict(n_bkps=2)
fig, ax = rpt.display(points, result, result, figsize=(15, 3))
figs.append(fig)
axs.append(ax)
plt.title(col + '\nChange Point Detection: Dynamic Programming Search Method')
plt.tight_layout(pad=1.0)
plt.xlim([0, len(points)-1])
plt.show()
# # getting the timestamps of the change points
bkps_timestamps = omni_data['DST1800'].iloc[[0] + result[:-1] +[-1]].index
# computing the durations between change points
durations = (bkps_timestamps[1:] - bkps_timestamps[:-1])
# hours
d = durations.seconds/60/60
d_f = DataFrame(d)

one = f'1st segment {d_f.values[0][0]} hours'
two = f'2nd segment {d_f.values[1][0]} hours'
three = f'3rd segment {d_f.values[2][0]} hours'

plt.title('Change Point Detection: Binary Segmentation Search Method')
plt.text(0.5, -9, 'Retrieved durations between change points, in hours:')
plt.text(0.5, -13, one)
plt.text(0.5, -18, two)
plt.text(0.5, -23, three)
plt.show()

fig, ax = plt.subplots(figsize=(15,3))
omni_data['DST1800'].plot(legend=False, ax=ax)
st = bkps_timestamps[1]
et = bkps_timestamps[2]
ax.axvspan(st, et, facecolor='#FFCC66', alpha=0.5)
window = omni_data['DST1800'].loc[st:et]
min_idx = window[window.values==min(window)].index
ax.axvline(min_idx.values, color='r', linewidth=2, linestyle='--')
plt.tight_layout()
plt.show()

# In[]: --- EXAMPLE --- 
# make random data with 100 samples and 9 columns 
# n_samples, n_dims, sigma = 100, 9, 2
# n_bkps = 4
# signal, bkps = rpt.pw_constant(n_samples, n_dims, n_bkps, noise_std=sigma)

# figs, axs = [], []
# for i in range(signal.shape[1]):
#     points = signal[:,i]
#     # detection of change points 
#     algo = rpt.Dynp(model='l2').fit(points)
#     result = algo.predict(n_bkps=2)
#     fig, ax = rpt.display(points, bkps, result, figsize=(15,3))
#     figs.append(fig)
#     axs.append(ax)
#     plt.show()

# fig, ax = plt.subplots()
# for i in range(5):
#     f, x = rpt.display(points, result, result)
#     fig.add_subplot(111)
    



# In[]: --- 
# detection 
points = np.array(omni_data['DST1800'])
algo = rpt.Dynp(model='l2').fit(points)
result = algo.predict(n_bkps=2)
# display 
rpt.display(points, result, result, figsize=(15, 3))
plt.title('Change Point Detection: Dynamic Programming Search Method')
plt.tight_layout(pad=1.0)
plt.xlim([0, len(points)-1])
plt.show()

# In[]: --- 
fig, axs = plt.subplots(8, 1, figsize=(15,13), sharex=True)
fig.suptitle('1-hr OMNI data: ' + str(start_datetime) +' - ' + str(end_datetime) + '\nChange Point Detection: Dynamic Programming Search Method')

axs[0].plot(omni_data['F1800'])
axs[0].set_ylabel('r$B_t$ $(nT)$')

axs[1].plot(omni_data['BX_GSE1800'], color='dodgerblue', label=r'$B_x$ $GSE$')
axs[1].plot(omni_data['BY_GSE1800'], color='green', label=r'$B_y$ $GSE$')
axs[1].plot(omni_data['BZ_GSE1800'], color='orange', label='r$B_z$ $GSE$')
axs[1].set_ylabel(r'$B_{x,y,z}\;(nT)$')

axs[2].plot(omni_data['V1800'], label='$V_{sw}$')
axs[2].set_ylabel(r'$V_{sw}$ $(km.s^{-1})$')

axs[3].plot(omni_data['T1800'], label='$T_p$')
# Calculating half the expected solar wind temperature (0.5Texp) 
# the method I found in the paper: 
# Lopez, R. E., & Freeman, J. W. (1986). 
# Solar wind proton temperature‐velocity relationship. 
# Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
if mean(omni_data['V1800']) > 500:
    Texp = (0.5*((0.031*omni_data['V1800']) - 4.39)**2)
else:
    Texp = (0.5*((0.77*omni_data['V1800']) - 265)**2)
Texp.rename('Texp', inplace=True)

axs[3].plot(Texp, label='$0.5T_{exp}$')
axs[3].set_ylabel('T (K)')
axs[3].set_yscale('log')

axs[4].plot(omni_data['Ratio1800'])
axs[4].set_ylabel('$\dfrac{N_{alpha}}{N_{p}}$')

axs[5].plot(omni_data['N1800'], label=r'$n_{p}$ $(cm^{-3})$')
axs[5].plot(omni_data['Pressure1800'], label='P (nPa)')

axs[6].plot(omni_data['Beta1800'])
axs[6].set_ylabel(r'$Plasma\;\beta$')

axs[7].plot(omni_data['DST1800'])
axs[7].set_ylabel('Dst (nT)')

for ax in axs:
    ax.legend(loc='upper right', frameon=False, prop={'size': 10})
    ax.set_xlim([0, len(points)-1])

plt.xlabel('Timestep')
fig.tight_layout(pad=1.0)            
plt.show()




# In[]: --- 



# In[]: --- 









# In[]: --- 





# In[]: --- 











# In[]: --- 


















