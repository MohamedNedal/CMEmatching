"""
Plotting OMNI Data
==================

Importing and plotting data from OMNI Web Interface.
OMNI provides interspersed data from various spacecrafts.
There are 55 variables provided in the OMNI Data Import.
"""
import numpy as np
from pandas import read_excel, DataFrame
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os.path
from necessary_functions import get_omni, G2001
import warnings
warnings.filterwarnings('ignore')

# In[]: Establishing the output folder 
save_path = 'D:/Study/Academic/Research/Master Degree/Master Work/Software/Codes/Python/Heliopy Examples/auto_examples_python/'
try:
    os.mkdir(save_path + 'Output_plots')
except OSError as error:
    print(error)

# In[]: IMPORT THE LIST OF RANDOM CME EVENTS 
sample = read_excel('Random_40_CMEs.xlsx', index_col='Datetime')

# Create an empty table to be filled with the CME info and its estimated transit time 
final_table = []

# In[]: --- Try to build a JSON file structure with these info for all events --- 
print('Predicting the transit time of the CMEs using G2001 model .. \n')

event_num = 4
arrival_datetime = G2001(sample.index[event_num], sample.Linear_Speed[event_num])

dt = arrival_datetime - sample.index[event_num]
print('CME number:', event_num+1)
print('Event launched at:', sample.index[event_num])
print('Arrived at:', arrival_datetime)
print('Estimated Transit time: {} days, {} hours, {} minutes' .format(dt.components.days, 
                                                                      dt.components.hours, 
                                                                      dt.components.minutes))
print('with mean error of:', 11.04, ' hours')
print('-------------------------------------------------------')

# In[]: --- 
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

# In[]: --- 
fig, axs = plt.subplots(8, 1, figsize=(15,15), dpi=300, sharex=True)

axs[0].plot(omni_data['F1800'], color='blue', label='$B_t$')
axs[0].plot(omni_data['BZ_GSE1800'], color='red', label='$B_z$')
axs[0].set_ylabel('B (nT)')

axs[1].plot(omni_data['THETA_AV1800'], label=r'$\theta$')
axs[1].plot(omni_data['PHI_AV1800'], label=r'$\phi$')
axs[1].set_ylabel('Angle (deg)')

axs[2].plot(omni_data['BX_GSE1800'], label=r'$B_x$ $GSE$')
axs[2].plot(omni_data['BY_GSE1800'], label=r'$B_y$ $GSE$')
axs[2].set_ylabel('B (nT)')

axs[3].plot(omni_data['V1800'], label='Flow Speed')
axs[3].set_ylabel(r'$V$ $(km.s^{-1})$')

axs[4].plot(omni_data['N1800'], label='Proton Density')
axs[4].set_ylabel(r'$n_{p}$ $(cm^{-3})$')

axs[5].plot(omni_data['Pressure1800'], label='Dynamic Pressure')
axs[5].set_ylabel('P (nPa)')

axs[6].plot(omni_data['Beta1800'], label='Plasma Beta')
axs[6].set_ylabel('Beta')

axs[7].plot(omni_data['DST1800'], label='Dst index')
axs[7].set_ylabel('Dst (nT)')

# Assign a threshold of the Dst (nT) to look for geomagnetic storms 
offset = 5
threshold = np.average(omni_data['DST1800'] - offset)
# Select all the rows which satisfies the criteria 
# Convert the collection of index labels to list 
Index_label_Dst = omni_data[omni_data['DST1800'] <= threshold].index.tolist()

print('\nDefine timestamps where Dst < ', round(threshold,2), 'nT')

# Find Geomagnetic Storms in Data 
for i in range(1, len(omni_data)):
    if omni_data['DST1800'][i] < threshold:

        axs[0].axvline(omni_data['F1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[0].axvspan(omni_data['F1800'].index[i-1], omni_data['F1800'].index[i], facecolor='#FFCC66', alpha=0.5)

        axs[1].axvline(omni_data['THETA_AV1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[1].axvspan(omni_data['THETA_AV1800'].index[i-1], omni_data['THETA_AV1800'].index[i], facecolor='#FFCC66', alpha=0.5)
        
        axs[2].axvline(omni_data['BX_GSE1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[2].axvspan(omni_data['BX_GSE1800'].index[i-1], omni_data['BX_GSE1800'].index[i], facecolor='#FFCC66', alpha=0.5)
        
        axs[3].axvline(omni_data['V1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[3].axvspan(omni_data['V1800'].index[i-1], omni_data['V1800'].index[i], facecolor='#FFCC66', alpha=0.5)

        axs[4].axvline(omni_data['N1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[4].axvspan(omni_data['N1800'].index[i-1], omni_data['N1800'].index[i], facecolor='#FFCC66', alpha=0.5)

        axs[5].axvline(omni_data['Pressure1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[5].axvspan(omni_data['Pressure1800'].index[i-1], omni_data['Pressure1800'].index[i], facecolor='#FFCC66', alpha=0.5)

        axs[6].axvline(omni_data['Beta1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[6].axvspan(omni_data['Beta1800'].index[i-1], omni_data['Beta1800'].index[i], facecolor='#FFCC66', alpha=0.5)        
        
        axs[7].axvline(omni_data['DST1800'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[7].axvspan(omni_data['DST1800'].index[i-1], omni_data['DST1800'].index[i], facecolor='#FFCC66', alpha=0.5)

''' 
 RUN A FOR-LOOP THAT CHECK THE MIN. DIFF. BETWEEN 
 'arrival_datetime' & 'Index_label_<ANY_SW_PARAM.>' 
 & STORE THE 'idx' VALUES FOR EACH SW PARAM. AS A LIST 
 FINALLY, TAKE THE AVG. OF THAT LIST, TO BE 
 THE PROPABLE ARRIVAL TIME OF THE EVENT. 
 
 ''' 

for ax in axs:
    # estimated travel time from G2001 model 
    ax.axvline(arrival_datetime, label='G2001', color='green', alpha=0.7, linewidth=3, linestyle='--')
    
    # Find the local min Dst value within the window of 'Index_label_Dst' 
    try: 
        if Index_label_Dst == []:
            print('Index_label_Dst is empty')
        else:            
            min_Dst_window = min(omni_data['DST1800'].loc[Index_label_Dst[0]:Index_label_Dst[-1]])
    except IndexError as idx_err:
        print(idx_err)
    
    # at min value of Dst index 
    # try:
    #     ax.axvline(all(omni_data[omni_data['DST1800']==min_Dst_window].index), label='Min(Dst)', color='red', alpha=0.7, linewidth=3, linestyle='--')
    # except ValueError as valerr:
    #     print(valerr)
    ax.axvline(omni_data[omni_data['DST1800']==min_Dst_window].index[0], label='Min(Dst)', color='red', alpha=0.7, linewidth=3, linestyle='--')
    
    dt_G2001_idxLabel = []
    for idx in range(len(Index_label_Dst)):
        dt_G2001_idxLabel.append(abs(arrival_datetime - Index_label_Dst[idx]))
    
    # at the index of the min diff betn t_G2001 and t_min(Dst) within the window 
    ax.axvline(Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))], label='Min(dt)', color='black', alpha=0.7, linewidth=3, linestyle='--')
    
    
    T_RED = omni_data[omni_data['DST1800']==min_Dst_window].index[0]
    T_BLACK = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))]
    T_GREEN = arrival_datetime
    
    from pandas import Timestamp
    AVG = Timestamp((T_RED.value + T_BLACK.value + T_GREEN.value)/3.0)
    
    ax.axvline(AVG, label='AVG_t', color='brown', alpha=0.7, linewidth=3, linestyle='--')
    
    ax.legend(loc='upper right', frameon=False, prop={'size': 10})
    ax.set_xlim([omni_data.index[0], omni_data.index[-1]])

plt.xlabel('Date')
fig.tight_layout()

# ============================================= 
st = str(omni_data.index[0].year)+str(omni_data.index[0].month)+str(omni_data.index[0].day)+str(omni_data.index[0].hour)+str(omni_data.index[0].minute)+str(omni_data.index[0].second)
en = str(omni_data.index[-1].year)+str(omni_data.index[-1].month)+str(omni_data.index[-1].day)+str(omni_data.index[-1].hour)+str(omni_data.index[-1].minute)+str(omni_data.index[-1].second)
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_Data_for_CME_No_'+str(event_num)+'_'+st+'-'+en+'.png'))

print('\nNaNs?', omni_data.isnull().values.any())

# APPEND THE OUTPUT TRANSIT TIME WITH THE CME INFO 
est_trans_time = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))] - sample.index[event_num]

tran_time_hours = (est_trans_time.components.days * 24) + (est_trans_time.components.minutes / 60) + (est_trans_time.components.seconds / 3600)

# In[]: FINALIZE THE OUTPUT 
cols = sample.columns

cols = cols.insert(0, 'CME_datetime')
cols = cols.insert(len(cols)+1, 'Transit_time_hrs')
cols = cols.insert(len(cols)+1, 'est_ICME_datetime')

final_table = DataFrame(columns=cols)

final_table = final_table.append({'CME_datetime': sample.index[event_num], 
                                  'Width': sample['Width'][event_num], 
                                  'Linear_Speed': sample['Linear_Speed'][event_num], 
                                  'Initial_Speed': sample['Initial_Speed'][event_num], 
                                  'Final_Speed': sample['Final_Speed'][event_num], 
                                  'Speed_20Rs': sample['Speed_20Rs'][event_num], 
                                  'Accel': sample['Accel'][event_num], 
                                  'MPA': sample['MPA'][event_num], 
                                  'Transit_time_hrs': tran_time_hours, 
                                  'est_ICME_datetime': est_trans_time}, ignore_index=True)

# In[] --- 




# In[] --- 




# In[] --- 








# In[] --- 








# In[]: --- 
# Days = est_trans_time.days
# Seconds = est_trans_time.seconds

# Hours = Seconds//3600 # // to get only the integer part 
# Minutes = (Seconds//60)%60 # to get the floar of the rest 

# # 1 day == 24 hours == 1440 minutes == 86400 seconds 
# # hours = (hours/day) + (hours/minutes) + (hours/seconds) 
# H = (Days*24) + (Minutes/60) + (Seconds/3600)

# print(round(H,2))

# In[] --- 

# from time import strftime

# # NOTE: The following resets if it goes over 23:59:59 
# print(strftime('%H:%M:%S', gmtime(S)))

# In[]: Define the duration (year, month day, hour, minute, second) 
# starttime = datetime(2013, 1, 17, 0, 0, 0)
# endtime = datetime(2013, 1, 20, 23, 59, 59)

# omni_raw = omni.h0_mrg1hr(starttime, endtime)
# data = omni_raw.data
# print('\nOMNI columns are:\n', *data.columns, sep='\n')

# In[]: --- 
# ======================== IMPORTING CME DATASET FROM SOHO/LASCO CATALOGUE ======================== 

# In[]: Import CME data 
# soho = read_excel('CMEs_SOHO.xlsx', sheet_name='CMEs_SOHO')

# In[]: --- 
# new_datetime = []
# uncertain_Accel = []
# for i in range(0, len(soho)):
#     try:
#         # Fix the date-time column 
#         soho['Date'][i] = soho['Date'][i].date()
#         new_datetime.append(datetime.combine(soho.Date[i], soho.Time[i]))
#         # Remove the '*' character in 'Accel' column 
#         if soho['Accel'][i][-1] == '*':
#             # uncertain_Accel.append(soho.index[i])
#             uncertain_Accel.append(new_datetime[i])
#             soho['Accel'][i] = soho['Accel'][i][:-1]
#     except:
#         pass

# soho.insert(0, 'Datetime', new_datetime)

# In[]: Selecting rows based on conditions 
# print('\nSOHO columns are:\n', *soho.columns, sep='\n')

# # Take these columns as energy channels 
# soho = soho.drop(columns={'Date','Time','CPA','Mass','KE'})
# # soho = soho.filter(soho[['Date','Time','CPA','Width']])

# print('\nFinal SOHO columns are:\n', *soho.columns, sep='\n')

# In[]: Clean data 
# filtered_soho = soho.drop(soho[soho.values == '----'].index)
# filtered_soho = soho.drop(soho.values == '----')

# In[]: Converting columns' types 

# print(filtered_soho.dtypes)

# filtered_soho['Linear_Speed'] = to_numeric(filtered_soho['Linear_Speed'], errors='coerce')
# filtered_soho['Linear_Speed'] = filtered_soho['Linear_Speed'].astype(float)
# filtered_soho['Initial_Speed'] = to_numeric(filtered_soho['Initial_Speed'], errors='coerce')
# filtered_soho['Initial_Speed'] = filtered_soho['Initial_Speed'].astype(float)
# filtered_soho['Final_Speed'] = to_numeric(filtered_soho['Final_Speed'], errors='coerce')
# filtered_soho['Final_Speed'] = filtered_soho['Final_Speed'].astype(float)
# filtered_soho['Speed_20Rs'] = to_numeric(filtered_soho['Speed_20Rs'], errors='coerce')
# filtered_soho['Speed_20Rs'] = filtered_soho['Speed_20Rs'].astype(float)
# filtered_soho['Accel'] = to_numeric(filtered_soho['Accel'], errors='coerce')
# filtered_soho['Width'] = filtered_soho['Width'].astype(float)
# filtered_soho['MPA'] = filtered_soho['MPA'].astype(float)

# print(filtered_soho.dtypes)

# In[]: --- 
# # Filter the data 
# filtered_soho_lv1 = filtered_soho[(filtered_soho['Linear_Speed'].values >= 700) & (filtered_soho['Width'].values >= 120)]

# # Randomly select number of rows 
# num_rand_samples = 40
# sample = filtered_soho_lv1.sample(n=num_rand_samples)

# sample = sample.set_index('Datetime')

# sample.to_excel('Random_'+str(num_rand_samples)+'_CMEs.xlsx') # already done 

# In[]: --- PLOT THE SIGMA VALUE --- 
# plt.figure(figsize=(15,3))
# plt.plot(omni_data.index, omni_data['ABS_B1800'], label='B_abs')

# plt.fill_between(omni_data.index, 
#                  omni_data['ABS_B1800']-omni_data['SIGMA$ABS_B1800'], 
#                  omni_data['ABS_B1800']+omni_data['SIGMA$ABS_B1800'], 
#                  facecolor='grey', alpha=0.15, label='Sigma')

# plt.xlabel('Datetime')
# plt.ylabel(r'$B_{ABS}$ $(nT)$')
# plt.legend(frameon=False)
# plt.xlim([omni_data.index[0], omni_data.index[-1]])
# plt.show()

# In[]: --- 
# fig, ax = plt.subplots(1, 1, figsize=(15,3), dpi=300, sharex=True)
# ax.plot(data['phi'].loc['2013-01-17':'2013-01-20'], label='phi')
# ax.plot(data['theta'].loc['2013-01-17':'2013-01-20'], label='theta')
# ax.legend(loc='upper right', frameon=False, prop={'size': 10})
# ax.set_xlim(['2013-01-17', '2013-01-21'])
# plt.xlabel('Date')
# plt.ylabel('Angle (degrees)')
# fig.tight_layout()
# plt.show()

# In[]: --- 
# [phi, theta, Br] = magnetic_conversion(data['BX_GSE'], data['BY_GSM'], data['BZ_GSM'])

# data.insert(15, 'Br', Br)
# data.insert(16, 'phi', phi)
# data.insert(17, 'theta', theta)

# In[] --- 
# TEST 
# st_ = datetime(2013,12,15,0,0,0)
# en_ = datetime(2013,12,16,23,0,0)
# test = get_omni(st_, en_)

# [phi, theta, Br] = magnetic_conversion(test['BX_GSE'], test['BY_GSM'], test['BZ_GSM'])

# test.insert(0, 'Br', Br)
# test.insert(1, 'phi', phi)
# test.insert(2, 'theta', theta)

# fig, axs = plt.subplots(3, 1, figsize=(15,7), dpi=300, sharex=True)
# axs[0].plot(test['phi'], label='phi')
# axs[1].plot(test['theta'], label='theta')
# axs[2].plot(test['Br'], label='Br')
# for ax in axs:
#     ax.legend(loc='upper right', frameon=False, prop={'size': 10})
# plt.xlabel('Date')
# fig.tight_layout()
# plt.show()

# In[] --- 
# # Import
# from statsmodels.tsa.seasonal import seasonal_decompose
# # Decompose time series into daily trend, seasonal, and residual components.
# # Note that the settlement price = average daily price.
# if fdata['Pressure'].isnull().values.any() == True:
#     print('NaNs found in data .. Proceeding with interpolation')
#     fdata['Pressure'] = fdata['Pressure'].fillna(method='backfill')
#     print('NaNs in data?', fdata['Pressure'].isnull().values.any())
# else:
#     print('NaNs in interpolateа data?', fdata['Pressure'].isnull().values.any())
    
# decomp = seasonal_decompose(fdata['Pressure'], period = 360)
# # Plot the decomposed time series to interpret.
# decomp.plot()

# plt.figure()
# fdata['Pressure'].plot()
# plt.xlabel('Date')
# plt.ylabel('Pressure (nPa)')
# plt.title('Raw Data')
# plt.show()

# plt.figure()
# trend = decomp.trend
# trend.plot()
# plt.xlabel('Date')
# plt.ylabel('Trend')
# plt.title('General Trend')
# plt.show()

# plt.figure()
# seasonal = decomp.seasonal
# seasonal.plot()
# plt.xlabel('Date')
# plt.ylabel('Seasonal')
# plt.title('Seasonal Variations')
# plt.show()

