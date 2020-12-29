"""
Plotting OMNI Data
==================

Importing and plotting data from OMNI Web Interface.
OMNI provides interspersed data from various spacecrafts.
There are 55 variables provided in the OMNI Data Import.
"""
import math
import numpy as np
from pandas import read_excel, to_numeric
# import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import heliopy.data.omni as omni
import os.path
import warnings
warnings.filterwarnings('ignore')

# In[]: --- 
def magnetic_conversion(Bx, By, Bz):
    '''
    This function is to obtain the ('phi','theta') angles 
    and the radial components of the IMF from its cartesian components. 

    Parameters
    ----------
    Bx : pandas.core.series.Series 
        The x-component of the interplanetary magnetic field. 
        
    By : pandas.core.series.Series 
        The y-component of the interplanetary magnetic field. 
        
    Bz : pandas.core.series.Series 
        The z-component of the interplanetary magnetic field. 

    Returns
    -------
    phi : pandas.core.series.Series 
        The angle between Bx and By (in rad. between +Pi and -Pi). 
        
    theta : pandas.core.series.Series 
        The angle between Bz and xy plane (in rad. between +Pi and -Pi). 
        
    Br : pandas.core.series.Series 
        The r-component (total) of the interplanetary magnetic field. 

    '''
    
    Br = np.sqrt((Bx.values**2) + (By.values**2) + (Bz.values**2))
        
    theta, phi = [], []
    
    for i in range(0, len(Bx)):
        
        theta_rad = math.acos(Bz.values[i]/Br[i])        # returns the angle in radians 
        phi_rad = math.atan2(By.values[i], Bx.values[i]) # returns the angle in radians 
        
        theta_deg = math.degrees(theta_rad) # returns the angle in degrees 
        phi_deg = math.degrees(phi_rad)     # returns the angle in degrees 
        
        theta.append(theta_deg) 
        phi.append(phi_deg) 
    
    theta = np.array(theta).astype('float32')
    phi = np.array(phi).astype('float32')

    return phi, theta, Br

# In[]: --- 
[phi, theta, Br] = magnetic_conversion(data['BX_GSE'], data['BY_GSM'], data['BZ_GSM'])

data.insert(15, 'Br', Br)
data.insert(16, 'phi', phi)
data.insert(17, 'theta', theta)

# In[]: --- 
fig, ax = plt.subplots(1, 1, figsize=(15,3), dpi=300, sharex=True)
ax.plot(data['phi'].loc['2013-01-17':'2013-01-20'], label='phi')
ax.plot(data['theta'].loc['2013-01-17':'2013-01-20'], label='theta')
ax.legend(loc='upper right', frameon=False, prop={'size': 10})
ax.set_xlim(['2013-01-17', '2013-01-21'])
plt.xlabel('Date')
plt.ylabel('Angle (degrees)')
fig.tight_layout()
plt.show()

# In[]: Establishing the output folder 
save_path = 'D:/Study/Academic/Research/Master Degree/Master Work/Software/Codes/Python/Heliopy Examples/auto_examples_python/'
try:
    os.mkdir(save_path + 'Output_plots')
except OSError as error:
    print(error)

# In[]: --- 
def get_omni(start_datetime, end_datetime):
    '''
    This function to get OMNI data and plot it, for CME-ICME matching. 
    
    Type: High Resolution OMNI (HRO) 
    Source: https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/ 
    Decsribtion: https://omniweb.gsfc.nasa.gov/html/HROdocum.html 
    
    Parameters
    ----------
    start_datetime : datetime object
        The format is 'yyyy,m,d,H,M,S'.
        
    end_datetime : datetime object
        The format is 'yyyy,m,d,H,M,S'.

    Returns
    -------
    Dataframe of the OMNI data within that period. 

    '''
    
    omni_data = omni.hro2_1min(start_datetime, end_datetime)
    data = omni_data.data
    
    # Consider only these columns  
    fdata = data.filter(data[['F',
                              'BX_GSE',
                              'BY_GSM',
                              'BZ_GSM',
                              'flow_speed',
                              'proton_density',
                              'Beta',
                              'Pressure',
                              'SYM_H']])
    
    # Get a list of columns names of Filtered data 
    print('\nFiltered columns are:\n', *fdata.columns, sep='\n')
    
    return fdata

# In[]: Define the duration (year, month day, hour, minute, second) 
starttime = datetime(2013, 1, 17, 0, 0, 0)
endtime = datetime(2013, 1, 20, 23, 59, 59)

# test 
# >>> ADJUST THE CODE TO WORK ON THIS MODULE 
omni_raw = omni.h0_mrg1hr(starttime, endtime)
data = omni_raw.data
# ---------------------------------------------------------- 
# # High Resolution OMNI (HRO)
# # Source: https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/ 
# # Decsribtion: https://omniweb.gsfc.nasa.gov/html/HROdocum.html 
omni_data = omni.hro2_1min(starttime, endtime)
data = omni_data.data

# Get a list of columns names of Raw data 
print('\nRaw columns are:\n', *omni_data.columns, sep= '\n')

# # Leave these columns only 
# fdata = data.filter(data[['F',
#                           'BX_GSE',
#                           'BY_GSM',
#                           'BZ_GSM',
#                           'flow_speed',
#                           'proton_density',
#                           'Beta',
#                           'Pressure',
#                           'SYM_H']])

# # Get a list of columns names of Filtered data 
# print('\nFiltered columns are:\n', *fdata.columns, sep='\n')
# labels = fdata.columns.values.tolist()

# In[]: Plotting 
# # Assign a threshold of the Dst (nT) to look for geomagnetic storms 
# threshold = -17
# print('\nDefine timestamps where Dst <', threshold, 'nT')

# fig, axs = plt.subplots(7, 1, figsize=(15,10), dpi=100, sharex=True)
# axs[0].plot(fdata['F'], label='Avg Mag Field')

# axs[1].plot(fdata['BX_GSE'], label='Bx GSE')
# axs[1].plot(fdata['BY_GSM'], label='By GSM')
# axs[1].plot(fdata['BZ_GSM'], label='Bz GSM')

# axs[2].plot(fdata['flow_speed'], label='Flow Speed')

# axs[3].plot(fdata['proton_density'], label='Proton Density')
# axs[4].plot(fdata['Pressure'], label='Pressure')
# axs[5].plot(fdata['Beta'], label='Beta')
# axs[6].plot(fdata['SYM_H'], label='SYM-H')

# # Find Geomagnetic Storms in Data 
# for i in range(1, len(fdata)):
#     if fdata['SYM_H'][i] < threshold:

#         axs[0].axvline(fdata['F'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
#         axs[0].axvspan(fdata['F'].index[i-1], fdata['F'].index[i], facecolor='#FFCC66', alpha=0.7)

#         axs[1].axvline(fdata['BX_GSE'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
#         axs[1].axvspan(fdata['BX_GSE'].index[i-1], fdata['BX_GSE'].index[i], facecolor='#FFCC66', alpha=0.7)

#         axs[2].axvline(fdata['flow_speed'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
#         axs[2].axvspan(fdata['flow_speed'].index[i-1], fdata['flow_speed'].index[i], facecolor='#FFCC66', alpha=0.7)

#         axs[3].axvline(fdata['proton_density'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
#         axs[3].axvspan(fdata['proton_density'].index[i-1], fdata['proton_density'].index[i], facecolor='#FFCC66', alpha=0.7)

#         axs[4].axvline(fdata['Pressure'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
#         axs[4].axvspan(fdata['Pressure'].index[i-1], fdata['Pressure'].index[i], facecolor='#FFCC66', alpha=0.7)

#         axs[5].axvline(fdata['Beta'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
#         axs[5].axvspan(fdata['Beta'].index[i-1], fdata['Beta'].index[i], facecolor='#FFCC66', alpha=0.7)        
        
#         axs[6].axvline(fdata['SYM_H'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
#         axs[6].axvspan(fdata['SYM_H'].index[i-1], fdata['SYM_H'].index[i], facecolor='#FFCC66', alpha=0.7)

#         # print('Value:', fdata['SYM_H'][i], 'at', 'Index:', fdata.index[fdata['SYM_H'][i]])

# for ax in axs:
#     ax.legend(loc='upper right', frameon=False, prop={'size': 10})
#     ax.set_xlim([starttime, endtime])
#     # ax.grid()
# plt.xlabel('Date')
# fig.tight_layout()

# # Select all the rows which satisfies the criteria 
# # convert the collection of index labels to list 
# Index_label = fdata[fdata['SYM_H'] < threshold].index.tolist()

# print('\nNaNs?', fdata.isnull().values.any())

# sta = str(starttime.year)+str(starttime.month)+str(starttime.day)+str(starttime.hour)+str(starttime.minute)+str(starttime.second)
# end = str(endtime.year)+str(endtime.month)+str(endtime.day)+str(endtime.hour)+str(endtime.minute)+str(endtime.second)
# plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_Data_'+sta+'--'+end+'.png'))

# In[]: Finding the BIG rate of change 
# def find_change(current_point, previous_point):
#     if current_point == previous_point:
#         return 0
#     try:
#         ROC = (abs(current_point - previous_point) / previous_point) * 100.0
#         if ROC > 30:
#             f, ax = plt.subplots(figsize=(15,3), dpi=100)
#             ax.plot(fdata['Pressure'], label='Pressure')
#             ax.legend(loc='upper right', frameon=False, prop={'size': 10})
#             ax.set_xlim([starttime, endtime])
#             ax.xlabel('Date')
#             ax.ylabel('Pressure (nPa)')
#             plt.show()
#         return ax
#     except ZeroDivisionError:
#         return float('inf')

# In[]: Finding the locations of BIG rate of change 
# ROC_timestamps = []
# for i in range(1, len(fdata)):
#     ROC = (abs(fdata['Pressure'][i] - fdata['Pressure'][i-1]) / fdata['Pressure'][i-1]) * 100.0
#     if ROC > 80:
#         ROC_timestamps.append(fdata['Pressure'].index[i])

# # Plotting ROC >30% on the data 
# fig, ax = plt.subplots(1, 1, figsize=(15,3), dpi=100)
# ax.plot(fdata['Pressure'], label='Pressure')
# for i in ROC_timestamps:
#     ax.axvline(x=i, label='ROC >30%', color='tomato', alpha=0.7, linewidth=1, linestyle='--')

# ax.set_xlim([starttime, endtime])
# plt.xlabel('Date')
# plt.ylabel('Pressure (nPa)')    
# plt.show()

# In[]: --- 
# ======================== IMPORTING CME DATASET FROM SOHO/LASCO CATALOGUE ======================== 

# In[]: Import CME data 
soho = read_excel('CMEs_SOHO.xlsx', sheet_name='CMEs_SOHO')

# In[]: --- 
new_datetime = []
uncertain_Accel = []
for i in range(0, len(soho)):
    try:
        # Fix the date-time column 
        soho['Date'][i] = soho['Date'][i].date()
        new_datetime.append(datetime.combine(soho.Date[i], soho.Time[i]))
        # Remove the '*' character in 'Accel' column 
        if soho['Accel'][i][-1] == '*':
            # uncertain_Accel.append(soho.index[i])
            uncertain_Accel.append(new_datetime[i])
            soho['Accel'][i] = soho['Accel'][i][:-1]
    except:
        pass

soho.insert(0, 'Datetime', new_datetime)

# In[]: Selecting rows based on conditions 
print('\nSOHO columns are:\n', *soho.columns, sep='\n')

# Take these columns as energy channels 
soho = soho.drop(columns={'Date','Time','CPA','Mass','KE'})
# soho = soho.filter(soho[['Date','Time','CPA','Width']])

print('\nFinal SOHO columns are:\n', *soho.columns, sep='\n')

# In[]: Clean data 
filtered_soho = soho.drop(soho[soho.values == '----'].index)
# filtered_soho = soho.drop(soho.values == '----')

# In[]: Converting columns' types 

print(filtered_soho.dtypes)

filtered_soho['Linear_Speed'] = to_numeric(filtered_soho['Linear_Speed'], errors='coerce')
filtered_soho['Linear_Speed'] = filtered_soho['Linear_Speed'].astype(float)
filtered_soho['Initial_Speed'] = to_numeric(filtered_soho['Initial_Speed'], errors='coerce')
filtered_soho['Initial_Speed'] = filtered_soho['Initial_Speed'].astype(float)
filtered_soho['Final_Speed'] = to_numeric(filtered_soho['Final_Speed'], errors='coerce')
filtered_soho['Final_Speed'] = filtered_soho['Final_Speed'].astype(float)
filtered_soho['Speed_20Rs'] = to_numeric(filtered_soho['Speed_20Rs'], errors='coerce')
filtered_soho['Speed_20Rs'] = filtered_soho['Speed_20Rs'].astype(float)
filtered_soho['Accel'] = to_numeric(filtered_soho['Accel'], errors='coerce')
filtered_soho['Width'] = filtered_soho['Width'].astype(float)
filtered_soho['MPA'] = filtered_soho['MPA'].astype(float)

print(filtered_soho.dtypes)

# In[]: --- 
# Filter the data 
filtered_soho_lv1 = filtered_soho[(filtered_soho['Linear_Speed'].values >= 700) & (filtered_soho['Width'].values >= 120)]

# Randomly select number of rows 
num_rand_samples = 40
sample = filtered_soho_lv1.sample(n=num_rand_samples)

sample = sample.set_index('Datetime')

# sample.to_excel('Random_'+str(num_rand_samples)+'_CMEs.xlsx') # already done 

# In[]: --- 
# ======================== INTEGRATE CME WITH OMNI DATA ============================== 

# In[]: G2001 Model -- it can be used in a for-loop for many events 
def G2001(CME_datetime, CME_speed):
    '''
    This function is to calculate the estimate transit time of CMEs, from Gopalswamy et al. (2001). 

    Parameters
    ----------
    CME_datetime : datetime object 
        The format is 'yyyy,m,d,H,M,S'. 
    
    CME_speed : 
        float number, in 'km/s'. 

    Returns
    -------
    Arrival time of the CME (datetime object). 

    '''
    AU = 149599999.99979659915  # Sun-Earth distance in km 
    d = 0.76 * AU               # cessation distance in km 
    
    a_calculated = (-10**-3) * ((0.0054 * CME_speed) - 2.2) # in km/s**2 
    squareRoot = np.sqrt((CME_speed**2) + (2*a_calculated*d))
    
    A = (-CME_speed + squareRoot) / a_calculated
    B = (AU - d) / squareRoot
    C = (A + B) / 60
    
    arrival_datetime = CME_datetime + timedelta(minutes=C)
    
    return arrival_datetime

# In[]: --- Try to build a JSON file structure with these info for all events --- 
print('Predicting the transit time of the CME using G2001 model .. \n')

event_num = 2
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

# # Get a list of columns names of Raw data 
# print('\nRaw columns are:\n', *omni_data.columns, sep= '\n')

# # Leave these columns only 
# fdata = data.filter(data[['F',
#                           'BX_GSE',
#                           'BY_GSM',
#                           'BZ_GSM',
#                           'flow_speed',
#                           'proton_density',
#                           'Beta',
#                           'Pressure',
#                           'SYM_H']])

# # Get a list of columns names of Filtered data 
# print('\nFiltered columns are:\n', *fdata.columns, sep='\n')
# labels = fdata.columns.values.tolist()

# In[]: --- 
plt.figure(figsize=(15,3), dpi=300)
omni_data['SYM_H'].plot()
plt.xlabel('Date')
plt.ylabel('Dst (nT)')
plt.show()

# In[]: --- 
# Assign a threshold of the Dst (nT) to look for geomagnetic storms 
threshold = np.average(omni_data['SYM_H'])

print('\nDefine timestamps where Dst <', threshold, 'nT')

fig, ax = plt.subplots(1, 1, figsize=(15,3), dpi=300)
ax.plot(omni_data['SYM_H'], label='SYM-H')

# Find Geomagnetic Storms in Data 
for i in range(1, len(omni_data)):
    if omni_data['SYM_H'][i] <= threshold:
        ax.axvline(omni_data['SYM_H'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        ax.axvspan(omni_data['SYM_H'].index[i-1], omni_data['SYM_H'].index[i], facecolor='#FFCC66', alpha=0.7)

ax.axvline(arrival_datetime, label='G2001', color='green', alpha=0.7, linewidth=3, linestyle='--')

ax.legend(loc='upper right', frameon=False, prop={'size': 10})
ax.set_xlim([omni_data.index[0], omni_data.index[-1]])
plt.xlabel('Date')
plt.ylabel('Dst (nT)')
fig.tight_layout()

# Select all the rows which satisfies the criteria 
# convert the collection of index labels to list 
Index_label_Dst = omni_data[omni_data['SYM_H'] <= threshold].index.tolist()

print('\nNaNs?', omni_data.isnull().values.any())

# ============================================= 
# >>> APPLY STEP NO. 6 FROM MY NOTEBOOK <<< 
# ================================================ 
# >>> APPEND (Br, phi, theta) IN THE DATASET <<< 
# ================================================== 

fig, axs = plt.subplots(7, 1, figsize=(15,10), dpi=300, sharex=True)

axs[0].plot(omni_data['F'], label='Avg Mag Field')
axs[0].set_ylabel('Bt (nT)')

axs[1].plot(omni_data['BX_GSE'], label='Bx GSE')
axs[1].plot(omni_data['BY_GSM'], label='By GSM')
axs[1].plot(omni_data['BZ_GSM'], label='Bz GSM')
axs[1].set_ylabel('nT')

axs[2].plot(omni_data['flow_speed'], label='Flow Speed')
axs[2].set_ylabel('Flow Speed\n(km/s)')

axs[3].plot(omni_data['proton_density'], label='Proton Density')
axs[3].set_ylabel('Proton Density\n(#/cm**3)')

axs[4].plot(omni_data['Pressure'], label='Pressure')
axs[4].set_ylabel('Pressure\n(nPa)')

axs[5].plot(omni_data['Beta'], label='Beta')
axs[5].set_ylabel('Plasma\nBeta')

axs[6].plot(omni_data['SYM_H'], label='SYM-H')
axs[6].set_ylabel('Dst (nT)')

# Find Geomagnetic Storms in Data 
for i in range(1, len(omni_data)):
    if omni_data['SYM_H'][i] < threshold:

        axs[0].axvline(omni_data['F'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[0].axvspan(omni_data['F'].index[i-1], omni_data['F'].index[i], facecolor='#FFCC66', alpha=0.7)
        
        axs[1].axvline(omni_data['BX_GSE'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[1].axvspan(omni_data['BX_GSE'].index[i-1], omni_data['BX_GSE'].index[i], facecolor='#FFCC66', alpha=0.7)
        
        axs[2].axvline(omni_data['flow_speed'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[2].axvspan(omni_data['flow_speed'].index[i-1], omni_data['flow_speed'].index[i], facecolor='#FFCC66', alpha=0.7)

        axs[3].axvline(omni_data['proton_density'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[3].axvspan(omni_data['proton_density'].index[i-1], omni_data['proton_density'].index[i], facecolor='#FFCC66', alpha=0.7)

        axs[4].axvline(omni_data['Pressure'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[4].axvspan(omni_data['Pressure'].index[i-1], omni_data['Pressure'].index[i], facecolor='#FFCC66', alpha=0.7)

        axs[5].axvline(omni_data['Beta'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[5].axvspan(omni_data['Beta'].index[i-1], omni_data['Beta'].index[i], facecolor='#FFCC66', alpha=0.7)        
        
        axs[6].axvline(omni_data['SYM_H'].index[i], color='tomato', alpha=0.1, linewidth=1, linestyle='--')
        axs[6].axvspan(omni_data['SYM_H'].index[i-1], omni_data['SYM_H'].index[i], facecolor='#FFCC66', alpha=0.7)

for ax in axs:
    ax.axvline(arrival_datetime, label='G2001', color='black', alpha=0.7, linewidth=3, linestyle='--')
    ax.legend(loc='upper right', frameon=False, prop={'size': 10})
    ax.set_xlim([omni_data.index[0], omni_data.index[-1]])
plt.xlabel('Date')
fig.tight_layout()
# plt.show()

# ========================================== 
st = str(omni_data.index[0].year)+str(omni_data.index[0].month)+str(omni_data.index[0].day)+str(omni_data.index[0].hour)+str(omni_data.index[0].minute)+str(omni_data.index[0].second)
en = str(omni_data.index[-1].year)+str(omni_data.index[-1].month)+str(omni_data.index[-1].day)+str(omni_data.index[-1].hour)+str(omni_data.index[-1].minute)+str(omni_data.index[-1].second)
plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_Data_'+st+'--'+en+'.png'))

# In[] --- 
''' 
 RUN A FOR-LOOP THAT CHECK THE MIN. DIFF. 
 BETWEEN 'arrival_datetime' & 'Index_label_<ANY_SW_PARAM.>' 
 & STORE THE 'idx' VALUES FOR EACH SW PARAM. AS A LIST 
 & FINALLY, TAKE THE AVG. OF THAT LIST, TO BE 
 THE PROPABLE ARRIVAL TIME OF THE EVENT. 
 ''' 

[val, idx] = min(abs(arrival_datetime - Index_label_Dst))



# In[] --- 






# In[] --- 





# In[]: --- 


# In[] --- 




# In[] --- 




# In[] --- 








# In[] --- 






# In[] --- 



# In[] --- 









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

