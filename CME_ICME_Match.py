"""
Plotting OMNI Data
==================

Importing and plotting data from OMNI Web Interface.
OMNI provides interspersed data from various spacecrafts.
There are 55 variables provided in the OMNI Data Import.
"""
# import numpy as np
from pandas import read_excel, DataFrame, Timestamp
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

# Try to build a JSON file structure with these info for all events 
print('\nPredicting the transit time of the CMEs using the G2001 model .. \n')

print('For more info about the G2001 model, check this paper:\nGopalswamy, N., Lara, A., Yashiro, S., Kaiser, M. L., and Howard,\nR. A.: Predicting the 1-AU arrival times of coronal mass ejections,\nJ. Geophys. Res., 106, 29 207, 2001a.\n')
print('And this paper:\nOwens, M., & Cargill, P. (2004, January).\nPredictions of the arrival time of Coronal Mass Ejections at 1AU:\nan analysis of the causes of errors.\nIn Annales Geophysicae (Vol. 22, No. 2, pp. 661-671). Copernicus GmbH.\n')
print('-------------------------------------------------------')

# In[]: Drop rows (CME events) that have problems with their respective OMNI data 
sample.drop([sample.index[4], sample.index[14]], inplace=True)

# In[]: --- 
# FINALIZE THE OUTPUT 
cols = sample.columns

cols = cols.insert(0, 'CME_datetime')
cols = cols.insert(len(cols)+1, 'Transit_time_hrs')
cols = cols.insert(len(cols)+1, 'est_ICME_datetime')

final_table = DataFrame(columns=cols)

for event_num in range(len(sample)):
        
    arrival_datetime = G2001(sample.index[event_num], sample.Linear_Speed[event_num])
    
    dt = arrival_datetime - sample.index[event_num]
    print('CME launced on:', sample.index[event_num])
    print('Estimated arrival time is:', arrival_datetime)
    print('Estimated Transit time is: {} days, {} hours, {} minutes' .format(dt.components.days, 
                                                                          dt.components.hours, 
                                                                          dt.components.minutes))
    print('with mean error of:', 11.04, ' hours')
    print('-------------------------------------------------------\n')
    
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
    
    # Assign a threshold of the Dst (nT) to look for geomagnetic storms 
    threshold = -40
    # Select all the rows which satisfies the criteria 
    # Convert the collection of index labels to list 
    Index_label_Dst = omni_data[omni_data['DST1800'] <= threshold].index.tolist()
    
    if Index_label_Dst == []:
        print('-------------------------------------------------------\n')
        print('Index_label_Dst is empty')
        print('-------------------------------------------------------\n')
    
    else:
        print('Define timestamps where Dst =< ', round(threshold,2), 'nT')
    
        if min(omni_data['DST1800']) <= threshold:
            
            fig, axs = plt.subplots(8, 1, figsize=(15,15), dpi=300, sharex=True)
            
            fig.suptitle('For CME event that is launched on: '+str(sample.index[event_num]))
            
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

            # Find Geomagnetic Storms in Data 
            for i in range(1, len(omni_data)):
                
                if omni_data['DST1800'][i] <= threshold:
                    
                    axs[0].axvspan(omni_data['F1800'].index[i-1], omni_data['F1800'].index[i], facecolor='#FFCC66', alpha=0.5)
                    axs[1].axvspan(omni_data['THETA_AV1800'].index[i-1], omni_data['THETA_AV1800'].index[i], facecolor='#FFCC66', alpha=0.5)
                    axs[2].axvspan(omni_data['BX_GSE1800'].index[i-1], omni_data['BX_GSE1800'].index[i], facecolor='#FFCC66', alpha=0.5)
                    axs[3].axvspan(omni_data['V1800'].index[i-1], omni_data['V1800'].index[i], facecolor='#FFCC66', alpha=0.5)
                    axs[4].axvspan(omni_data['N1800'].index[i-1], omni_data['N1800'].index[i], facecolor='#FFCC66', alpha=0.5)
                    axs[5].axvspan(omni_data['Pressure1800'].index[i-1], omni_data['Pressure1800'].index[i], facecolor='#FFCC66', alpha=0.5)
                    axs[6].axvspan(omni_data['Beta1800'].index[i-1], omni_data['Beta1800'].index[i], facecolor='#FFCC66', alpha=0.5)
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
                        min_Dst_window = min(omni_data['DST1800'].loc[Index_label_Dst[0]:Index_label_Dst[-1]])
                        ax.axvline(omni_data[omni_data['DST1800']==min_Dst_window].index[0], label='Min(Dst)', color='red', alpha=0.7, linewidth=3, linestyle='--')
                        
                        dt_G2001_idxLabel = []
                        for idx in range(len(Index_label_Dst)):
                            dt_G2001_idxLabel.append(abs(arrival_datetime - Index_label_Dst[idx]))
                        
                        # at the index of the min diff betn t_G2001 and t_min(Dst) within the window 
                        ax.axvline(Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))], label='Min(dt)', color='black', alpha=0.7, linewidth=3, linestyle='--')     
                
                        T_RED = omni_data[omni_data['DST1800']==min_Dst_window].index[0]
                        T_BLACK = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))]
                        T_GREEN = arrival_datetime
                        
                        AVG = Timestamp((T_RED.value + T_BLACK.value + T_GREEN.value)/3.0)
                        
                        ax.axvline(AVG, label='AVG_t', color='brown', alpha=0.7, linewidth=3, linestyle='--')
                        
                        ax.legend(loc='upper right', frameon=False, prop={'size': 10})
                        ax.set_xlim([omni_data.index[0], omni_data.index[-1]])
                        
                        plt.xlabel('Date')
                        fig.tight_layout()
                        
                        st = str(omni_data.index[0].year)+str(omni_data.index[0].month)+str(omni_data.index[0].day)+str(omni_data.index[0].hour)+str(omni_data.index[0].minute)+str(omni_data.index[0].second)
                        en = str(omni_data.index[-1].year)+str(omni_data.index[-1].month)+str(omni_data.index[-1].day)+str(omni_data.index[-1].hour)+str(omni_data.index[-1].minute)+str(omni_data.index[-1].second)
                        plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_Data_for_CME_No_'+str(event_num)+'_'+st+'-'+en+'.png'))
                                            
                        # APPEND THE OUTPUT TRANSIT TIME WITH THE CME INFO 
                        est_trans_time = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))] - sample.index[event_num]
                        
                        tran_time_hours = (est_trans_time.components.days * 24) + (est_trans_time.components.minutes / 60) + (est_trans_time.components.seconds / 3600)
                        
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
                        
                        Index_label_Dst, min_Dst_window = [], []


        else:
            print('The OMNI data from '+str(start_datetime)+' to '+str(end_datetime)+' has no Dst value below '+str(threshold)+' nT.')
            print('-------------------------------------------------------\n')

# In[]: PLOT SPEED VS TRANSIT TIME 
plt.figure()
plt.scatter(final_table['Linear_Speed'], final_table['Transit_time_hrs'])
plt.xlabel('Speed (km/s)')
plt.ylabel('Transit time (hrs)')
plt.show()
