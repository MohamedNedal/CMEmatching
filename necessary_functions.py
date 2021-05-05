# -*- coding: utf-8 -*-
""" 
This file contains all the customized functions 
to be imported in the main file. 

""" 

import math
from statistics import mean
import numpy as np
from pandas import Timestamp
from datetime import timedelta
import heliopy.data.omni as omni
import trendet
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

# In[]: --- 
def get_omni_hr(start_datetime, end_datetime):
    '''
    This function to download OMNI data from CDAWEB server. 
    
    Type: High Resolution OMNI (HRO) 
    Source: https://cdaweb.gsfc.nasa.gov/misc/NotesO.html#OMNI2_H0_MRG1HR 
    Decsribtion: https://omniweb.gsfc.nasa.gov/html/ow_data.html 
    
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
    
    omni_data = omni.h0_mrg1hr(start_datetime, end_datetime)
    data = omni_data.data
    
    # Consider only these columns  
    fdata = data.filter(data[[
        
        # --- INTERPLANETARY MAGNETIC FIELD --- 
        
        'ABS_B1800', # 1AU IP Average B Field Magnitude (nT) 
        'F1800', # 1AU IP Magnitude of average field vector (nT) 
        'THETA_AV1800', # 1AU IP Latitude/Theta of average B vector (deg) 
        'PHI_AV1800', # 1AU IP Longitude/Phi of average B vector (deg) 
        'BX_GSE1800', # 1AU IP Bx (nT), GSE 
        'BY_GSE1800', # 1AU IP By (nT), GSE 
        'BZ_GSE1800', # 1AU IP Bz (nT), GSE 
        
        # --- SOLAR WIND PLASMA --- 
        
        'T1800', # 1AU IP Plasma Temperature (K) 
        'N1800', # 1AU IP Ion number density (n/cc) 
        'V1800', # 1AU IP plasma flow speed (km/s) 
        'PHI$V1800', # 1AU IP plasma flow direction longitude (deg), phi 
        'THETA$V1800', # 1AU IP plasma flow direction latitude (deg), theta 
        'Ratio1800', # 1AU IP Alpha/proton ratio 
        'Pressure1800', # 1AU IP Flow pressure (nPa) 
        'E1800', # 1AU IP Electric Field (mV/m) 
        'Beta1800', # 1AU IP Plasma beta 
        'Mach_num1800', # 1AU IP Alfven mach number 
        'Mgs_mach_num1800', # 1AU IP Magnetosonic mach number 
        
        # --- EERGETIC PARTICLES --- 
        
        'PR$FLX_11800', # 1AU Proton flux > 1 MeV, 1/(SQcm-ster-s) 
        'PR$FLX_21800', # 1AU Proton flux >2 MeV (1/(SQcm-ster-s)) [PR-FLX_21800]
        'PR$FLX_41800', # 1AU Proton flux >4 MeV (1/(SQcm-ster-s)) 
        'PR$FLX_101800', # 1AU Proton flux >10 MeV (1/(SQcm-ster-s)) 
        'PR$FLX_301800', # 1AU Proton flux >30 MeV (1/(SQcm-ster-s)) 
        'PR$FLX_601800', # 1AU Proton flux >60 MeV (1/(SQcm-ster-s)) 
        
        'F10_INDEX1800', # F10.7 - Daily 10.7 cm solar radio flux, units: 10**(-22) Joules/second/square-meter/Hertz, from NGDC 
        'DST1800', # Dst - 1-hour Dst index from WDC Kyoto 
        'Solar_Lyman_alpha1800', # Solar Lyman-alpha index 
        'Proton_QI1800', # Solar wind (magnetic energy density)/(kinetic energy density) 
        
        # --- SIGMA VALUES --- 
        'SIGMA$ABS_B1800', # RMS deviation of average B magnitude (nT)                               
        'SIGMA$Bx1800', # RMS deviation Bx (nT), GSE 
        'SIGMA$By1800', # RMS deviation By (nT), GSE 
        'SIGMA$Bz1800', # RMS deviation Bz (nT), GSE 
        'SIGMA$T1800', # RMS deviation of plasma temperature (deg k) 
        'SIGMA$N1800', # RMS deviation of ion number density (n/cc) 
        'SIGMA$V1800', # RMS deviation in plasma flow velocity (km/s) 
        'SIGMA$PHI$V1800', # RMS deviation in plasma flow direction longitude (deg), phi 
        'SIGMA$THETA$V1800', # RMS deviation in plasma flow direction latitude (deg), theta 
        'SIGMA$ratio1800' # RMS deviation alpha/proton ratio 
        ]])
    
    # Get a list of columns names of Filtered data 
    # print('\nFiltered columns are:\n', *fdata.columns, sep='\n')
    
    return fdata

# In[]: --- 
def get_omni_min(start_datetime, end_datetime):

    ''' 
    # High Resolution OMNI (HRO)
    # Source: https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/ 
    # Decsribtion: https://omniweb.gsfc.nasa.gov/html/HROdocum.html 

    Parameters
    ----------
    start_datetime : datetime object
        The format is 'yyyy,m,d,H,M,S'. 
    end_datetime : datetime object
        The format is 'yyyy,m,d,H,M,S'. 

    Returns
    -------
    fdata : Dataframe 
        The OMNI data within that period. 

    '''
    
    omni_data = omni.hro2_1min(start_datetime, end_datetime)
    data = omni_data.data
    
    # Consider only these columns  
    fdata = data.filter(data[[
        
        # --- INTERPLANETARY MAGNETIC FIELD --- 
        
        'F', # IMF magnitude average (nT) 
        'BX_GSE', # IMF x-comp. (nT) 
        'BY_GSM', # IMF y-comp. (nT) 
        'BZ_GSM', # IMF z-comp. (nT) 
        'E', # Electric field (mV/m) 
        'SYM_H', # High-resolution Dst index (nT) 
        
        # --- SOLAR WIND PLASMA --- 
        
        'flow_speed', # Solar wind speed (km/s) 
        'Vx', # Solar wind x-speed (km/s) 
        'Vy', # Solar wind y-speed (km/s) 
        'Vz', # Solar wind z-speed (km/s) 
        'proton_density', # Proton density (n/cc) 
        'T', # Plasma Temperature (K) 
        'NaNp_Ratio', # Alpha/proton ratio 
        'Pressure', # Flow pressure (nPa) 
        'Beta', # Plasma beta 
        'Mach_num', # Alfven mach number 
        'Mgs_mach_num' # Magnetosonic mach number 
        ]])
    
    return fdata

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

# In[]: --- 

def using_trendet_hr(omni_data_raw):
    ''' 
    Applying the trendet package on hourly-OMNI data 

    Parameters
    ----------
    omni_data_raw : Dataframe table of OMNI hourly-data 

    Returns
    -------
    Plot and timestamp 

    ''' 
    
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'DST1800']])
    omni_data = omni_data.astype('float64')
    
    # Subplot for V, n, T, Dst
    sns.set(style='darkgrid')
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15,7))
    fig.suptitle('Comparison between trends for OMNI hourly data')
    
    # Plasma speed 
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'DST1800']]).astype('float64')
    v = trendet.identify_df_trends(df=omni_data, column='V1800', window_size=3, identify='both')
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
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'DST1800']]).astype('float64')
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
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'DST1800']]).astype('float64')
    T = trendet.identify_df_trends(df=omni_data, column='T1800', window_size=3, identify='both')
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
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'DST1800']]).astype('float64')
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
    
    return intersection[-1]

# In[]: --- 

def using_trendet_min(omni_data_raw):
    ''' 
    Applying the trendet package on 1-minute-OMNI data 

    Parameters
    ----------
    omni_data_raw : Dataframe table of OMNI hourly-data 

    Returns
    -------
    Plot and timestamp 

    ''' 
    
    omni_data = omni_data_raw.filter(omni_data_raw[['F', 'BZ_GSM', 'T', 'proton_density', 'flow_speed', 'Pressure', 'SYM_H']])
    omni_data = omni_data.astype('float64')
    
    # Subplot for V, n, T, Dst
    sns.set(style='darkgrid')
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15,7))
    fig.suptitle('Comparison between trends for OMNI hourly data')
    
    # Plasma speed 
    omni_data = omni_data_raw.filter(omni_data_raw[['F', 'BZ_GSM', 'T', 'proton_density', 'flow_speed', 'Pressure', 'SYM_H']]).astype('float64')
    v = trendet.identify_df_trends(df=omni_data, column='flow_speed', window_size=3, identify='both')
    v.reset_index(inplace=True)
    sns.lineplot(ax=axes[0], x=v['Time'], y=v['flow_speed'])
    labels = v['Up Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[0], x=v[v['Up Trend'] == label]['Time'], 
                     y=v[v['Up Trend'] == label]['flow_speed'], color='green')
        axes[0].axvspan(v[v['Up Trend'] == label]['Time'].iloc[0], 
                        v[v['Up Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='green')   
    labels = v['Down Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[0], x=v[v['Down Trend'] == label]['Time'], 
                     y=v[v['Down Trend'] == label]['flow_speed'], color='red')
        axes[0].axvspan(v[v['Down Trend'] == label]['Time'].iloc[0], 
                        v[v['Down Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='red')
    axes[0].set(xlim=(v['Time'].iloc[0], v['Time'].iloc[-1]))
    axes[0].set_ylabel('V (km/s)')
    
    # Plasma density 
    omni_data = omni_data_raw.filter(omni_data_raw[['F', 'BZ_GSM', 'T', 'proton_density', 'flow_speed', 'Pressure', 'SYM_H']]).astype('float64')
    n = trendet.identify_df_trends(df=omni_data, column='proton_density', window_size=3, identify='both')
    n.reset_index(inplace=True)
    sns.lineplot(ax=axes[1], x=n['Time'], y=n['proton_density'])
    labels = n['Up Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[1], x=n[n['Up Trend'] == label]['Time'], 
                     y=n[n['Up Trend'] == label]['proton_density'], color='green')
        axes[1].axvspan(n[n['Up Trend'] == label]['Time'].iloc[0], 
                        n[n['Up Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='green')   
    labels = n['Down Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[1], x=n[n['Down Trend'] == label]['Time'], 
                     y=n[n['Down Trend'] == label]['proton_density'], color='red')
        axes[1].axvspan(n[n['Down Trend'] == label]['Time'].iloc[0], 
                        n[n['Down Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='red')
    axes[1].set(xlim=(n['Time'].iloc[0], n['Time'].iloc[-1]))
    axes[1].set_ylabel('n (#/cm3)')
    
    # Plasma temperature 
    omni_data = omni_data_raw.filter(omni_data_raw[['F', 'BZ_GSM', 'T', 'proton_density', 'flow_speed', 'Pressure', 'SYM_H']]).astype('float64')
    T = trendet.identify_df_trends(df=omni_data, column='T', window_size=3, identify='both')
    T.reset_index(inplace=True)
    sns.lineplot(ax=axes[2], x=T['Time'], y=T['T'])
    labels = T['Up Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[2], x=T[T['Up Trend'] == label]['Time'], 
                     y=T[T['Up Trend'] == label]['T'], color='green')
        axes[2].axvspan(T[T['Up Trend'] == label]['Time'].iloc[0], 
                        T[T['Up Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='green')   
    labels = T['Down Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[2], x=T[T['Down Trend'] == label]['Time'], 
                     y=T[T['Down Trend'] == label]['T'], color='red')
        axes[2].axvspan(T[T['Down Trend'] == label]['Time'].iloc[0], 
                        T[T['Down Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='red')
    axes[2].set(xlim=(T['Time'].iloc[0], T['Time'].iloc[-1]))
    axes[2].set_yscale('log')
    axes[2].set_ylabel('T (K)')
    
    # Dst 
    omni_data = omni_data_raw.filter(omni_data_raw[['F', 'BZ_GSM', 'T', 'proton_density', 'flow_speed', 'Pressure', 'SYM_H']]).astype('float64')
    dst = trendet.identify_df_trends(df=omni_data, column='SYM_H', window_size=3, identify='both')
    dst.reset_index(inplace=True)
    sns.lineplot(ax=axes[3], x=dst['Time'], y=dst['SYM_H'])
    labels = dst['Up Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[3], x=dst[dst['Up Trend'] == label]['Time'], 
                     y=dst[dst['Up Trend'] == label]['SYM_H'], color='green')
        axes[3].axvspan(dst[dst['Up Trend'] == label]['Time'].iloc[0], 
                        dst[dst['Up Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='green')   
    labels = dst['Down Trend'].dropna().unique().tolist()
    for label in labels:
        sns.lineplot(ax=axes[3], x=dst[dst['Down Trend'] == label]['Time'], 
                     y=dst[dst['Down Trend'] == label]['SYM_H'], color='red')
        axes[3].axvspan(dst[dst['Down Trend'] == label]['Time'].iloc[0], 
                        dst[dst['Down Trend'] == label]['Time'].iloc[-1],
                        alpha=0.2, color='red')
    axes[3].set(xlim=(dst['Time'].iloc[0], dst['Time'].iloc[-1]))
    axes[3].set_ylabel('Dst (nT)')
    
    # NEXT: FIND THE TIME INDEX AT WHICH THE D-TREND OF DST AND N = U-TREND OF V AND T 
    omni_trends = DataFrame({'Datetime': omni_data['Time'], 
                             'V': omni_data['flow_speed'], 
                             'n': omni_data['proton_density'], 
                             'T': omni_data['T'], 
                             'Dst': omni_data['SYM_H'], 
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
    
    return intersection[-1]

# In[]: --- 

def PA(omni_data, sample, event_num, arrival_datetime):
    
    '''
    Parametric Approach (PA) 

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
    AVG : Timestamp 
        Average estimated ICME arrival time. 
    PA_tran_time_hours : Float number 
        Estimated transit time of the CME in hours. 

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
            AVG_lst = []
            
            if mean(omni_data['V1800']) > 500:
                # for the high-speed wind 
                Texp = (0.5*((0.031*omni_data['V1800']) - 4.39)**2) # try without the 0.5 -- since it should be Tp/Texp < 0.5 
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
                    
                    AVG_lst.append(Timestamp((T_RED.value + T_BLACK.value + T_GREEN.value)/3.0))


            # APPEND THE OUTPUT TRANSIT TIME WITH THE CME INFO 
            try:
                PA_est_trans_time = Index_label_Dst[dt_G2001_idxLabel.index(min(dt_G2001_idxLabel))] - sample.CME_Datetime[event_num]
                PA_est_trans_time
            except IndexError as idx_err:
                print(idx_err)

            
            AVG_lst = DataFrame(AVG_lst, columns={'timestamps'})
            
            # Taking the average of those timestamps 
            AVG = Timestamp(np.nanmean([tsp.value for tsp in AVG_lst['timestamps']]))
            
            PA_tran_time_hours = (PA_est_trans_time.components.days * 24) + (PA_est_trans_time.components.minutes / 60) + (PA_est_trans_time.components.seconds / 3600)
    
    
    return AVG, PA_tran_time_hours

# In[]: --- 

def ChP(omni_data, sample, event_num):

    '''
    Change Point (ChP) Detection: Dynamic Programming Search Method 

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
    
    import ruptures as rpt

    # Filter the OMNI columns 
    # omni_col_lst = omni_data.columns.values.tolist()
    omni_data = omni_data.filter(omni_data.columns[[1,6,7,8,9,13,25]])
    
    col_lst = omni_data.columns
    # number of subplots
    n_subpl = len(omni_data.columns) # or omni_data.shape[1] 
    # give figure a name to refer to it later 
    figname = '1-hr OMNI data'
    
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
        # plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_chp_for_CME_'+str(event_num)+'.png'))
        plt.show()

        
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
    # plt.savefig(os.path.join(save_path, 'Output_plots' + '/', 'OMNI_for_CME_'+str(event_num)+'.png'))
    plt.show()
    
    
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
    
    
    return chp_ICME_est_arrival_time, est_tran_time_chp_method


