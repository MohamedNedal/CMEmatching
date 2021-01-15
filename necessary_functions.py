# -*- coding: utf-8 -*-
""" 
This file contains all the customized functions 
to be imported in the main file. 

""" 

import math
import numpy as np
from datetime import timedelta
import heliopy.data.omni as omni

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
























