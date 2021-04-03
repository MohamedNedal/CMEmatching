# -*- coding: utf-8 -*-
""" 
Download OMNI data from CDAWEB server 
    Type: High Resolution OMNI (HRO) 
    Source: https://cdaweb.gsfc.nasa.gov/misc/NotesO.html#OMNI2_H0_MRG1HR 
    Decsribtion: https://omniweb.gsfc.nasa.gov/html/ow_data.html 
Created on Sat Apr  3 19:14:51 2021 
@author: Mohamed Nedal 
"""
import heliopy.data.omni as omni

def get_omni_hr(start_datetime, end_datetime):
    '''
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
        
        # --- ENERGETIC PARTICLES --- 
        
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
