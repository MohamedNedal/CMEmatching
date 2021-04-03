# -*- coding: utf-8 -*-
""" 
Calculate the expecgted CME arrival time, from Gopalswamy et al. (2001) 
Created on Sat Apr  3 19:16:37 2021 
@author: Mohamed Nedal 
"""
import numpy as np
from datetime import timedelta

def G2001(CME_datetime, CME_speed):
    '''
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
