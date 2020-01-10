# CME-ICME matching based on solar wind data 
Matching each CME with its counterpart ICME at the near-Earth orbit from the solar wind indices. 

DESCRIBTION 
============ 
SOHOdata.xlsx ----------------> is the set of CMEs obtained from SOHO/LASCO CME catalog. 
omni_Example.txt -------------> is the solar wind data obtained from OMNI database. 
omni.fmt.txt -----------------> is the format of OMNI data. 
semimanual_matching_RUN.m ----> is the main file (Run this). 
cme_icme_V4.m ----------------> is the matching function and it includes a prediction model based on G2001 model (Gopalswamy et al. 2001). 
command_window.txt -----------> is the output report. 
