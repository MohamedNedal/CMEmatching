# -*- coding: utf-8 -*-
""" 
Applying the trendet package on hourly-OMNI data 
Created on Sat Apr  3 19:20:37 2021 
@author: Mohamed Nedal 
""" 
import trendet
import seaborn as sns
from statistics import mean
from pandas import DataFrame
import matplotlib.pyplot as plt

def trendet_hr(omni_data_raw):
    ''' 
    Parameters
    ----------
    omni_data_raw : Dataframe table of OMNI hourly-data 

    Returns
    -------
    Plot and timestamp 
    ''' 
    
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 
                                                    'N1800', 'V1800', 'Pressure1800', 
                                                    'Beta1800', 'DST1800']]).astype('float64')
    # omni_data = omni_data.astype('float64')

    # Calculating half the expected solar wind temperature (0.5Texp) 
    ''' 
    define the Texp as mentioned in: 
        Lopez, R. E., & Freeman, J. W. (1986). 
        Solar wind proton temperature‐velocity relationship. 
        Journal of Geophysical Research: Space Physics, 91(A2), 1701-1705. 
    ''' 
    # To check this condition Tp/Texp < 0.5 --> signature of ICME (MC) 
    if mean(omni_data['V1800']) > 500:
        # for the high-speed wind 
        Texp = ((0.031 * omni_data['V1800']) - 4.39)**2
    else:
        # for the high-speed wind 
        Texp = ((0.77 * omni_data['V1800']) - 265)**2
    Texp.rename('Texp', inplace=True)
    ICME_duration = []
    for i in range(len(Texp)):
        if omni_data['T1800'].values[i]/Texp.values[i] < 0.5:
            ICME_duration.append(Texp.index[i])
    
    # # plot Tp and Texp 
    # plt.figure(figsize=(15,3))
    # plt.plot(omni_data['T1800'], label='Tp')
    # plt.plot(Texp, 'r', label='Texp')    
    # plt.axvline(ICME_duration[0], color='g', linewidth=2, linestyle='--', label='start')
    # plt.axvline(ICME_duration[-1], color='y', linewidth=2, linestyle='--', label='end')
    # plt.yscale('log')
    # plt.legend(loc=0, frameon=False)
    # plt.xlabel('Date')
    # plt.ylabel('T (K)')
    # plt.xlim(Texp.index[24], Texp.index[-100])
    # plt.tight_layout()
    # plt.show()
    
    # # plot Beta 
    # plt.figure(figsize=(15,3))
    # plt.plot(omni_data['Beta1800'])
    # plt.axhline(y=1, color='r', linestyle='--')
    # plt.yscale('log')
    # plt.xlim(Texp.index[24], Texp.index[-100])
    # plt.tight_layout()
    # plt.show()

    # Subplot for V, n, T, Dst
    sns.set(style='darkgrid')
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15,7))
    fig.suptitle('Comparison between trends for OMNI hourly data')
    
    # Plasma speed 
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'Beta1800', 'DST1800']]).astype('float64')
    v = trendet.identify_df_trends(df=omni_data, column='V1800', window_size=3, identify='both')
    v.reset_index(inplace=True)
    sns.lineplot(ax=axes[0], x=v['Time'], y=v['V1800'])
    try:
        if 'Up Trend' in v:
            labels = v['Up Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[0], x=v[v['Up Trend'] == label]['Time'], 
                             y=v[v['Up Trend'] == label]['V1800'], color='green')
                axes[0].axvspan(v[v['Up Trend'] == label]['Time'].iloc[0], 
                                v[v['Up Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='green')
        else:
            print("\n'Up Trend' not found in v\n")                
        if 'Down Trend' in v:
            labels = v['Down Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[0], x=v[v['Down Trend'] == label]['Time'], 
                             y=v[v['Down Trend'] == label]['V1800'], color='red')
                axes[0].axvspan(v[v['Down Trend'] == label]['Time'].iloc[0], 
                                v[v['Down Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='red')
        else:
            print("\n'Down Trend' not found in v\n")                
    except KeyError as ke:
        print(ke)
    axes[0].set(xlim=(v['Time'].iloc[0], v['Time'].iloc[-1]))
    axes[0].set_ylabel('V (km/s)')
    
    # Plasma density 
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'Beta1800', 'DST1800']]).astype('float64')
    n = trendet.identify_df_trends(df=omni_data, column='N1800', window_size=3, identify='both')
    n.reset_index(inplace=True)
    sns.lineplot(ax=axes[1], x=n['Time'], y=n['N1800'])
    try:
        if 'Up Trend' in n:
            labels = n['Up Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[1], x=n[n['Up Trend'] == label]['Time'], 
                             y=n[n['Up Trend'] == label]['N1800'], color='green')
                axes[1].axvspan(n[n['Up Trend'] == label]['Time'].iloc[0], 
                                n[n['Up Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='green')
        else:
            print("\n'Up Trend' not found in n\n")
            
        if 'Down Trend' in n:
            labels = n['Down Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[1], x=n[n['Down Trend'] == label]['Time'], 
                             y=n[n['Down Trend'] == label]['N1800'], color='red')
                axes[1].axvspan(n[n['Down Trend'] == label]['Time'].iloc[0], 
                                n[n['Down Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='red')
        else:
            print("\n'Down Trend' not found in n\n")
    except KeyError as ke:
        print(ke)
    axes[1].set(xlim=(n['Time'].iloc[0], n['Time'].iloc[-1]))
    axes[1].set_ylabel('n (#/cm3)')
    
    # Plasma temperature 
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'Beta1800', 'DST1800']]).astype('float64')
    T = trendet.identify_df_trends(df=omni_data, column='T1800', window_size=3, identify='both')
    T.reset_index(inplace=True)
    sns.lineplot(ax=axes[2], x=T['Time'], y=T['T1800'])
    try:
        if 'Up Trend' in T:
            labels = T['Up Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[2], x=T[T['Up Trend'] == label]['Time'], 
                             y=T[T['Up Trend'] == label]['T1800'], color='green')
                axes[2].axvspan(T[T['Up Trend'] == label]['Time'].iloc[0], 
                                T[T['Up Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='green')
        else:
            print("\n'Up Trend' not found in T\n")
                
        if 'Down Trend' in T:
            labels = T['Down Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[2], x=T[T['Down Trend'] == label]['Time'], 
                             y=T[T['Down Trend'] == label]['T1800'], color='red')
                axes[2].axvspan(T[T['Down Trend'] == label]['Time'].iloc[0], 
                                T[T['Down Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='red')
        else:
            print("\n'Down Trend' not found in T\n")                
    except KeyError as ke:
        print(ke)
    axes[2].set(xlim=(T['Time'].iloc[0], T['Time'].iloc[-1]))
    axes[2].set_yscale('log')
    axes[2].set_ylabel('T (K)')
    
    # Dst 
    omni_data = omni_data_raw.filter(omni_data_raw[['F1800', 'BZ_GSE1800', 'T1800', 'N1800', 'V1800', 'Pressure1800', 'Beta1800', 'DST1800']]).astype('float64')
    dst = trendet.identify_df_trends(df=omni_data, column='DST1800', window_size=3, identify='both')
    dst.reset_index(inplace=True)
    sns.lineplot(ax=axes[3], x=dst['Time'], y=dst['DST1800'])
    try:
        if 'Up Trend' in dst:
            labels = dst['Up Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[3], x=dst[dst['Up Trend'] == label]['Time'], 
                             y=dst[dst['Up Trend'] == label]['DST1800'], color='green')
                axes[3].axvspan(dst[dst['Up Trend'] == label]['Time'].iloc[0], 
                                dst[dst['Up Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='green')
        else:
            print("\n'Up Trend' not found in dst\n")
            
        if 'Down Trend' in dst:
            labels = dst['Down Trend'].dropna().unique().tolist()
            for label in labels:
                sns.lineplot(ax=axes[3], x=dst[dst['Down Trend'] == label]['Time'], 
                             y=dst[dst['Down Trend'] == label]['DST1800'], color='red')
                axes[3].axvspan(dst[dst['Down Trend'] == label]['Time'].iloc[0], 
                                dst[dst['Down Trend'] == label]['Time'].iloc[-1],
                                alpha=0.2, color='red')
        else:
            print("\n'Down Trend' not found in dst\n")            
    except KeyError as ke:
        print(ke)
        print("\ndst['Down Trend'] not found")
    axes[3].set(xlim=(dst['Time'].iloc[0], dst['Time'].iloc[-1]))
    axes[3].set_ylabel('Dst (nT)')
    
    # NEXT: FIND THE TIME INDEX AT WHICH THE D-TREND OF DST AND N = U-TREND OF V AND T 
    omni_trends = DataFrame({'Datetime': omni_data['Time'], 
                             'V': omni_data['V1800'], 
                             'n': omni_data['N1800'], 
                             'T': omni_data['T1800'], 
                             'Dst': omni_data['DST1800']})

    if 'Up Trend' in v: omni_trends['U-trend_V'] = v['Up Trend']
    else: print("'Up Trend' is missing from v")
    
    if 'Down Trend' in v: omni_trends['D-trend_V'] = v['Down Trend']
    else: print("'Down Trend' is missing from v")
    
    if 'Up Trend' in n: omni_trends['U-trend_n'] = n['Up Trend']
    else: print("'Up Trend' is missing from n")
    
    if 'Down Trend' in n: omni_trends['D-trend_n'] = n['Down Trend']
    else: print("'Down Trend' is missing from n")
    
    if 'Up Trend' in T: omni_trends['U-trend_T'] = T['Up Trend']
    else: print("'Up Trend' is missing from T")
    
    if 'Down Trend' in T: omni_trends['D-trend_T'] = T['Down Trend']
    else: print("'Down Trend' is missing from T")
    
    if 'Up Trend' in dst: omni_trends['U-trend_dst'] = dst['Up Trend']
    else: print("'Up Trend' is missing from dst")
    
    if 'Down Trend' in dst: omni_trends['D-trend_dst'] = dst['Down Trend']
    else: print("'Down Trend' is missing from dst")

    omni_trends = omni_trends.set_index('Datetime')
    
    # check the existence of columns 
    if 'U-trend_V' in omni_trends and 'U-trend_n' in omni_trends:
        V_n = omni_trends[['U-trend_V', 'U-trend_n']]
        V_n.dropna(inplace=True)
    
    if 'U-trend_V' in omni_trends and 'U-trend_T' in omni_trends:
        V_T = omni_trends[['U-trend_V', 'U-trend_T']]
        V_T.dropna(inplace=True)
    
    if 'U-trend_n' in omni_trends and 'U-trend_T' in omni_trends:
        n_T = omni_trends[['U-trend_n', 'U-trend_T']]
        n_T.dropna(inplace=True)
    
    if 'U-trend_V' in omni_trends and 'U-trend_Dst' in omni_trends:
        V_Dst = omni_trends[['U-trend_V', 'U-trend_Dst']]
        V_Dst.dropna(inplace=True)

    if 'U-trend_n' in omni_trends and 'U-trend_Dst' in omni_trends:
        n_Dst = omni_trends[['n-trend_n', 'U-trend_Dst']]
        n_Dst.dropna(inplace=True)
    
    if 'U-trend_T' in omni_trends and 'U-trend_Dst' in omni_trends:
        T_Dst = omni_trends[['U-trend_T', 'U-trend_Dst']]
        T_Dst.dropna(inplace=True)
    
    # Find the intersection timestamp 
    # if V_n != [] and V_T != [] and n_T != [] and V_Dst != [] and n_Dst != [] and T_Dst != []:
    try: 
        V_Dst
        if all(var is not None for var in [V_n, V_T, n_T, V_Dst, n_Dst, T_Dst]):
            intersection = set(set(set(set(set(set(V_n.index).intersection(V_T.index)).intersection(n_T.index)).intersection(V_Dst.index)).intersection(n_Dst.index)).intersection(T_Dst.index)).intersection(ICME_duration)
            intersection = [*intersection,]
        
        # Plot a dashed black line representing the estimated CME arrival time 
        for ax in axes:
            ax.axvline(intersection[-1], color='k', linewidth=2, linestyle='--')
            
        else: print('\nOne variable required for the intersection is missing\n')
    
    except NameError: print('\nV_Dst not found\n')

    fig.tight_layout()
    plt.show()
    
    try: return intersection[-1]
    except NameError as NE: print(NE, "\n'intersection' not found")
