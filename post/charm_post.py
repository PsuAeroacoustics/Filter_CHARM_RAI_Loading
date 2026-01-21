#!/usr/bin/env python3

import os
import numpy as np
from dependencies.pyWopwop.wopwop import *  
from dependencies.pyWopwop.wopwop_io import *  
from scipy.signal import welch
import post.plot_styles as plot_styles
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.',':']
marker = ['o','^']
#%%

def process_wopwop(cases_directory,cases = 'cases.nam'):
    f1 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'pressure')
    f2 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'spl_spectrum')
    apply_to_namelist([f1], cases_directory=cases_directory, cases=cases)

case_name = ["quickROD_NPSI128/quickROD.1PSU-WOPWOP",'quickROD_SDOF_GEOM_OAR15/quickROD.1PSU-WOPWOP']
case_dir = os.path.join(os.getcwd())
mics = [0,4,8]
title = [r"$\phi=23^\circ$",r"$\phi=0^\circ$",r"$\phi=-23^\circ$"]
leg_labs = ['BL','SDOF']
acs_data ={}

for case in case_name:
    
    if not os.path.exists(os.path.join(case_dir,f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(case_dir,case),cases = 'cases.nam')

    acs_data.update({case:import_results_from_wopwop(cases_directory=os.path.join(case_dir,case))})
    acs_data[case]['geometry_values'] = (acs_data[case]['geometry_values'])[mics]
    acs_data[case]['function_values'] = (acs_data[case]['function_values'])[mics]

    theta = np.round(np.arctan2(acs_data[case]['geometry_values'][:,:,0,1],acs_data[case]['geometry_values'][:,:,0,0])*180/np.pi)
    phi = np.arctan2(acs_data[case]['geometry_values'][:,:,0,-1],np.linalg.norm((acs_data[case]['geometry_values'][:,:,0,0],acs_data[case]['geometry_values'][:,:,0,1]),axis = 0))

    nperseg = acs_data[case]['function_values'].shape[-2]
    dt = np.diff(acs_data[case]['function_values'][0,0,:2,0])[0]
    df = (nperseg*dt)**-1
    f,pxx = welch(acs_data[case]['function_values'], fs=dt**-1, window='boxcar', nperseg=nperseg, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-2, average='mean')
    
    acs_data[case].update({'f':f,'pxx':pxx,'theta':theta,'phi':phi})

for theta_iter in range(acs_data[case]['geometry_values'].shape[0]):
    for phi_iter in range(acs_data[case]['geometry_values'].shape[1]):
        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(left = .15,bottom = .15)
        for case_itr,case in enumerate(case_name):
            ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],np.roll(acs_data[case]['function_values'][theta_iter,phi_iter,:,-1],-128),linestyle = linestyle[case_itr])
        ax.legend(['Baseline','Treated'])
        ax.set(ylabel = r'$Pressure \ [Pa]$', xlabel =r'$Rev \ Fraction$',title = rf'$Mic \ {mics[theta_iter]}: \phi = {phi[theta_iter,phi_iter]}^\circ$',xlim = [0,1],ylim = [-1.5,1.5])
        ax.grid()
        plt.savefig(os.path.join(case_dir,f'p_tseries_m{mics[theta_iter]}.png'),format = 'png')
        plt.close()

for theta_iter in range(acs_data[case]['geometry_values'].shape[0]):
    for phi_iter in range(acs_data[case]['geometry_values'].shape[1]):
        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(left = .15,bottom = .15)
        ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],np.roll(acs_data[case_name[1]]['function_values'][theta_iter,phi_iter,:,-1]-acs_data[case_name[0]]['function_values'][theta_iter,phi_iter,:,-1],-128))
        ax.legend(['Baseline','Treated'])
        ax.set(ylabel = r'$Pressure \ [Pa]$', xlabel =r'$Rev \ Fraction$',title = rf'$Mic \ {mics[theta_iter]}: \phi = {phi[theta_iter,phi_iter]}^\circ$',xlim = [0,1],ylim = [0.5,-.5])
        ax.grid()


    figsize = (6.5*.95,6.5*.95)
    fig, ax = plt.subplots(len(mics),2,figsize = figsize)
    plt.subplots_adjust(left=0.1,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
    for mic_itr in range(len(mics)):
        for case_itr,case in enumerate(case_name):
            ax[mic_itr,0].plot(acs_data[case]['function_values'][mic_itr,0,:,0]/acs_data[case]['function_values'][mic_itr,0,-1,0],np.roll(acs_data[case]['function_values'][mic_itr,0,:,-1],-128),c=np.roll(default_colors,-case_itr)[0], linestyle=np.roll(linestyle,-case_itr)[0], label=case)
        ax[mic_itr,0].set(xlim = [0,1],ylim = [-1.4,1.4],title = title[mic_itr])
        if mic_itr !=len(mics)-1:
            ax[mic_itr,0].set_xticklabels([])
        ax[mic_itr,0].grid()
    ax[-1,0].set(xlabel = 'Rev. Fraction')
    ax[int(len(mics)/2),0].set(ylabel = 'Pressure [Pa]')
    for mic_itr in range(len(mics)):
        for case_itr,case in enumerate(case_name):
            markerline, stemlines, baseline = ax[mic_itr,1].stem(acs_data[case]['f'],10*np.log10(acs_data[case]['pxx'][mic_itr,0,:,-1]*np.diff(acs_data[case]['f'][:2])[0]/20e-6**2))
            stemlines.set(color = default_colors[case_itr])
            markerline.set(color = default_colors[case_itr],marker = marker[case_itr])
            ax[mic_itr,1].set(ylim = [0,90],xscale = 'log',xlim = [100,5e3],title = title[mic_itr])
        if mic_itr !=len(mics)-1:
            ax[mic_itr,1].set_xticklabels([])
        ax[mic_itr,1].grid()
    ax[-1,1].set_xlabel('Frequency [Hz]')
    ax[int(len(mics)/2),1].set_ylabel(r'SPL, dB (re: 20$\mathrm{\mu}$Pa)')
    fig.legend(leg_labs,ncol = 4,loc='lower center',bbox_to_anchor=(.5, -0.01))
    plt.savefig(f'tseries_psd_{"__".join([os.path.dirname(case) for case in case_name])}.pdf',format = 'pdf')
    plt.close()



        
fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = .15,bottom = .15)
for theta_iter in range(acs_data[case]['geometry_values'].shape[0]):
    for phi_iter in range(acs_data[case]['geometry_values'].shape[1]):
        ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],np.roll(acs_data[case_name[1]]['function_values'][theta_iter,phi_iter,:,-1]-acs_data[case_name[0]]['function_values'][theta_iter,phi_iter,:,-1],-128),label = f'Mic {mics[theta_iter]}')
ax.set(ylabel = r'$Pressure \ [Pa]$', xlabel =r'$Rev \ Fraction$',xlim = [0,1],ylim = [0.5,-.5])
ax.grid()
ax.legend()

dp = acs_data[case_name[1]]['function_values'][...,1:]-acs_data[case_name[0]]['function_values'][...,1:]
dp_inplane = acs_data[case_name[3]]['function_values'][...,1:]-acs_data[case_name[2]]['function_values'][...,1:]
dp_outplane = acs_data[case_name[-1]]['function_values'][...,1:]-acs_data[case_name[-2]]['function_values'][...,1:]

for theta_iter in range(acs_data[case]['geometry_values'].shape[0]):
    for phi_iter in range(acs_data[case]['geometry_values'].shape[1]):
        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(left = .15,bottom = .15)
        ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],dp[theta_iter,phi_iter,:,-1])
        ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],dp_inplane[theta_iter,phi_iter,:,-1])
        ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],dp_outplane[theta_iter,phi_iter,:,-1])
        ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],(dp_outplane+dp_inplane)[theta_iter,phi_iter,:,-1],linestyle = '-.')
        ax.grid()
        ax.legend(['total','inplane','outplane','total 2'])

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = .15,bottom = .15)
ax.plot(acs_data[case]['function_values'][0,0,:,0]/acs_data[case]['function_values'][0,0,-1,0],dp[3,0,:,-1])
ax.plot(acs_data[case]['function_values'][0,0,:,0]/acs_data[case]['function_values'][0,0,-1,0],dp_inplane[3,0,:,-1],linestyle = '--')
ax.plot(acs_data[case]['function_values'][0,0,:,0]/acs_data[case]['function_values'][0,0,-1,0],-0.5*(dp[3,0,:,-1]+dp[5,0,:,-1]))
ax.plot(acs_data[case]['function_values'][0,0,:,0]/acs_data[case]['function_values'][0,0,-1,0],dp_outplane[3,0,:,-1],linestyle = '--')
ax.plot(acs_data[case]['function_values'][0,0,:,0]/acs_data[case]['function_values'][0,0,-1,0],dp[3,0,:,-1]-0.5*(dp[3,0,:,-1]+dp[5,0,:,-1]),linestyle = '--')

ax.plot(acs_data[case]['function_values'][0,0,:,0]/acs_data[case]['function_values'][0,0,-1,0],dp_inplane[6,0,:,-1],linestyle = '-.')
ax.grid()

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(left = .15,bottom = .15)
for theta_iter in range(acs_data[case]['geometry_values'].shape[0]):
    for phi_iter in range(acs_data[case]['geometry_values'].shape[1]):
        ax.plot(acs_data[case]['function_values'][theta_iter,phi_iter,:,0]/acs_data[case]['function_values'][theta_iter,phi_iter,-1,0],np.roll(acs_data[case_name[1]]['function_values'][theta_iter,phi_iter,:,-1]-acs_data[case_name[0]]['function_values'][theta_iter,phi_iter,:,-1],-128),label = f'Mic {mics[theta_iter]}')
ax.set(ylabel = r'$Pressure \ [Pa]$', xlabel =r'$Rev \ Fraction$',xlim = [0,1],ylim = [0.5,-.5])
ax.grid()
ax.legend()

for theta_iter in range(acs_data[case]['geometry_values'].shape[0]):
    for phi_iter in range(acs_data[case]['geometry_values'].shape[1]):
        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(left = .15,bottom = .15)
        for case_itr,case in enumerate(case_name):
            ax.stem(acs_data[case]['f'],10*np.log10(acs_data[case]['pxx'][theta_iter,phi_iter,:,-1]*df/20e-6**2),linefmt = default_colors[case_itr])
        ax.legend(['Noncompact','Compact Line','Compact Point'])
        ax.set(ylabel = r'$SPL, \ (re: \ 20 \mu Pa)$', xlabel =r'$Frequency \ [Hz]$',title = rf'$Mic \ {mics[theta_iter]}: \phi = {phi[theta_iter,phi_iter]}^\circ$',xlim = [100,10e3],xscale = 'log',ylim = [0,100])
        ax.grid()
        plt.savefig(os.path.join(case_dir,f'spectra_m{mics[theta_iter]}.png'),format = 'png')
        plt.close()
