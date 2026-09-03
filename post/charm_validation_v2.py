#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))

from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  
from pyPostAcs.pyPostAcsFun import *
import plot_styles

from scipy.signal import welch
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-','--','--','-.']

#%%

exp_dir ='/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/RES_BLADE_9_25/rotor_airframe_data/10_6/kde_t1775_bl_d05_m90_r2'
charm_dir = "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/charm/rotor_af_interaction"
charm_cases= ['quickROD_NPSI128/quickROD.1PSU-WOPWOP']
title =[r"$\mathrm{\phi = 23^\circ}$",r"$\mathrm{\phi = 0^\circ}$",r"$\mathrm{\phi = -23^\circ}$"]
leg_lab = [r'Measured',r'Predicted']
class arg:pass
args = arg()

exp_data = import_h5(os.path.join(exp_dir, 'acs_data.h5'))
if "Performance_Data" not in exp_data: 
    apply_fun(exp_dir,[],args)
    exp_data = import_h5(os.path.join(exp_dir, 'acs_data.h5'))

args.mics = [8,4,0]
args.tonal_separation=True
args.filter_harmonics=[1,100]
args.align=True
args.filter_shaft_order=False
args.start_t = 0
args.end_t = exp_data['Acoustic Data'].shape[-1]/exp_data['Sampling Rate']
args.start_ind = int(args.start_t*exp_data['Sampling Rate'])
args.end_ind = int(args.end_t*exp_data['Sampling Rate'])
args.window = 'hann'
args.overlap=0.5
args.nperseg=exp_data['Acoustic Data'].shape[-1]
args.plot = False

t,xn_avg,xn_records,xn_bb,f_tonal,pxx_tonal,pxx_err,f_bb,pxx_bb = tonal_separation(exp_data,args)
df_tonal = np.diff(f_tonal[:2])[0]

acs_data = {}
# case_name =os.path.basename(os.path.dirname(charm_dir)) 
for case in charm_cases:
    if not os.path.exists(os.path.join(charm_dir,case,f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(charm_dir,case),cases = 'cases.nam')

    acs_data.update({case:import_results_from_wopwop(cases_directory=os.path.join(charm_dir,case))})
    acs_data[case]['geometry_values'] = np.flip(acs_data[case]['geometry_values'],axis = 0)[args.mics]
    acs_data[case]['function_values'] = np.flip(acs_data[case]['function_values'],axis = 0)[args.mics]

    theta = np.round(np.arctan2(acs_data[case]['geometry_values'][:,:,0,1],acs_data[case]['geometry_values'][:,:,0,0])*180/np.pi)
    phi = np.arctan2(acs_data[case]['geometry_values'][:,:,0,-1],np.linalg.norm((acs_data[case]['geometry_values'][:,:,0,0],acs_data[case]['geometry_values'][:,:,0,1]),axis = 0))

    nperseg = acs_data[case]['function_values'].shape[-2]
    dt = np.diff(acs_data[case]['function_values'][0,0,:2,0])[0]
    df = (nperseg*dt)**-1
    f,pxx = welch(acs_data[case]['function_values'], fs=dt**-1, window='boxcar', nperseg=nperseg, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-2, average='mean')
    
    acs_data[case].update({'f':f,'pxx':pxx,'theta':theta,'phi':phi})

# Xm = fft(acs_data['function_values'][:,:,:,1:],axis = -2)*dt
# Gxx = (1/(nperseg*dt)*np.abs(Xm)**2)[:,:,:int(nperseg/2)+1]
# Gxx[1:-1] = 2*Gxx[1:-1]

#%%

for mic_itr in range(len(args.mics)):
    fig,ax = plt.subplots(1,1, figsize = (3,2/3*3))
    plt.subplots_adjust(left = .175,bottom = .175,top = 0.95)
    ax.plot(t/t[-1],xn_avg[mic_itr],linestyle = linestyle[0],c = default_colors[0],label = 'Measured',linewidth = 2)
    for case_itr,case in enumerate(charm_cases):
        if case_itr==1:
            ax.plot(acs_data[case]['function_values'][mic_itr,0,:,0]/acs_data[case]['function_values'][mic_itr,0,-1,0],np.roll(acs_data[case]['function_values'][mic_itr,0,:,-1],-60),linestyle = linestyle[case_itr+1],label = leg_lab[case_itr+1],linewidth =2)
        else:
            ax.plot(acs_data[case]['function_values'][mic_itr,0,:,0]/acs_data[case]['function_values'][mic_itr,0,-1,0],np.roll(acs_data[case]['function_values'][mic_itr,0,:,-1],-100),linestyle = linestyle[case_itr+1],label = leg_lab[case_itr+1],linewidth =2)
    ax.plot(t/t[-1],xn_records[mic_itr],linestyle = linestyle[0],c = 'grey',alpha = 0.05,zorder = 0)
    ax.legend(leg_lab,loc='lower center',fontsize=9,borderaxespad=0.25,handletextpad=0.5)
    ax.set(ylabel = r'p [Pa]', xlabel =r'Rev. Fraction',xlim = [0,1],ylim = [-20,8])
    ax.grid()
    plt.savefig(os.path.join(os.path.join(os.path.dirname(os.path.dirname(charm_dir)),f'p_tseries_m{args.mics[mic_itr]}_{os.path.basename(exp_dir)}_{os.path.basename(os.path.dirname(charm_dir))}.pdf')),format = 'pdf',bbox_inches='tight')
    plt.close()

figsize = (6.5*.95,6.5*.95)
fig, ax = plt.subplots(len(args.mics),2,figsize = figsize)
plt.subplots_adjust(left=0.11,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
for mic_itr in range(len(args.mics)):
    ax[mic_itr,0].plot(t/t[-1],xn_avg[mic_itr],linestyle = linestyle[0],c = default_colors[0],label = 'Measured',linewidth = 2)
    for case in charm_cases:
        ax[mic_itr,0].plot(acs_data[case]['function_values'][mic_itr,0,:,0]/acs_data[case]['function_values'][mic_itr,0,-1,0],np.roll(acs_data[case]['function_values'][mic_itr,0,:,-1],-114),linestyle = linestyle[1],c = default_colors[1],label = 'Predicted',linewidth =2)
    ax[mic_itr,0].plot(t/t[-1],xn_records[mic_itr],linestyle = linestyle[0],c = 'grey',alpha = 0.05,zorder = 0)
    ax[mic_itr,0].set(xlim = [0,1],ylim = [-2,2],title =title[mic_itr])
    if mic_itr !=len(args.mics)-1:
        ax[mic_itr,0].set_xticklabels([])
    ax[mic_itr,0].grid()
ax[-1,0].set(xlabel = 'Rev. Fraction')
ax[int(len(args.mics)/2),0].set(ylabel = 'p [Pa]')
for mic_itr in range(len(args.mics)):
    ax[mic_itr,1].stem(f_tonal,10*np.log10(pxx_tonal[mic_itr]*df_tonal/20e-6**2),linefmt = default_colors[0])
    ax[mic_itr,1].stem(f,10*np.log10(pxx[mic_itr,0,:,-1]*df/20e-6**2),linefmt = default_colors[1],markerfmt=f'^')
    ax[mic_itr,1].set(ylim = [0,100],xscale = 'log',xlim = [10,10e3],title =title[mic_itr])
    if mic_itr !=len(args.mics)-1:
        ax[mic_itr,1].set_xticklabels([])
    ax[mic_itr,1].grid()
ax[-1,1].set_xlabel('Frequency [Hz]')
ax[int(len(args.mics)/2),1].set_ylabel(r'SPL, dB (re: 20$\mathrm{\mu}$Pa)')
fig.legend(['Measured','Predicted'],ncol = 2,loc='lower center',bbox_to_anchor=(.5, -0.01))
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(charm_dir)),f'{os.path.basename(exp_dir)}_{os.path.basename(os.path.dirname(charm_dir))}_validation.png'),format = 'png',dpi = 400)

plt.savefig(os.path.join(os.path.dirname(os.path.dirname(charm_dir)),f'{os.path.basename(exp_dir)}_{os.path.basename(os.path.dirname(charm_dir))}_validation.pdf'),format = 'pdf')
plt.close()
