#!/usr/bin/env python3

import os
import numpy as np
from dependencies.pyWopwop.wopwop import *  
from dependencies.pyWopwop.wopwop_io import *  
from dependencies.pyPostAcs.pyPostAcsFun import *

from scipy.signal import welch
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.']

#%%

exp_dir ='/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/RES_BLADE_9_25/rotor_airframe_data/10_6/kde_t1775_bl_d05_m90_r2'
charm_dir = "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/charm/rotor_af_interaction/quickROD_NPSI128/quickROD.1PSU-WOPWOP"

class arg:pass
args = arg()

exp_data = import_h5(os.path.join(exp_dir, 'acs_data.h5'))
if "Performance_Data" not in exp_data: 
    apply_fun(exp_dir,[],args)
    exp_data = import_h5(os.path.join(exp_dir, 'acs_data.h5'))

args.mics = [0,1,2,3,4,5,6,7,8,9]
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
#%%
def process_wopwop(cases_directory,cases = 'cases.nam'):
    f1 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'pressure')
    f2 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'spl_spectrum')
    apply_to_namelist([f1], cases_directory=cases_directory, cases=cases)

case_name =os.path.basename(os.path.dirname(charm_dir)) 
    
if not os.path.exists(os.path.join(charm_dir,f'{case_name}.h5')):
    process_wopwop(cases_directory=charm_dir,cases = 'cases.nam')

acs_data = import_results_from_wopwop(cases_directory=charm_dir)
acs_data['geometry_values'] = np.flip(acs_data['geometry_values'],axis = 0)
acs_data['function_values'] = np.flip(acs_data['function_values'],axis = 0)

theta = np.round(np.arctan2(acs_data['geometry_values'][:,:,0,1],acs_data['geometry_values'][:,:,0,0])*180/np.pi)
phi = np.round(np.arctan2(acs_data['geometry_values'][:,:,0,-1]-acs_data['geometry_values'][2,:,0,-1],np.linalg.norm((acs_data['geometry_values'][:,:,0,0],acs_data['geometry_values'][:,:,0,1]),axis = 0))*180/np.pi)

nperseg = acs_data['function_values'].shape[-2]
dt = np.diff(acs_data['function_values'][0,0,:2,0])[0]
df = (nperseg*dt)**-1
f,pxx = welch(acs_data['function_values'], fs=dt**-1, window='boxcar', nperseg=nperseg, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-2, average='mean')

# Xm = fft(acs_data['function_values'][:,:,:,1:],axis = -2)*dt
# Gxx = (1/(nperseg*dt)*np.abs(Xm)**2)[:,:,:int(nperseg/2)+1]
# Gxx[1:-1] = 2*Gxx[1:-1]

#%%
for mic_itr,mic in enumerate(args.mics):
    fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
    plt.subplots_adjust(left = .15,bottom = .15)
    ax.plot(t/t[-1],xn_records[mic_itr],linestyle = linestyle[0],c = 'grey',alpha = 0.05)
    ax.plot(t/t[-1],xn_avg[mic_itr],linestyle = linestyle[0],c = default_colors[0],label = 'Measured',linewidth = 3)
    ax.plot(acs_data['function_values'][mic_itr,0,:,0]/acs_data['function_values'][mic_itr,0,-1,0],np.roll(acs_data['function_values'][mic_itr,0,:,-1],-128),linestyle = linestyle[1],c = default_colors[1],label = 'Predicted',linewidth =3)
    ax.set(xlim = [0,1],ylim = [None,None],ylabel = r'$Pressure \ [Pa]$', xlabel =r'$Rev \ Fraction$',title = rf'$Mic \ {mic}: \phi = {phi[mic_itr,0]}^\circ$')
    ax.legend()
    ax.grid()
    # plt.savefig(os.path.join(charm_dir,f'p_tseries_m{mic}.png'),format = 'png')
    # plt.close()


    fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
    plt.subplots_adjust(left = .15,bottom = .15)
    ax.stem(f_tonal,10*np.log10(pxx_tonal[mic_itr]*df_tonal/20e-6**2),linefmt = default_colors[0])
    ax.stem(f,10*np.log10(pxx[mic_itr,0,:,-1]*df/20e-6**2),linefmt = default_colors[1])
    ax.legend(['Measured','Predicted'])
    ax.set(ylabel = r'$SPL, \ (re:\ 20 \mu Pa)$', xlabel =r'$Frequency \ [Hz]$',title = rf'$Mic \ {mic}: \phi = {phi[mic_itr,0]}^\circ$',xlim = [100,10e3],xscale = 'log',ylim = [0,100])
    ax.grid()
    plt.savefig(os.path.join(charm_dir,f'psd_m{mic}.png'),format = 'png')
    plt.close()