#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))

from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  

from pyPostAcs.pyPostAcsFun import *
# from extract_loading_charm import extract_loading
import plot_styles



default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.']

#%%

exp_dir ='/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/RES_BLADE_9_25/rotor_airframe_data/10_6/'
exp_cases = ['kde_t1775_bl_dn05_m90_r2','kde_t1775_sdof_dn05_m90_r2']
exp_mics = [8,4,0]
# x,y,z coordinates of center mic which is approximately aligned with the rod
center_mic_coord = np.asarray([ 0.18097501,  -1.524     , -0.04064   ])
center_mic_ind = 4

validate = True
charm_dir = "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/charm/rotor_af_interaction/"
charm_cases = ['quickROD_DN05_IAERO1/quickROD.1PSU-WOPWOP','quickROD_DN05_SDOF_GEOM_OAR15/quickROD.1PSU-WOPWOP']
charm_mics = [8,4,0]
charm_loading_fname = ['loading0200.dat','loading0200_sdof_geom_oar15.dat']
c0 = 340
leg_labs = ['BL','SDOF']
title =[r"$\mathrm{\phi = 23^\circ}$",r"$\mathrm{\phi = 0^\circ}$",r"$\mathrm{\phi = -23^\circ}$"]
#%%
class arg:pass
args = arg()
args.mics = exp_mics
args.tonal_separation=True
args.filter_harmonics=[1,100]
args.align=True
args.filter_shaft_order=False
args.start_t = 0
args.window = 'hann'
args.overlap=0.5
args.df = 5
args.plot = False


# imports data from charm
N = 352

charm_data = {}
for case in charm_cases:
    if not os.path.exists(os.path.join(charm_dir,f'{charm_cases}.h5')):
        process_wopwop(cases_directory=os.path.join(charm_dir,case),cases = 'cases.nam')
    charm_data.update({case:import_results_from_wopwop(cases_directory=os.path.join(charm_dir,case))})

    charm_data[case]['geometry_values'] = np.flip(charm_data[case]['geometry_values'],axis = 0)[charm_mics]
    charm_data[case]['function_values'] = np.flip(charm_data[case]['function_values'],axis = 0)[charm_mics]

    theta = np.round(np.arctan2(charm_data[case]['geometry_values'][:,:,0,1],charm_data[case]['geometry_values'][:,:,0,0])*180/np.pi)
    phi = np.arctan2(charm_data[case]['geometry_values'][:,:,0,-1],np.linalg.norm((charm_data[case]['geometry_values'][:,:,0,0],charm_data[case]['geometry_values'][:,:,0,1]),axis = 0))

    nperseg = charm_data[case]['function_values'].shape[-2]
    dt = np.diff(charm_data[case]['function_values'][0,0,:2,0])[0]
    df = (nperseg*dt)**-1
    f,pxx = welch(charm_data[case]['function_values'], fs=dt**-1, window='boxcar', nperseg=nperseg, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-2, average='mean')

    charm_data[case].update({'f':f,'pxx':pxx,'theta':theta,'phi':phi})

f_charm,dpxx_charm = welch(charm_data[charm_cases[1]]['function_values']-charm_data[charm_cases[0]]['function_values'], fs=dt**-1, window='boxcar', nperseg=nperseg, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-2, average='mean')


exp_data = {}
for case in exp_cases:
    
    exp_data.update({case:import_h5(os.path.join(exp_dir,case, 'acs_data.h5'))})
    if "Performance_Data" not in exp_data[case]: 
        apply_fun(os.path.join(exp_dir,case),[],args)
        exp_data.update({case:import_h5(os.path.join(exp_dir,case, 'acs_data.h5'))})
    

    args.end_t = exp_data[case]['Acoustic Data'].shape[-1]/exp_data[case]['Sampling Rate']
    args.start_ind = int(args.start_t*exp_data[case]['Sampling Rate'])
    args.end_ind = int(args.end_t*exp_data[case]['Sampling Rate'])

    args.nperseg= exp_data[case]['Sampling Rate']/args.df
    f,pxx = welch(exp_data[case]['Acoustic Data'][args.mics,args.start_ind:args.end_ind], fs=exp_data[case]['Sampling Rate'], window=args.window, nperseg=args.nperseg, noverlap=int(args.overlap*args.nperseg), nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

    args.nperseg=exp_data[case]['Acoustic Data'].shape[-1]

    t,xn_avg,xn_records,xn_bb,f_tonal,pxx_tonal,pxx_err,f_bb,pxx_bb = tonal_separation(exp_data[case],args)
    fs = np.diff(t[:2])[0]**-1

    exp_data[case].update({'f':f,'fs':fs,'pxx':pxx,'t':t,'xn_avg':xn_avg,'xn_records':xn_records,'xn_bb':xn_bb,'f_tonal':f_tonal,'pxx_tonal':pxx_tonal,'pxx_err':pxx_err,'f_bb':f_bb,'pxx_bb':pxx_bb})

N_pnts = np.asarray([len(exp_data[case]['t']) for case in exp_cases])
if np.diff(N_pnts)[0]!=0:
    N_upsample = np.max(N_pnts)
    t = np.arange(N_upsample)*np.diff(exp_data[exp_cases[N_pnts.argmax()]]['t'][:2])[0]
    for i,case in enumerate(exp_cases):
        dt = np.diff(exp_data[case]['t'][:2])[0]
        fs = N*dt**-1/N_pnts[i]
        xn_avg = upsample(exp_data[case]['xn_avg'],dt**-1,N = N_upsample)
        xn_records = np.asarray([upsample(x.T,dt**-1,N = N).T for x in xn_records])
        exp_data[case].update({'t':t,'xn_avg':xn_avg,'xn_records':xn_records,'fs':fs})

fs = np.mean((exp_data[exp_cases[0]]['fs'],exp_data[exp_cases[1]]['fs']))
dt = fs**-1
t_shift = np.zeros(len(exp_cases))
for i,case in enumerate(exp_cases[1:]):
    N = np.max((exp_data[exp_cases[0]]['xn_avg'].shape[-1],exp_data[case]['xn_avg'].shape[-1]))
    t,Rxy,_ = correlation(np.concatenate((exp_data[exp_cases[0]]['xn_avg'][0],np.zeros(2*N-len(exp_data[exp_cases[0]]['xn_avg'][0])))),np.concatenate((exp_data[case]['xn_avg'][0],np.zeros(2*N-len(exp_data[case]['xn_avg'][0])))),fs = fs,auto = False)
    t_shift[i+1] = t[Rxy.argmax(axis = -1)]

# # aligns the time series from the two test points 
t,Rxy,_ = correlation(np.concatenate((exp_data[exp_cases[0]]['xn_avg'][0],np.zeros(N))),np.concatenate((exp_data[exp_cases[1]]['xn_avg'][0],np.zeros(N))),fs = fs,auto = False)
exp_data[exp_cases[1]]['xn_avg'] = np.roll(exp_data[exp_cases[1]]['xn_avg'],-np.round(t[Rxy.argmax(axis = -1)]/exp_data[exp_cases[1]]['fs']**-1),axis = -1)
# difference in the averaged pressure time series of the two two data points 
dp = (exp_data[exp_cases[1]]['xn_avg']-exp_data[exp_cases[0]]['xn_avg']).T

t = np.arange(N)*dt
f_exp,dpxx_exp = welch(dp, fs=dt**-1, window='boxcar', nperseg=len(dp), noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='mean')



#%%

figsize = (6.5*.95,6.5*.95)
fig, ax = plt.subplots(len(charm_mics),2,figsize = figsize)
plt.subplots_adjust(left=0.125,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
for mic_itr in range(len(charm_mics)):
    for case_itr,case in enumerate(charm_cases):
        ax[mic_itr,0].plot(charm_data[case]['function_values'][mic_itr,0,:,0]/charm_data[case]['function_values'][mic_itr,0,-1,0],np.roll(charm_data[case]['function_values'][mic_itr,0,:,-1],-114),c=np.roll(default_colors,-case_itr)[0], linestyle=np.roll(linestyle,-case_itr)[0], label=case)
    ax[mic_itr,0].set(xlim = [0,1],ylim = [-15,5],title =title[mic_itr])
    if mic_itr !=len(charm_mics)-1:
        ax[mic_itr,0].set_xticklabels([])
    ax[mic_itr,0].grid()
ax[-1,0].set(xlabel = 'Rev. Fraction')
ax[int(len(charm_mics)/2),0].set(ylabel = 'P [Pa]')
for mic_itr in range(len(charm_mics)):
    for case_itr,case in enumerate(charm_cases):
        markerline, stemlines, baseline = ax[mic_itr,1].stem(charm_data[case]['f'],10*np.log10(charm_data[case]['pxx'][mic_itr,0,:,-1]*np.diff(charm_data[case]['f'][:2])[0]/20e-6**2))
        stemlines.set(color = default_colors[case_itr])
        markerline.set(color = default_colors[case_itr])
        ax[mic_itr,1].set(ylim = [0,90],xscale = 'linear',xlim = [100,10e3],title =title[mic_itr])
    if mic_itr !=len(charm_mics)-1:
        ax[mic_itr,1].set_xticklabels([])
    ax[mic_itr,1].grid()
ax[-1,1].set_xlabel('Frequency [Hz]')
ax[int(len(charm_mics)/2),1].set_ylabel(r'SPL, dB (re: 20$\mathrm{\mu}$Pa)')
fig.legend(leg_labs,ncol = 2,loc='lower center',bbox_to_anchor=(.5, -0.01))
plt.savefig(os.path.join(charm_dir,f'p_tseries_psd_{os.path.dirname(charm_cases[1])}.pdf'),format = 'pdf')

fig, ax = plt.subplots(len(args.mics),2,figsize = figsize)
plt.subplots_adjust(left=0.125,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
for mic_itr in range(len(args.mics)):
    for case_itr,case in enumerate(exp_cases):
        ax[mic_itr,0].plot(exp_data[case]['t']/exp_data[case]['t'][-1],exp_data[case]['xn_avg'][mic_itr],c=np.roll(default_colors,-case_itr)[0], linestyle=np.roll(linestyle,-case_itr)[0], label=case)
    for case_itr,case in enumerate(charm_cases):
        ax[mic_itr,0].plot(charm_data[case]['function_values'][mic_itr,0,:,0]/charm_data[case]['function_values'][mic_itr,0,-1,0],np.roll(charm_data[case]['function_values'][mic_itr,0,:,-1],-114),c=np.roll(default_colors,-case_itr)[3])
    ax[mic_itr,0].set(xlim = [0,1],ylim = [None,None],title =title[mic_itr])
    if mic_itr !=len(args.mics)-1:
        ax[mic_itr,0].set_xticklabels([])
    ax[mic_itr,0].grid()
ax[-1,0].set(xlabel = 'Rev. Fraction')
ax[int(len(args.mics)/2),0].set(ylabel = 'P [Pa]')
for mic_itr in range(len(args.mics)):
    for case_itr,case in enumerate(exp_cases):
        ax[mic_itr,1].errorbar(exp_data[case]['f_tonal'], 10*np.log10(exp_data[case]['pxx_tonal'][mic_itr]*np.diff(exp_data[case]['f_tonal'][:2])[0]/20e-6**2), yerr=10*np.log10(exp_data[case]['pxx_err'][mic_itr]), fmt='o',color = np.roll(default_colors,-case_itr)[0],ecolor = np.roll(default_colors,-case_itr)[0],capsize=6,capthick=2)
        line= ax[mic_itr,1].plot(exp_data[case]['f'],10*np.log10(exp_data[case]['pxx'][mic_itr]*np.diff(exp_data[case]['f'][:2])[0]/20e-6**2))
        line[0].set(color=np.roll(default_colors,-case_itr)[0], linestyle=np.roll(linestyle,-case_itr)[0], label=case)
        ax[mic_itr,1].set(ylim = [0,None],xscale = 'linear',xlim = [100,10e3],title =title[mic_itr])
    if mic_itr !=len(args.mics)-1:
        ax[mic_itr,1].set_xticklabels([])
    ax[mic_itr,1].grid()
ax[-1,1].set_xlabel('Frequency [Hz]')
ax[int(len(args.mics)/2),1].set_ylabel(r'SPL, dB \ (re: 20$\mathrm{\mu}$Pa)')
fig.legend(leg_labs,ncol = 2,loc='lower center',bbox_to_anchor=(.5, -0.01))
plt.savefig(os.path.join(charm_dir,f'p_tseries_psd_{exp_cases[1]}.pdf'),format = 'pdf')


fig, ax = plt.subplots(len(args.mics),2,figsize = figsize)
plt.subplots_adjust(left=0.125,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
for mic_itr in range(len(args.mics)):
    ax[mic_itr,0].plot(exp_data[exp_cases[0]]['t']/exp_data[exp_cases[0]]['t'][-1],np.roll(dp[:,mic_itr],-np.round(t_shift[1]/exp_data[exp_cases[1]]['t'][1])))
    ax[mic_itr,0].plot(charm_data[charm_cases[0]]['function_values'][mic_itr,0,:,0]/charm_data[charm_cases[0]]['function_values'][mic_itr,0,-1,0],np.roll((charm_data[charm_cases[1]]['function_values']-charm_data[charm_cases[0]]['function_values'])[mic_itr,0,:,-1],-114))
    ax[mic_itr,0].set(xlim = [0,1],ylim = [-4,4],title =title[mic_itr])
    if mic_itr !=len(args.mics)-1:
        ax[mic_itr,0].set_xticklabels([])
    ax[mic_itr,0].grid()
ax[-1,0].set(xlabel = 'Rev. Fraction')
ax[int(len(args.mics)/2),0].set(ylabel = r'$\mathrm{\Delta}$ P [Pa]')
for mic_itr in range(len(args.mics)):
    markerline, stemlines, baseline = ax[mic_itr,1].stem(f_exp,10*np.log10(dpxx_exp[:,mic_itr]*np.diff(f_exp[:2])[0]/20e-6**2))
    stemlines.set(color = default_colors[0])
    markerline.set(color = default_colors[0])
    markerline, stemlines, baseline = ax[mic_itr,1].stem(f_charm,10*np.log10(dpxx_charm[mic_itr,0,:,-1]*np.diff(f_charm[:2])[0]/20e-6**2))
    stemlines.set(color = default_colors[1])
    markerline.set(color = default_colors[1])
    ax[mic_itr,1].set(ylim = [0,100],xscale = 'log',xlim = [100,5e3],title =title[mic_itr])
    if mic_itr !=len(args.mics)-1:
        ax[mic_itr,1].set_xticklabels([])
    ax[mic_itr,1].grid()
ax[-1,1].set_xlabel('Frequency [Hz]')
ax[int(len(args.mics)/2),1].set_ylabel(r'$\mathrm{\Delta}$ SPL, dB \ (re: 20$\mathrm{\mu}$Pa)')
fig.legend(['Measured','Predicted'],ncol = 2,loc='lower center',bbox_to_anchor=(.5, -0.01))
plt.savefig(os.path.join(charm_dir,f'dp_tseries_psd_{os.path.dirname(charm_cases[1])}.pdf'),format = 'pdf')

fig, ax = plt.subplots(len(args.mics),1,figsize = figsize)
plt.subplots_adjust(left=0.1,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
for mic_itr in range(len(args.mics)):
    markerline, stemlines, baseline = ax[mic_itr].stem(f_exp,10*np.log10(dpxx_exp[:,mic_itr]*np.diff(f_exp[:2])[0]/20e-6**2))
    stemlines.set(color = default_colors[0])
    markerline.set(color = default_colors[0])
    markerline, stemlines, baseline = ax[mic_itr].stem(f_charm,10*np.log10(dpxx_charm[mic_itr,0,:,-1]*np.diff(f_charm[:2])[0]/20e-6**2))
    stemlines.set(color = default_colors[1])
    markerline.set(color = default_colors[1])
    ax[mic_itr].set(ylim = [0,100],xscale = 'linear',xlim = [100,5e3],title =title[mic_itr])
    if mic_itr !=len(args.mics)-1:
        ax[mic_itr].set_xticklabels([])
    ax[mic_itr].grid()
ax[-1].set_xlabel('Frequency [Hz]')
ax[int(len(args.mics)/2)].set_ylabel(r'$\mathrm{\Delta}$ SPL, dB \ (re: 20$\mathrm{\mu}$Pa)')
fig.legend(['Measured','Predicted'],ncol = 2,loc='lower center',bbox_to_anchor=(.5, -0.01))


fig, ax = plt.subplots(2,1,figsize = figsize)
plt.subplots_adjust(left=0.1,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
for mic_itr in range(len(args.mics)):
    ax[0].plot(exp_data[exp_cases[0]]['t']/exp_data[exp_cases[0]]['t'][-1],np.roll(dp[:,mic_itr],-np.round(t_shift[1]/exp_data[exp_cases[1]]['t'][1])))
    ax[1].plot(charm_data[charm_cases[0]]['function_values'][mic_itr,0,:,0]/charm_data[charm_cases[0]]['function_values'][mic_itr,0,-1,0],np.roll((charm_data[charm_cases[1]]['function_values']-charm_data[charm_cases[0]]['function_values'])[mic_itr,0,:,-1],-115))
for i in range(2):
    ax[i].set(xlim = [0,1],ylim = [-.5,.5],)
    ax[i].grid()





#%%