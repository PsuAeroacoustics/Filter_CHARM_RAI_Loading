#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))

from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  
from scipy.signal import welch
from scipy.fft import fft
import plot_styles
import matplotlib.colors as mcolors
import cmap as cm

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.',':']
marker = ['o','^','*']
#%%

# def process_wopwop(cases_directory,cases = 'cases.nam'):
#     f1 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'pressure')
#     f2 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'spl_spectrum')
#     apply_to_namelist([f1], cases_directory=cases_directory, cases=cases)

#%%

case_name = ['quickROD.1PSU-WOPWOP_HEMI_UP_BP','quickROD.1PSU-WOPWOP_HEMI_UP_ROD_ONLY','quickROD.1PSU-WOPWOP_HEMI_UP_ROTOR_ONLY']
select_points = [(270,15),(270,30),(180,15),(90,30),(90,15)]

upper = False
lambert = True
A_weighting = False
raispl = True

#%%

case_dir = os.path.join(os.getcwd())
acs_data ={}

lambert = lambda x: np.sqrt(2)*r.mean()*np.sin((np.pi/2-x)/2)
stereo = lambda x: 2*r.mean()*np.tan((np.pi/2-x)/2)
phi_labels = (np.arange(7)/6*90)

A_weight = lambda f: 10**((2+20*np.log10(12194**2*f**4/((f**2+20.6**2)*np.sqrt((f**2+107.7**2)*(f**2+737.9**2))*(f**2+12194**2))))/10)

for case in case_name:
    
    if not os.path.exists(os.path.join(case_dir,f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(case_dir,case),cases = 'cases_hemi.nam')

    acs_data.update({case:import_results_from_wopwop(cases_directory=os.path.join(case_dir,case))})
    
    if A_weighting:
        dt = acs_data[case]['function_values'][0,0,1,0]
        N = len(acs_data[case]['function_values'][0,0,:,0])
        df = 1/(N*dt)
        f = np.arange(int(np.ceil(N/2)+1))*df
        Xm = fft(acs_data[case]['function_values'][...,-1],axis = -1)*dt
        Sxx = 1/(dt*N)*A_weight(f)*np.abs(Xm[...,:int(np.ceil(N/2)+1)])**2
        Sxx[1:-1] = 2*Sxx[1:-1]
        oaspl = 10*np.log10(np.trapezoid(Sxx,dx = df,axis = -1)/20e-6**2)
    else:
        oaspl = np.round(10*np.log10(np.mean(acs_data[case]['function_values'][...,-1]**2,axis = -1)/20e-6**2),2)

    theta = np.arctan2(acs_data[case]['geometry_values'][:,:,0,1],acs_data[case]['geometry_values'][:,:,0,0])%(2*np.pi)
    theta[-1] = 2*np.pi

    phi = np.abs(np.arctan2(acs_data[case]['geometry_values'][:,:,0,-1],np.linalg.norm((acs_data[case]['geometry_values'][:,:,0,0],acs_data[case]['geometry_values'][:,:,0,1]),axis = 0)))

    # if upper:
    #     phi = (2*np.pi-phi)%(2*np.pi)
    dphi = np.abs(np.diff(phi[0,:-1]).mean()*180/np.pi)
    dtheta = np.abs(np.diff(theta[:-1,0]).mean()*180/np.pi)
    r = np.linalg.norm(acs_data[case]['geometry_values'][:,:,0],axis = -1)
    if lambert:
        rho = lambert(phi)
    else:
        rho = stereo(phi)

    acs_data[case].update({'theta':theta,'phi':phi,'r':r,'rho':rho,'oaspl':oaspl,'dphi':dphi,'dtheta':dtheta})

cmap = plt.cm.get_cmap('inferno')

if lambert:
    phi_ticks = lambert(phi_labels*np.pi/180)
else:
    phi_ticks = stereo(phi_labels*np.pi/180)
if upper:
    phi_ticklabels = np.asarray([rf"${i}^\circ$" for i in np.round(phi_labels).astype(int)])
else:
    phi_ticklabels = np.asarray([rf"${-i}^\circ$" for i in np.round(phi_labels).astype(int)])
phi_ticklabels[1::2] = ' '

fig,ax = plt.subplots(subplot_kw=dict(projection = 'polar'),figsize = (3.4,3.4/1.25))
# levels = np.linspace(90,110,41)
# cbar_ticks = np.round(levels)[::4]

levels = np.linspace(70,90,21)
cbar_ticks = np.round(levels)[::2]

# levels = np.linspace(70,100,31)
# cbar_ticks = np.round(levels)[::3]

# ax.scatter(acs_data[case_name[0]]['theta'][:,:-1]%(2*np.pi),acs_data[case_name[0]]['rho'][:,:-1],c = 'gray',alpha = 1)
dist = ax.contourf(acs_data[case_name[0]]['theta'],acs_data[case_name[0]]['rho'],acs_data[case_name[0]]['oaspl'],cmap = cmap,    levels=levels)

for pnt_itr,pnt in enumerate(select_points):
    phi_ind = np.abs(pnt[1]*np.pi/180-acs_data[case_name[0]]['phi'][0]).argmin()
    theta_ind = np.abs(pnt[0]*np.pi/180-acs_data[case_name[0]]['theta'][:,0]).argmin()
    if lambert:
        ax.scatter(acs_data[case_name[0]]['theta'][theta_ind,phi_ind],lambert(acs_data[case_name[0]]['phi'][theta_ind,phi_ind]),c = 'black')
    else:
        ax.scatter(acs_data[case_name[0]]['theta'][theta_ind,phi_ind],stereo(acs_data[case_name[0]]['phi'][theta_ind,phi_ind]),c = 'black')
    plt.annotate(f'O{pnt_itr+1}', (acs_data[case_name[0]]['theta'][theta_ind,phi_ind], lambert(acs_data[case_name[0]]['phi'][theta_ind,phi_ind])), xytext=None, textcoords="offset points")

cbar = fig.colorbar(dist,pad = .15)
if A_weighting:
    cbar.ax.set_ylabel(r'OASPL, dB(A) (re: 20$\mathrm{\mu}$Pa)')
elif raispl:
    cbar.ax.set_ylabel(r'RAISPL, dB (re: 20$\mathrm{\mu}$Pa)')
else:
    cbar.ax.set_ylabel(r'OASPL, dB (re: 20$\mathrm{\mu}$Pa)')
cbar.ax.set_yticks(cbar_ticks)
ax.set_yticks(phi_ticks)
ax.set_yticklabels(phi_ticklabels)
plt.savefig(os.path.join(case_dir,f'hemi.png'),format = 'png',dpi = 600)
plt.close()


for pnt_itr,pnt in enumerate(select_points):
    fig,ax = plt.subplots(1,1, figsize = (2.5,2.5/1.25))
    plt.subplots_adjust(left = .25,bottom = .21,top = 0.875)
    phi_ind = np.abs(pnt[1]*np.pi/180-acs_data[case]['phi'][0]).argmin()
    theta_ind = np.abs(pnt[0]*np.pi/180-acs_data[case]['theta'][:,0]).argmin()
    # ax.plot(acs_data[case]['function_values'][theta_ind,phi_ind,:,0]/acs_data[case]['function_values'][theta_ind,phi_ind,-1,0],np.roll(acs_data[case]['function_values'][theta_ind,phi_ind,:,-1],-95))
    for case_itr,case in enumerate(case_name):
        ax.plot(acs_data[case]['function_values'][theta_ind,phi_ind,:,0]/acs_data[case]['function_values'][theta_ind,phi_ind,-1,0],np.roll(acs_data[case]['function_values'][theta_ind,phi_ind,:,-1],-95),linestyle = linestyle[case_itr])

    # ax.legend(leg_labs,loc='lower center',fontsize=9,borderaxespad=0.25,handletextpad=0.5)
    ax.set(ylabel = r'p [Pa]', xlabel =r'Rev. Fraction',xlim = [0,1],ylim = [-2,2])
    ax.set_title(rf'$\mathrm{{O{pnt_itr+1}}} \ (\psi={int(np.round(acs_data[case]['theta'][theta_ind,phi_ind]*180/np.pi))}^\circ, \phi={int(np.round(acs_data[case]['phi'][theta_ind,phi_ind]*180/np.pi))}^\circ)$', fontdict={'fontsize': 11})
    ax.grid()
    ax.legend(['Total','Rod','Rotor'],ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)
    plt.savefig(os.path.join(case_dir,f'p_tseries_{pnt_itr}.png'),format = 'png',dpi = 600)
    plt.close()
