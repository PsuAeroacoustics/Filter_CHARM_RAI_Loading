#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))

from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  
from scipy.signal import welch
import plot_styles
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cmap as cm

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.',':']
marker = ['o','^','*']

#%%

# r = np.arange(1,20)*0.235
# phi = np.arange(8)*45*np.pi/180

# y = r[:,None]*np.cos(phi)
# z = r[:,None]*np.sin(phi)

case_name = ['quickROD.1PSU-WOPWOP_HEMI_LOW_BP','quickROD.1PSU-WOPWOP_HEMI_LOW_PHI1_NR5_BP']
select_points = [(270,15),(270,30),(180,15),(90,30),(90,15)]
cases_fname = [ 'cases_hemi.nam', 'cases_hemi.nam','cases_hemi.nam']
upper = False
lambert = True

#%%

case_dir = os.path.join(os.getcwd())
acs_data ={}

lambert = lambda x: np.sqrt(2)*r.mean()*np.sin((np.pi/2-x)/2)
stereo = lambda x: 2*r.mean()*np.tan((np.pi/2-x)/2)
phi_labels = (np.arange(7)/6*90)
obs_ind = [18,6]

for case_itr,case in enumerate(case_name):
    
    if not os.path.exists(os.path.join(case_dir,f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(case_dir,case),cases =cases_fname[case_itr])

    acs_data.update({case:import_results_from_wopwop(cases_directory=os.path.join(case_dir,case))})
    oaspl = np.round(10*np.log10(np.mean(acs_data[case]['function_values'][...,-1]**2,axis = -1)/20e-6**2),2)

    theta = np.arctan2(acs_data[case]['geometry_values'][:,:,0,1],acs_data[case]['geometry_values'][:,:,0,0])%(2*np.pi)
    theta[-1] = 2*np.pi
    phi = np.abs(np.arctan2(acs_data[case]['geometry_values'][:,:,0,-1],np.linalg.norm((acs_data[case]['geometry_values'][:,:,0,0],acs_data[case]['geometry_values'][:,:,0,1]),axis = 0)))

    dphi = np.abs(np.diff(phi[0,:-1]).mean()*180/np.pi)
    dtheta = np.abs(np.diff(theta[:-1,0]).mean()*180/np.pi)
    r = np.linalg.norm(acs_data[case]['geometry_values'][:,:,0],axis = -1)
    if lambert:
        rho = lambert(phi)
    else:
        rho = stereo(phi)

    acs_data[case].update({'theta':theta,'phi':phi,'r':r,'rho':rho,'oaspl':oaspl,'dphi':dphi,'dtheta':dtheta})

# cmap = plt.cm.get_cmap('coolwarm').reversed()
cmap = cm.Colormap("crameri:vik").to_mpl()  # case insensitive
# cmap = plt.cm.get_cmap('inferno')
# cmap =LinearSegmentedColormap.from_list(
#         f'trunc({cmap.name})',
#         cmap(np.linspace(0.2, 1, 256))
#     )
if lambert:
    phi_ticks = lambert(phi_labels*np.pi/180)
else:
    phi_ticks = stereo(phi_labels*np.pi/180)
if upper:
    phi_ticklabels = np.asarray([rf"${i}^\circ$" for i in np.round(phi_labels).astype(int)])
else:
    phi_ticklabels = np.asarray([rf"${-i}^\circ$" for i in np.round(phi_labels).astype(int)])
phi_ticklabels[1::2] = ' '

doaspl = (acs_data[case_name[1]]['oaspl']-acs_data[case_name[0]]['oaspl'])

fig,ax = plt.subplots(subplot_kw=dict(projection = 'polar'),figsize = (3.4,3.4/1.25))
# levels = np.linspace(np.floor(doaspl.min()),-np.floor(doaspl.min()),int(np.diff((np.floor(doaspl.min()),-np.floor(doaspl.min())))[0]*2+1))
levels = np.linspace(-10,10,41)
cbar_ticks = np.round(levels)[::2]
# ax.scatter(acs_data[case_name[0]]['theta'][:,:-1]%(2*np.pi),acs_data[case_name[0]]['rho'][:,:-1],c = 'gray',alpha = 1)
dist = ax.contourf(acs_data[case_name[0]]['theta'],acs_data[case_name[0]]['rho'],doaspl,cmap = cmap, levels=levels,norm = mcolors.CenteredNorm())
dist2 = ax.contour(acs_data[case_name[0]]['theta'],acs_data[case_name[0]]['rho'],doaspl,levels = levels[::4],colors = 'k',linestyles = '-.',linewidths = .7,alpha = .75)
plt.clabel(dist2)

for pnt_itr,pnt in enumerate(select_points):
    # phi_ind = int(np.round(pnt[1]/acs_data[case_name[0]]['dphi']))
    # theta_ind = int(np.round(pnt[0]/acs_data[case_name[0]]['dtheta']))
    phi_ind = np.abs(pnt[1]*np.pi/180-acs_data[case]['phi'][0]).argmin()
    theta_ind = np.abs(pnt[0]*np.pi/180-acs_data[case]['theta'][:,0]).argmin()
    if lambert:
        ax.scatter(acs_data[case]['theta'][theta_ind,phi_ind],lambert(acs_data[case]['phi'][theta_ind,phi_ind]),c = 'black')
    else:
        ax.scatter(acs_data[case]['theta'][theta_ind,phi_ind],stereo(acs_data[case]['phi'][theta_ind,phi_ind]),c = 'black')
    plt.annotate(f'O{pnt_itr+1}', (acs_data[case]['theta'][theta_ind,phi_ind], lambert(acs_data[case]['phi'][theta_ind,phi_ind])), xytext=None, textcoords="offset points")

cbar = fig.colorbar(dist,pad = .15)
# cbar.ax.set_ylabel(r'$\mathrm{\Delta}$ OASPL, dB (re: 20$\mathrm{\mu}$Pa)')
cbar.ax.set_ylabel(r'$\mathrm{\Delta}$ RAISPL, dB (re: 20$\mathrm{\mu}$Pa)')
cbar.ax.set_yticks(cbar_ticks[::2])
ax.set_yticks(phi_ticks)
ax.set_yticklabels(phi_ticklabels)
ax.scatter(acs_data[case]['theta'][obs_ind[0],obs_ind[1]],acs_data[case]['rho'][obs_ind[0],obs_ind[1]],c = 'white',marker = '^',s = 40)

plt.savefig(os.path.join(case_dir,f'hemi.png'),format = 'png',dpi = 600)
plt.close()


for pnt_itr,pnt in enumerate(select_points):
    fig,ax = plt.subplots(1,1, figsize = (2.5,2.5/1.25))
    plt.subplots_adjust(left = .25,bottom = .21,top = 0.875)
    phi_ind = np.abs(pnt[1]*np.pi/180-acs_data[case]['phi'][0]).argmin()
    theta_ind = np.abs(pnt[0]*np.pi/180-acs_data[case]['theta'][:,0]).argmin()
    for case_itr,case in enumerate(case_name):
        ax.plot(acs_data[case]['function_values'][theta_ind,phi_ind,:,0]/acs_data[case]['function_values'][theta_ind,phi_ind,-1,0],np.roll(acs_data[case]['function_values'][theta_ind,phi_ind,:,-1],-95),linestyle= linestyle[case_itr])
    ax.set(ylabel = r'p [Pa]', xlabel =r'Rev. Fraction',xlim = [0,1],ylim = [-10,10])
    ax.set_title(rf'$\mathrm{{O{pnt_itr+1}}} \ (\psi={int(np.round(acs_data[case]['theta'][theta_ind,phi_ind]*180/np.pi))}^\circ, \phi={int(np.round(acs_data[case]['phi'][theta_ind,phi_ind]*180/np.pi))}^\circ)$', fontdict={'fontsize': 11})
    ax.grid()
    ax.legend(['Baseline','Treated'],loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5)
    # ax.legend(['Baseline','Treated ($\mathrm{\mathcal{R}/|\mathcal{R}|}$)'],loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5)
    t = ax.text(0.7, .9, rf'$\Delta$ dB={np.round((acs_data[case_name[1]]['oaspl'][theta_ind,phi_ind]-acs_data[case_name[0]]['oaspl'][theta_ind,phi_ind]),1)} dB', ha="center",va="center",transform=ax.transAxes,fontsize=9)
    t.set_bbox(dict(facecolor='white' ,edgecolor = 'gray' ,alpha=.6,boxstyle='round'))

    plt.savefig(os.path.join(case_dir,f'p_tseries_{pnt_itr}.png'),format = 'png',dpi = 600)
    plt.close()


