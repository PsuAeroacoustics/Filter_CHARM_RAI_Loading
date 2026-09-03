#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__))))
from help_funcs import read_results_from_h5

import matplotlib.pyplot as plt
import plot_styles
import matplotlib.colors as mcolors
from matplotlib import cm

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.',':']
marker = ['o','^','*']

case_name = "quickROD_DN05_SDOF_DIST_OAR15_OPT/quickROD.1PSU-WOPWOP_HEMI_LOW_PHI1_NR5_BP"

mdof = False
case_dir = os.path.join(os.getcwd())
saved_params = read_results_from_h5(os.path.join(case_dir,case_name))

# saved_params['Z_smeared'] = 1j*np.imag(saved_params['Z_smeared'])
R = (saved_params['Z_smeared']-1)/(saved_params['Z_smeared']+1)

# cmap = cm.get_cmap('inferno', 8)
cmap = cm.inferno(np.linspace(0, .85, len(saved_params['patch_types'])))

N_patches = len(saved_params['res_params']['l'])
plot_order = saved_params['res_params']['l'].squeeze().argsort()
if N_patches>1:
    leglab = [f"$l={np.round(l,1)}~mm$" for l in saved_params['res_params']['l'][plot_order].squeeze()*1e3]

fig,ax = plt.subplots(2,1, figsize = (3,3))
plt.subplots_adjust(bottom = 0.15,left = 0.175,top =0.95,right = 0.95)
if N_patches==1:
    ax[0].plot(saved_params['f'],np.real(saved_params['Z_smeared']),c = cmap[0],linestyle = linestyle[0])
    ax[1].plot(saved_params['f'],np.imag(saved_params['Z_smeared']),c = cmap[0],linestyle = linestyle[0])
else:
    for i,patch_itr in enumerate(saved_params['res_params']['l'].squeeze().argsort()):
        ax[0].plot(saved_params['f'],np.real(saved_params['Z_smeared'][:,patch_itr]),c = cmap[i],linestyle = linestyle[i])
        ax[1].plot(saved_params['f'],np.imag(saved_params['Z_smeared'][:,patch_itr]),c = cmap[i],linestyle = linestyle[i])

ax[0].set_xticklabels([])
ax[0].set_ylabel(r'$\mathrm{Resistance}, \ \overline{\theta}$')
ax[0].set_xlim([0,5e3])
ax[0].set_ylim([0,10])
ax[0].grid()
ax[-1].set_ylabel(r'$\mathrm{Reactance}, \ \overline{\chi}$')
ax[-1].set_xlabel(r'Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([0,5e3])
ax[-1].set_ylim([-5, 5])
if N_patches>1:
    ax[-1].legend(leglab,loc='lower right',ncol = 1,fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)


fig,ax = plt.subplots(2,1, figsize = (3,3))
plt.subplots_adjust(bottom = 0.15,left = 0.21,top =0.95,right = 0.95)
if N_patches==1:
    ax[0].plot(saved_params['f'],np.abs(R.squeeze()),c = cmap[0],linestyle = linestyle[0])
    ax[1].plot(saved_params['f'],np.unwrap(np.angle(R.squeeze())),c = cmap[0],linestyle = linestyle[0])
else:
    for i,patch_itr in enumerate(saved_params['res_params']['l'].squeeze().argsort()):
        ax[0].plot(saved_params['f'],np.abs(R[:,patch_itr]),c = cmap[i],linestyle = linestyle[i])
        ax[1].plot(saved_params['f'],np.unwrap(np.angle(R[:,patch_itr])),c = cmap[i],linestyle = linestyle[i])

ax[0].set(ylabel = r'$\mathrm{Reflection}, \ |\mathcal{R}|$',xlim = [0,5e3],ylim = [0,1],xticklabels = [])
ax[0].grid()
ax[1].set(ylabel = r'$\mathrm{Phase}, \ \varphi \ [rad]$',xlabel = r'Frequency [Hz]',xlim = [0,5e3],ylim = [-3*np.pi,np.pi/4])
ax[1].grid()


# ax[-1].set_ylim([-5, 5])
if N_patches>1:
    ax[-1].legend(leglab,loc='lower right',ncol = 1,fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)
plt.savefig(os.path.join(case_dir,'Z_resp.pdf'),format = 'pdf',pad_inches=.05,bbox_inches='tight')
plt.savefig(os.path.join(case_dir,'Z_resp.png'),format = 'png',dpi=400)
plt.close()

# ax[0,0].set_xticklabels([])
# phase = np.zeros(len(saved_params['f']))
# wrap_ind = np.insert(np.where(np.diff(np.angle(R[:,patch_itr]))>0)[0],0,0)
# for ind in wrap_ind[:-1]:
#     phase = np.angle(R[:,patch_itr])[ind-1]-np.angle(R[:,patch_itr])[ind:ind+1]


fig,ax = plt.subplots(2,2, figsize = (6.5*.95,2/3*6.5*.95))
# plt.subplots_adjust(left=0.1,right = .975,top = 0.95,bottom=0.175,hspace = 0.1,wspace = 0.325)
if N_patches==1:
    plt.subplots_adjust(left=0.1,right = .975,top = 0.95,bottom=0.12,hspace = 0.1,wspace = 0.325)
    ax[0,0].plot(saved_params['f'],np.real(saved_params['Z_smeared']),c = cmap[0],linestyle = linestyle[0])
    ax[1,0].plot(saved_params['f'],np.imag(saved_params['Z_smeared']),c = cmap[0],linestyle = linestyle[0])
    ax[0,1].plot(saved_params['f'],np.abs(R.squeeze()),c = cmap[0],linestyle = linestyle[0])
    ax[1,1].plot(saved_params['f'],np.unwrap(np.angle(R.squeeze())),c = cmap[0],linestyle = linestyle[0])

else:
    plt.subplots_adjust(left=0.1,right = .975,top = 0.95,bottom=0.175,hspace = 0.1,wspace = 0.325)
    for i,patch_itr in enumerate(saved_params['res_params']['l'].squeeze().argsort()):
        ax[0,0].plot(saved_params['f'],np.real(saved_params['Z_smeared'][:,patch_itr]),c = cmap[i],linestyle = linestyle[i])
        ax[1,0].plot(saved_params['f'],np.imag(saved_params['Z_smeared'][:,patch_itr]),c = cmap[i],linestyle = linestyle[i])
        ax[0,1].plot(saved_params['f'],np.abs(R[:,patch_itr]),c = cmap[i],linestyle = linestyle[i])
        ax[1,1].plot(saved_params['f'],np.unwrap(np.angle(R[:,patch_itr]),period =2*np.pi,discont=np.pi/2+np.pi/4),c = cmap[i],linestyle = linestyle[i])
        # ax[1,1].plot(saved_params['f'],np.angle(R[:,patch_itr]),c = cmap[i],linestyle = linestyle[i])
        # ax[1,1].plot(saved_params['f'],phase,c = cmap[i],linestyle = linestyle[i])

# ax[0,0].set_xticklabels([])
ax[0,0].set(ylabel = r'$\mathrm{Resistance}, \ \overline{\theta}$',xlim = [0,5e3],ylim = [0,10],xticklabels = [])
ax[0,0].grid()
ax[1,0].set(ylabel = r'$\mathrm{Reactance}, \ \overline{\chi}$',xlabel = r'Frequency [Hz]',xlim = [0,5e3],ylim = [-40,40])
ax[1,0].grid()
ax[0,1].set(ylabel = r'$\mathrm{Reflection}, \ |\mathcal{R}|$',xlim = [0,5e3],ylim = [0,1],xticklabels = [])
ax[0,1].grid()
ax[1,1].set(ylabel = r'$\mathrm{Phase}, \ \varphi \ [rad]$',xlabel = r'Frequency [Hz]',xlim = [0,5e3],ylim = [-np.pi/4,np.pi/4])
ax[-1,1].grid()
if N_patches>1:
    fig.legend(leglab,ncol = N_patches,loc='lower center',fontsize=9,bbox_to_anchor=(.5, -0.01),borderaxespad=0.25,handletextpad=0.5,handlelength=1.5,columnspacing=1.2,borderpad=0.3)
plt.savefig(os.path.join(case_dir,'Z_resp.pdf'),format = 'pdf',pad_inches=.05,bbox_inches='tight')
plt.close()


# ylim = [-2*np.pi-np.pi/4,np.pi/4]
