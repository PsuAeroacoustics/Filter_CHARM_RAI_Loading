#!/usr/bin/env python3
import os
import sys
import plot_styles
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))
from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__))))
from help_funcs import read_results_from_h5
from scipy import signal
from matplotlib import cm
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# from dependencies.resonator import *  
# from res_funcs import *
linestyle = ['-',':','--','-.',':']

#%%

case_name = ['quickROD.1PSU-WOPWOP_HEMI_LOW_PHI1_NR5_ROD_ONLY_COMPACT']
# loading data file name
loading_fname = ['loading0200_mod_line.dat']

# patch data file name
geometry_fname = 'geometry0200_line.dat'

obs_ind = [18,6]
leg_labs = []

filt = False
upper = False
lambert = True
A_weighting = False
raispl = True

baseline = False

sos = 343

figsize = (3,3/1.25)
skip = False
#%%
case_dir = os.getcwd()

loading = {}
geometry = {}
acs_data = {}
saved_params = {}

lambert = lambda x: np.sqrt(2)*r.mean()*np.sin((np.pi/2-x)/2)
stereo = lambda x: 2*r.mean()*np.tan((np.pi/2-x)/2)
phi_labels = (np.arange(7)/6*90)

for i,case in enumerate(case_name):
    loading.update({case:functional_data(os.path.join(case_dir,case,loading_fname[i]),endianness='big')})
    geometry.update({case:patch_data(os.path.join(case_dir,case,geometry_fname),endianness='big')})
    saved_params.update({case:read_results_from_h5(os.path.join(case_dir,case))})

    if not os.path.exists(os.path.join(case_dir,f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(case_dir,case),cases ='cases_hemi.nam')

    acs_data.update({case:import_results_from_wopwop(cases_directory=os.path.join(case_dir,case))})
    oaspl = np.round(10*np.log10(np.mean(acs_data[case]['function_values'][...,-1]**2,axis = -1)/20e-6**2),2)

    theta = np.arctan2(acs_data[case]['geometry_values'][:,:,0,1],acs_data[case]['geometry_values'][:,:,0,0])%(2*np.pi)
    theta[-1] = 2*np.pi
    phi = np.abs(np.arctan2(acs_data[case]['geometry_values'][:,:,0,-1],np.linalg.norm((acs_data[case]['geometry_values'][:,:,0,0],acs_data[case]['geometry_values'][:,:,0,1]),axis = 0)))

    dphi = np.abs(np.diff(phi[0,:-1]).mean()*180/np.pi)
    dtheta = np.abs(np.diff(theta[:-1,0]).mean()*180/np.pi)
    r = np.linalg.norm(acs_data[case]['geometry_values'][:,:,0],axis = -1)
    rho = lambert(phi)
    acs_data[case].update({'theta':theta,'phi':phi,'r':r,'rho':rho,'oaspl':oaspl,'dphi':dphi,'dtheta':dtheta})

rng = np.abs(geometry[case_name[0]].zones[0].nodes.squeeze()[0]-geometry[case_name[0]].zones[0].nodes.squeeze()[-1])
ds = np.abs(np.diff(geometry[case_name[0]].zones[0].nodes.squeeze()[:,rng.argmax()])).mean()

r = (acs_data[case_name[0]]['geometry_values'][:,:,0,:][...,None]-geometry[case_name[0]].zones[0].nodes.squeeze().T).transpose(0,1,-1,2)
d = np.linalg.norm(r,axis = -1)
r_hat = r/d[...,None]
t_delay = d/sos
N_pnts = len(loading[case_name[0]].zones[0].data)
T = (acs_data[case_name[0]]['function_values'][0,0,-1,0]-acs_data[case_name[0]]['function_values'][0,0,0,0])
dt =T/N_pnts
t_delay_ind = ((np.arange(N_pnts)-np.round(t_delay/dt).T[...,None]).T%N_pnts).astype(int)
if filt:
    sos = signal.butter(10, (T**-1*2*1.9,T**-1*2*12.1), 'bp', fs=dt**-1, output='sos')

for i,case in enumerate(case_name):
    dl_dt = np.gradient(-loading[case].zones[0].data,edge_order=2,axis = 0)/dt
    lr = np.einsum('tri,klri->tklr',-loading[case].zones[0].data,r_hat)
    dlr_dt = np.einsum('tri,klri->tklr',dl_dt,r_hat)
    dlr_dt_ret = np.take_along_axis(dlr_dt,t_delay_ind,axis = 0)
    p_sigma = 1/(4*np.pi*sos)*dlr_dt_ret/d
    p = 1/(4*np.pi*sos)*np.sum(dlr_dt_ret/d*ds,axis = -1)
    if filt:
        p = signal.sosfilt(sos, p,axis = 0)
    acs_data[case].update({'p':p,'p_sigma':p_sigma})

if len(obs_ind) ==0:
    obs_ind = np.asarray(np.where(acs_data[case_name[0]]['oaspl']==acs_data[case_name[0]]['oaspl'].max())).squeeze()

N_elements = loading[case_name[0]].zones[0].data.shape[1]
for i,case in enumerate(case_name):
    if baseline:
    # if 'res_params' not in saved_params[case]:
        # cmap = cm.get_cmap('inferno', 8)
        cmap = cm.inferno(np.linspace(0, .85, N_elements))
        
        fig,ax = plt.subplots(1,1, figsize =figsize)
        plt.subplots_adjust(left = .2,bottom = .15,right = 0.95,top = 0.9)
        for i in range(N_elements):
            ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.linalg.norm(loading[case].zones[0].data,axis = -1)[:,i],-24,axis = 0),c = cmap[i],alpha = 0.4)
        # ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.prod(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1]]*ds,axis = -1),c='red',linestyle = '-.')
        ax.set(ylabel = r'$|\bm{l}| \ [N]$', xlabel =r'Rev. Fraction',xlim = [0,1])
        ax.grid()
        ax.legend(leg_labs,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)

        fig,ax = plt.subplots(1,1, figsize = figsize)
        plt.subplots_adjust(left = .2,bottom = .15,right = 0.95,top = 0.9)
        for i in range(N_elements):
            ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.gradient(np.linalg.norm(loading[case].zones[0].data,axis = -1),axis = 0,edge_order=2)[:,i]*ds/dt,-24,axis = 0),c = cmap[i],alpha = 0.4)
        # ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.prod(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1]]*ds,axis = -1),c='red',linestyle = '-.')
        ax.set(ylabel = r'$\partial |\bm{l}|/\partial t \ [Ns^{-1}]$', xlabel =r'Rev. Fraction',xlim = [0,1])
        ax.grid()
        ax.legend(leg_labs,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)


        fig,ax = plt.subplots(1,1, figsize = figsize)
        plt.subplots_adjust(left = .2,bottom = .15,right = 0.95,top = 0.9)
        for i in range(N_elements):
            ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1],i]*ds,-24,axis = 0),c = cmap[i],alpha = 0.4)
        # ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.prod(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1]]*ds,axis = -1),c='red',linestyle = '-.')
        ax.set(ylabel = r'p [Pa]', xlabel =r'Rev. Fraction',xlim = [0,1])
        ax.grid()
        ax.legend(leg_labs,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)

    else:
        # N_patches = 5
        # leglab = ['$l=37.0~mm$', '$l=44.6~mm$', '$l=69.9~mm$', '$l=92.0~mm$', '$l=108.7~mm$']
        # plot_order = np.asarray([3, 4, 2, 1, 0])
        N_patches = len(saved_params[case]['res_params']['l'])
        # number of radial and circumfrential points per patch 
        N_r_patch = int(N_elements/N_patches)
        cmap = cm.inferno(np.linspace(0, .85, N_patches))
        if N_patches>1:
            plot_order = saved_params[case]['res_params']['l'].squeeze()[::-1].argsort()
            leglab = [f"$l={np.round(l,1)}~mm$" for l in saved_params[case]['res_params']['l'].squeeze()[::-1][plot_order]*1e3]
        else:
            plot_order = [0]
            leglab = [f"$l={np.round(saved_params[case]['res_params']['l'].squeeze()*1e3,1)}~mm$" ]

        if skip:
            skip_ind = int(N_patches/2)

        fig,ax = plt.subplots(1,1, figsize = figsize)
        plt.subplots_adjust(left = .2,bottom = .15,right = 0.95,top = 0.9)
        for i,plt_ind in enumerate(plot_order):
            if skip: 
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.linalg.norm(loading[case].zones[0].data[:,plt_ind*N_r_patch+skip_ind],axis = -1),-24,axis = 0),c = cmap[i],alpha = 0.4,label = leglab[i])
            else:
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.linalg.norm(loading[case].zones[0].data[:,plt_ind*N_r_patch],axis = -1),-24,axis = 0),c = cmap[i],alpha = 0.4,label = leglab[i])
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.linalg.norm(loading[case].zones[0].data[:,plt_ind*N_r_patch+1:(plt_ind+1)*N_r_patch],axis = -1),-24,axis = 0),c = cmap[i],alpha = 0.4,label="_nolegend_")

        ax.set(ylabel = r'$ |\bm{l}|$ [N]', xlabel =r'Rev. Fraction',xlim = [0,1],ylim = [0,300])
        ax.grid()
        ax.legend(leglab,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)
        plt.savefig(os.path.join(case_dir,'l_compact_loads.png'),format = 'png',dpi = 400)
        # plt.savefig(os.path.join(case_dir,'l_compact_loads.pdf'),format = 'pdf',pad_inches=.05,bbox_inches='tight')
        plt.close()

        fig,ax = plt.subplots(1,1, figsize = figsize)
        plt.subplots_adjust(left = .175,bottom = .15,right = 0.95,top = 0.9)
        for i,plt_ind in enumerate(plot_order):
            if skip: 
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.gradient(np.linalg.norm(loading[case].zones[0].data[:,plt_ind*N_r_patch+skip_ind],axis = -1),axis = 0,edge_order=2)*ds/dt,-24,axis = 0),c = cmap[i],alpha = 0.4,label = leglab[i])
            else:
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.gradient(np.linalg.norm(loading[case].zones[0].data[:,plt_ind*N_r_patch+1:(plt_ind+1)*N_r_patch],axis = -1)*ds/dt,axis = 0,edge_order=2),-24,axis = 0),c = cmap[i],alpha = 0.4,label="_nolegend_")
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.gradient(np.linalg.norm(loading[case].zones[0].data[:,plt_ind*N_r_patch],axis = -1),axis = 0,edge_order=2)*ds/dt,-24,axis = 0),c = cmap[i],alpha = 0.4,label = leglab[i])
        ax.set(ylabel = r'$\partial |\bm{l}| /\partial t \ [Ns^{-1}]$', xlabel =r'Rev. Fraction',xlim = [0,1],ylim = [-5e3,5e3])
        ax.grid()
        ax.legend(leglab,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        plt.savefig(os.path.join(case_dir,'dl_dt_compact_loads.png'),format = 'png',dpi=400)
        # plt.savefig(os.path.join(case_dir,'dl_dt_compact_loads.pdf'),format = 'pdf',pad_inches=.05,bbox_inches='tight')
        plt.close()

        fig,ax = plt.subplots(1,1, figsize =figsize)
        plt.subplots_adjust(left = .22,bottom = .15,right = 0.95,top = 0.9)
        for i,plt_ind in enumerate(plot_order):
            if skip: 
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1],plt_ind*N_r_patch+skip_ind]*ds,-24,axis = 0),c = cmap[i],alpha = 0.4,label = leglab[i])
            else:
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1],plt_ind*N_r_patch+1:(plt_ind+1)*N_r_patch]*ds,-24,axis = 0),c = cmap[i],alpha = 0.4,label="_nolegend_")
                ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1],plt_ind*N_r_patch]*ds,-24,axis = 0),c = cmap[i],alpha = 0.4,label = leglab[i])
        ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.roll(np.mean(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1]]*ds,axis = -1),-24,axis = 0),c = 'cyan',linewidth = 2)
        ax.set(ylabel = r'p [Pa]', xlabel =r'Rev. Fraction',xlim = [0,1],ylim = [-.7,.7])
        ax.grid()
        ax.legend(leglab,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)
        plt.savefig(os.path.join(case_dir,'compact_loads_p.png'),format = 'png',dpi=400)
        plt.savefig(os.path.join(case_dir,'compact_loads_p.pdf'),format = 'pdf',pad_inches=.05,bbox_inches='tight')
        plt.close()


print('done')



# fig,ax = plt.subplots(1,1, figsize = (3.5,3/4*3.5))
# plt.subplots_adjust(left = .2,bottom = .2)
# for i,case in enumerate(case_name):
#     ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],acs_data[case]['p'][:,obs_ind[0],obs_ind[1]],linestyle = linestyle[i])
# ax.set(ylabel = r'p [Pa]', xlabel =r'Rev. Fraction',xlim = [0,1])
# ax.grid()
# ax.legend(leg_labs,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)

# fig,ax = plt.subplots(1,1, figsize = (3.5,3/4*3.5))
# plt.subplots_adjust(left = .2,bottom = .2)
# for i,case in enumerate(case_name):
#     ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.gradient(acs_data[case]['p'][:,obs_ind[0],obs_ind[1]],edge_order=2),linestyle = linestyle[i])
# ax.set(ylabel = r'dp/dt [Pa/rad]', xlabel =r'Rev. Fraction',xlim = [0,1])
# ax.grid()
# ax.legend(leg_labs,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)

# for i,case in enumerate(case_name):
#     fig,ax = plt.subplots(1,1, figsize = (4.5,3/4*4.5))
#     plt.subplots_adjust(left = .2,bottom = .2)
#     for i in range(N_elements):
#         ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1],i]*ds,c = cmap[i],alpha = 0.4)
#     # ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.prod(acs_data[case]['p_sigma'][:,obs_ind[0],obs_ind[1]]*ds,axis = -1),c='red',linestyle = '-.')
#     ax.set(ylabel = r'p [Pa]', xlabel =r'Rev. Fraction',xlim = [0,1])
#     ax.grid()
#     ax.legend(leg_labs,ncol = 1,loc='lower right',fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)

