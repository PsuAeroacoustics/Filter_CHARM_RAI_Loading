#!/usr/bin/env python3
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import welch

from dependencies.pyWopwop.wopwop import *  
from dependencies.pyWopwop.wopwop_io import *  
from dependencies.resonator import *  
from res_funcs import *

#%%
case_dir = "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/charm/rotor_af_interaction/quickROD_SDOF_GEOM_OAR15/quickROD.1PSU-WOPWOP"
# set to true if rod is positioned below the rotor in its wake
below = False

# loading data file name
loading_fname = 'loading0200.dat'
# patch data file name
geometry_fname = 'geometry0200.dat'
# resonator parameter json file name
res_params_fname = 'sdof_geom_param.json'

# nondimensionalized spanwise extents (x-direction) over which to filter the blade loads
r_lim =[0.025,0.975]
# nondimensionalized radial extents (y-direction) over which to filter the blade loads
c_lim = [0.15,0.85]

# number of resonators (number of nodes to filter)
N_res = 209
# set to true to plot the response of the resonators
plot = False

#%%

loading = functional_data(os.path.join(case_dir,loading_fname),endianness='big')

if loading.header.data_type == 2:
    dt = loading.zones[0].period/loading.zones[0].N_time_steps
    f = np.arange(1,int(loading.zones[0].N_time_steps/2)+1)*1/loading.zones[0].period
#%%
geometry = patch_data(os.path.join(case_dir,geometry_fname),endianness='big')

#%%
rng = geometry.zones[0].nodes[0].max(axis =0)-geometry.zones[0].nodes[0].min(axis =0)
span_axis = np.argmax(rng)
r = np.linalg.norm(geometry.zones[0].nodes[0][...,1:],axis = -1)
phi = np.arctan2(geometry.zones[0].nodes[0][...,2],geometry.zones[0].nodes[0][...,1])%(2*np.pi)

select_r_ind = (((geometry.zones[0].nodes[0][:,span_axis]-geometry.zones[0].nodes[0][:,span_axis].min())/(geometry.zones[0].nodes[0][:,span_axis].max()-geometry.zones[0].nodes[0][:,span_axis].min())) >= r_lim[0]) & (((geometry.zones[0].nodes[0][:,span_axis]-geometry.zones[0].nodes[0][:,span_axis].min())/(geometry.zones[0].nodes[0][:,span_axis].max()-geometry.zones[0].nodes[0][:,span_axis].min())) <=r_lim[1])
R = r[(select_r_ind)].mean()

if below:
    phi_lim =-np.asarray([np.arccos(-(1-c_lim[0])),np.arccos(c_lim[1])])%(2*np.pi)
else:
    phi_lim =[np.arccos(c_lim[1]),np.arccos(-(1-c_lim[0]))]

phi_lim = [1*np.pi-0*np.pi/180,2*np.pi+0*np.pi/180]
select_phi_ind = (phi>=phi_lim[0]) & (phi<=phi_lim[1])
select_ind = np.where((select_r_ind & select_phi_ind))[0]

# rng = np.random.default_rng()
# del_ind = rng.choice(len(select_ind),len(select_ind)-N_res,replace=False, shuffle=False)
# select_ind = np.delete(select_ind,del_ind)
#%%

with open(os.path.join(os.getcwd(),res_params_fname)) as res_file:
    res_params = json.load(res_file)
res_params['OAR'] = .15
A_s = r[(select_r_ind & select_phi_ind)].mean()*(phi[(select_r_ind & select_phi_ind)].max()-phi[(select_r_ind & select_phi_ind)].min())*np.abs(geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),span_axis].max()-geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),span_axis].min())
# # A_s = np.linalg.norm(geometry.zones[0].norms[0],axis = -1)[select_r_ind & select_phi_ind].mean()
out = get_sample_info(np.asarray([True]),np.asarray([A_s]),**res_params)

#%%
# for i in range(res_params['N_patches']):
#     Z = smeared_Z(f,A_s,**out[f'patch_{i}'])
# R = ((Z-1)/(Z+1)).squeeze()

Z = init_res(f,a_n = res_params['x'][0],L_n = res_params['x'][1]/2,a_c = res_params['x'][0],L_c = res_params['x'][1]/2).Z

filt_resp = (get_filt_resp((Z/res_params['OAR'])[:,None]))
filt_data = apply_filt(loading.zones[0].data[:,select_ind],np.abs(filt_resp))


loading.zones[0].data[:,select_ind] = filt_data
loading.zones[0].data[...,0] = 0.0
# loading.zones[0].data[...,-1] = 0.0
loading_fname_split = os.path.splitext(loading_fname)
loading.file_dir = os.path.join(case_dir,f'{loading_fname_split[0]}_mdof_geom_oar15{loading_fname_split[1]}')
loading.write(ascii=False)

R = (Z-1)/(Z+1)
# r_obs = np.asarray([[ 0.18097501, -1.524     , -0.65760421],
#        [ 0.18097501, -1.524     ,  -0.04064       ],
#        [ 0.18097501, -1.524     ,  0.63089029]])

# lr = np.sum(loading.zones[0].data[...,None]*(r_obs.T/np.linalg.norm(r_obs,axis = -1)),axis = -2)
# dlr_dt = np.sum(np.gradient(loading.zones[0].data,axis = 0)[...,None]*(r_obs.T/np.linalg.norm(r_obs,axis = -1)),axis = -2)
# dlr_dt_filt = np.sum(np.gradient(filt_data[...,None],axis = 0)*(r_obs.T/np.linalg.norm(r_obs,axis = -1)),axis = -2)

# #%%
# from scipy.special import jv
# a = res_params['x'][0]
# L = res_params['x'][1]
# sos = 343
# nu = 14.88e-6
# Pr = 0.707
# k = 2*np.pi*f/sos
# alpha = 1j**(3/2)*a*np.sqrt(2*np.pi*f/nu)
# n = (1+(1.4-1)/1.4*(jv(2,alpha*np.sqrt(Pr))/jv(0,alpha*np.sqrt(Pr))))**-1
# phi = k*np.sqrt(jv(0,alpha)/jv(2,alpha)*1.4/n)
# filt_resp_2 = get_filt_resp_2((1+np.exp(phi*L)/np.exp(-phi*L))[:,None])
# filt_data = apply_filt(loading.zones[0].data[:,select_ind],np.conj(filt_resp_2))

# fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
# plt.subplots_adjust(bottom = 0.15,left = 0.15)
# ax[0].plot(f,np.abs(filt_resp[1:len(f)+1]))
# ax[0].set(xticklabels = [],xlim = [0,5e3],ylim =[0,1],ylabel = r'$Reflection, \ |\mathit{R}|$' )
# ax[0].grid()
# ax[1].plot(f,np.unwrap(np.angle((filt_resp)[1:len(f)+1].squeeze()))*180/np.pi)
# ax[1].set(xlim = [0,5e3],ylabel = r'$Phase, \ \phi \ [rad]$',xlabel =r'Frequency [Hz]')
# ax[1].grid()

# fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
# plt.subplots_adjust(bottom = 0.15,left = 0.15)
# ax[0].plot(f,np.abs(np.conj(filt_resp_2)[1:len(f)+1]))
# ax[0].set(xticklabels = [],xlim = [0,5e3],ylim =[0,1],ylabel = r'$Reflection, \ |\mathit{R}|$' )
# ax[0].grid()
# ax[1].plot(f,np.unwrap(np.angle(np.conj(filt_resp_2)[1:len(f)+1].squeeze()))*180/np.pi)
# ax[1].set(xlim = [0,5e3],ylabel = r'$Phase, \ \phi \ [rad]$',xlabel =r'Frequency [Hz]')
# ax[1].grid()

#%%

if plot:
        ind = np.where(geometry.zones[0].nodes[0][select_ind,0] == np.unique(geometry.zones[0].nodes[0][select_ind,0])[-2])[0]
        ind = np.sqrt(np.mean(loading.zones[0].data[:,(select_r_ind & select_phi_ind),1]**2,axis = 0)).argmax()
        
        fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax[0].plot(f,np.real(Z))
        ax[0].set(xticklabels = [],xlim = [0,5e3],ylim =[0,20],ylabel = r'$Resistance, \ \overline{\theta}$' )
        ax[0].grid()
        ax[1].plot(f,np.imag(Z))
        ax[1].set(xlim = [0,None],ylim =[-10,10],ylabel = r'$Reactance, \ \overline{\chi}$',xlabel =r'Frequency [Hz]')
        ax[1].grid()
        plt.savefig(os.path.join(case_dir,f'Z.png'),format = 'png')
        plt.close()


        fig,ax = plt.subplots(2,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax[0].plot(f,np.abs(filt_resp[1:len(f)+1]))
        ax[0].set(xticklabels = [],xlim = [0,5e3],ylim =[0,1],ylabel = r'$Reflection, \ |\mathit{R}|$' )
        ax[0].grid()
        ax[1].plot(f,np.unwrap(np.angle((filt_resp)[1:len(f)+1].squeeze()))*180/np.pi)
        ax[1].set(xlim = [0,5e3],ylabel = r'$Phase, \ \phi \ [rad]$',xlabel =r'Frequency [Hz]')
        ax[1].grid()
        plt.savefig(os.path.join(case_dir,f'R.png'),format = 'png')
        plt.close()
        
        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind,-1])
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.linalg.norm(filt_data[:,ind],axis = -1))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$P \ [Pa]$',xlim = [0,1])
        ax.grid()
        ax.legend(['Untreated','Treated'])
        plt.savefig(os.path.join(case_dir,f'p_tseries.png'),format = 'png')

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind,1])
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,filt_data[:,ind,1])
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$P \ [Pa]$',xlim = [0,1])
        ax.grid()
        ax.legend(['Untreated','Treated'])

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient((loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind][:,1]),edge_order=2))
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient(filt_data[:,ind,1],edge_order=2))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$\partial P/\partial \psi \ [Pa/deg]$',xlim = [0,1])
        ax.grid()
        ax.legend(['Untreated','Treated'])
        plt.savefig(os.path.join(case_dir,f'dp_tseries.png'),format = 'png')


        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        # ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.linalg.norm(loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind],axis = -1))
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.linalg.norm(filt_data[:,ind],axis = -1))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$P \ [Pa]$',xlim = [0,1])
        ax.grid()
        ax.legend(['Untreated','Treated'])

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient(np.linalg.norm(loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind],axis = -1),axis = 0))
        # ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient(np.linalg.norm(filt_data[:,ind],axis = -1),axis = 0))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$P \ [Pa]$',xlim = [0,1])
        ax.grid()
        ax.legend(['Untreated','Treated'])

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient(loading.zones[0].data[:,(select_ind)][:,ind,0],axis = 0))
        # ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient(filt_data[:,np.invert(ind),2],axis = 0))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$P \ [Pa]$',xlim = [0,1])
        ax.grid()
        ax.legend(['Untreated','Treated'])

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,dlr_dt_filt[:,ind,0]-dlr_dt[:,select_ind][:,ind,0])
        # ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,dlr_dt[:,select_ind][:,ind,0])
        # ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,dlr_dt_filt[:,ind,0])

        # ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient(filt_data[:,np.invert(ind),2],axis = 0))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$P \ [Pa]$',xlim = [0,1])
        ax.grid()
        ax.legend(['Untreated','Treated'])

        f,pxx = welch(loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind,-1], fs=dt**-1, window='boxcar', nperseg=len(loading.zones[0].data), noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='mean')
        f,pxx_filt = welch(filt_data[:,ind,-1], fs=dt**-1, window='boxcar', nperseg=len(loading.zones[0].data), noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='mean')

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.stem(f,10*np.log10(pxx*np.diff(f[:2])),linefmt='C0-')
        ax.stem(f,10*np.log10(pxx_filt*np.diff(f[:2])),linefmt='C1-')
        ax.set(xscale = 'linear',ylim = [0,None])

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,-np.linalg.norm(loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind],axis = -1)+np.linalg.norm(filt_data[:,ind],axis = -1))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$\partial P/\partial \psi \ [Pa/deg]$',xlim = [0,1])
        ax.grid()

        fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
        plt.subplots_adjust(bottom = 0.15,left = 0.15)
        ax.plot(np.arange(loading.zones[0].N_time_steps)/loading.zones[0].N_time_steps,np.gradient((np.linalg.norm(loading.zones[0].data[:,(select_r_ind & select_phi_ind)][:,ind],axis = -1))-np.linalg.norm(filt_data[:,ind],axis = -1),edge_order=2))
        ax.set(xlabel = r'Rev Fraction', ylabel = r'$\partial P/\partial \psi \ [Pa/deg]$',xlim = [0,1])
        ax.grid()


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),0][ind],geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),1][ind],geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),2][ind],c = 'black')
        ax.scatter(geometry.zones[0].nodes[0][np.invert(select_r_ind & select_phi_ind),0],geometry.zones[0].nodes[0][np.invert(select_r_ind & select_phi_ind),1],geometry.zones[0].nodes[0][np.invert(select_r_ind & select_phi_ind),2])
        ax.scatter(geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),0],geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),1],geometry.zones[0].nodes[0][(select_r_ind & select_phi_ind),2])
        ax.scatter(geometry.zones[0].nodes[0][select_r_ind,0][445],geometry.zones[0].nodes[0][select_r_ind,1][445],geometry.zones[0].nodes[0][select_r_ind,2][445],c = 'black')

        ax.set(xlabel = 'x',ylabel = 'y',zlabel = 'z')
        ax.invert_zaxis()
        ax.set_box_aspect(np.abs(np.asarray((np.diff(ax.get_xlim()),np.diff(ax.get_ylim()),np.diff(ax.get_zlim())))).squeeze())
        plt.savefig(os.path.join(case_dir,f'filt_pnts.png'),format = 'png')
        plt.close()


        #%%
        t_ind = 0
        n_theta = 100
        n_z = 100
        R = (geometry.zones[0].nodes[0].max(axis = 0)-geometry.zones[0].nodes[0].min(axis = 0))[1]/2
        theta_grid = np.linspace(0, 2*np.pi, n_theta)
        x_grid     = np.linspace(geometry.zones[0].nodes[0][...,span_axis].min(), geometry.zones[0].nodes[0][...,span_axis].max(), n_z)
        T, X = np.meshgrid(theta_grid, x_grid)
        V_grid = griddata(points=np.column_stack((phi, geometry.zones[0].nodes[0][:,span_axis])),values=loading.zones[0].data[t_ind],xi=(T, X),method='cubic')
        Y = R * np.cos(T)
        Z = R * np.sin(T)
        V_grid[np.isnan(V_grid)] = 0


        cmap = plt.cm.Spectral.reversed()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        norm = plt.Normalize(np.mean(V_grid)-2*np.std(V_grid), np.mean(V_grid)+2*np.std(V_grid))
        dist = ax.plot_surface(X, Y,Z,rstride=1, cstride=1,facecolors=cmap(norm(V_grid)),shade=False)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6,pad = .2)
        cbar.ax.set_ylabel(r'$|F| \ [N]$')
        ax.set(xlabel = 'x',ylabel = 'y',zlabel = 'z')
        ax.invert_zaxis()
        ax.set_box_aspect(np.abs(np.asarray((np.diff(ax.get_xlim()),np.diff(ax.get_ylim()),np.diff(ax.get_zlim())))).squeeze())

