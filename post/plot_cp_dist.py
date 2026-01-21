#!/usr/bin/env python3
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

from dependencies.pyWopwop.wopwop import *  
from dependencies.pyWopwop.wopwop_io import *  
from dependencies.resonator import *  
from res_funcs import *

#%%
case_dir = "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/charm/rotor_af_interaction/quickROD_DN05_IAERO1/quickROD.1PSU-WOPWOP"
# set to true if rod is positioned below the rotor in its wake
below = True

# loading data file name
loading_fname = 'loading0200.dat'
# patch data file name
geometry_fname = 'geometry0200.dat'

# nondimensionalized spanwise extents (x-direction) over which to filter the blade loads
r_lim =[0.01,0.99]

#%%
loading = functional_data(os.path.join(case_dir,loading_fname),endianness='big')

if loading.header.data_type == 2:
    dt = loading.zones[0].period/loading.zones[0].N_time_steps
    f = np.arange(int(loading.zones[0].N_time_steps/2))*1/loading.zones[0].period
#%%
geometry = patch_data(os.path.join(case_dir,geometry_fname),endianness='big')

rng = geometry.zones[0].nodes[0].max(axis =0)-geometry.zones[0].nodes[0].min(axis =0)
span_axis = np.argmax(rng)
r = np.linalg.norm(geometry.zones[0].nodes[0][...,1:],axis = -1)
phi = np.arctan2(geometry.zones[0].nodes[0][...,2],geometry.zones[0].nodes[0][...,1])%(2*np.pi)

select_r_ind = (((geometry.zones[0].nodes[0][:,span_axis]-geometry.zones[0].nodes[0][:,span_axis].min())/(geometry.zones[0].nodes[0][:,span_axis].max()-geometry.zones[0].nodes[0][:,span_axis].min())) >= r_lim[0]) & (((geometry.zones[0].nodes[0][:,span_axis]-geometry.zones[0].nodes[0][:,span_axis].min())/(geometry.zones[0].nodes[0][:,span_axis].max()-geometry.zones[0].nodes[0][:,span_axis].min())) <=r_lim[1])
R = r[(select_r_ind)].mean()

#%%
phi_unq= np.unique(phi[select_r_ind])
# r_unq= np.unique(r[select_r_ind])
# phi_meshgrid,r_meshgrid = np.meshgrid(phi_unq,r_unq)

phi_grid = phi[select_r_ind].reshape((int(len(phi_unq)),int(len(phi[select_r_ind])/len(phi_unq))),order = 'F')
r_grid = geometry.zones[0].nodes[0][...,span_axis][select_r_ind].reshape((int(len(phi_unq)),int(len(phi[select_r_ind])/len(phi_unq))),order = 'F')
data = loading.zones[0].data[:,select_r_ind].reshape((len(loading.zones[0].data),int(len(phi_unq)),int(len(phi[select_r_ind])/len(phi_unq)),3),order = 'F')

t_ind = 13

sorted_ind = np.argsort(phi_grid,axis = 0)
phi_sort = np.take_along_axis(phi_grid, sorted_ind, axis=0)
r_sort = np.take_along_axis(r_grid, sorted_ind, axis=0)
# data_sort = np.take_along_axis(np.linalg.norm(data[t_ind,:,:],axis = -1), sorted_ind, axis=0)
data_sort = np.take_along_axis(np.gradient(data,axis = 0,edge_order=2)[t_ind,:,:,-1], sorted_ind, axis=0)

cmap = plt.cm.Spectral.reversed()
levels = np.linspace(-5e3,5e3,81)

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15,left = 0.15)
dist = ax.contourf(abs(r_sort),phi_sort,data_sort,levels = levels,cmap = cmap)
cbar = fig.colorbar(dist)
cbar.ax.set_ylabel(r'$P \ [Pa]$')
ax.set(xlabel = r'$r_{rod} \ [m]$',ylabel = r'$\phi_{rod} \ [rad]$',title = rf"$\psi_b = {loading.zones[0].keys[t_ind]}^\circ$")

plt.savefig(os.path.join(case_dir,f'rod_cp_t{loading.zones[0].keys[t_ind]}.png'),format = 'png')
plt.close()
# cbar.ax.set_yticks(cbar_ticks)
# ax.set_rlim([0,saved_params['r'][-1]])

max_ind = [np.where(np.linalg.norm(loading.zones[0].data,axis = -1)==np.linalg.norm(loading.zones[0].data,axis = -1).max())][0]

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15,left = 0.15)
# ax.plot(loading.zones[0].keys/loading.zones[0].keys[-1],np.linalg.norm(loading.zones[0].data[:,max_ind[1][0]],axis = -1))
ax.plot(loading.zones[0].keys/loading.zones[0].keys[-1],np.gradient(np.linalg.norm(loading.zones[0].data[:,max_ind[1][0]],axis = -1),edge_order=2))
ax.set(xlabel = r'$Rev \ Fraction$',ylabel = r'$\partial p/\partial \psi \ [Pa/deg]$',title = rf"$\psi_b = {loading.zones[0].keys[t_ind]}^\circ$",xlim = [0,1])
ax.grid()
plt.savefig(os.path.join(case_dir,f'dcp_dt_t.png'),format = 'png')
plt.close()
