#!/usr/bin/env python3
import os
import sys
import plot_styles
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))
from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  

from matplotlib import cm

# from dependencies.resonator import *  
# from res_funcs import *

#%%

case_name = ["quickROD.1PSU-WOPWOP_HEMI_LOW_ROD_ONLY_COMPACT","quickROD.1PSU-WOPWOP_HEMI_LOW_PHI1_NR5_ROD_ONLY_COMPACT_PHASE_ONLY"]
case_dir = os.getcwd()

# loading data file name
loading_fname = ['loading0200_mod_line.dat','loading0200_sdof_dist_oar15_line.dat']
# patch data file name
geometry_fname = 'geometry0200_line.dat'

loading = {}
geometry = {}

for i,case in enumerate(case_name):
    loading.update({case:functional_data(os.path.join(case_dir,case,loading_fname[i]),endianness='big')})
    geometry.update({case:patch_data(os.path.join(case_dir,case,geometry_fname),endianness='big')})

cmap = cm.inferno(np.linspace(0, .85, loading[case].zones[0].data[:,5::10,-1].shape[-1]))

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15,left = 0.15)
for i in range(loading[case].zones[0].data[:,5::10,-1].shape[-1]):
    ax.plot(loading[case_name[0]].zones[0].keys/loading[case].zones[0].keys[-1],np.gradient(loading[case_name[0]].zones[0].data[:,5::10,-1][:,i],edge_order=2,axis = 0),c = cmap[i])
    ax.plot(loading[case].zones[0].keys/loading[case].zones[0].keys[-1],np.gradient(loading[case].zones[0].data[:,5::10,-1][:,i],edge_order=2,axis = 0),linestyle = '-.',c = cmap[i])
ax.set(xlabel = r'$Rev \ Fraction$',ylabel = r'$\partial p/\partial \psi \ [Pa/deg]$',xlim = [0,1])
ax.grid()
