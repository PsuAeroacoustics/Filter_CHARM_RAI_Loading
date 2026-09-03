#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))

from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  
from scipy.signal import welch
import plot_styles
from matplotlib import cm

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.',':']
marker = ['o','^','*']
#%%
R = 0.235
case_name = ['quickROD_DN05_IAERO1/quickROD.1PSU-WOPWOP_UJACK']
case_dir = os.path.join(os.getcwd())
acs_data ={}

for case in case_name:
    
    if not os.path.exists(os.path.join(case_dir,f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(case_dir,case),cases = 'cases.nam')

    acs_data.update({case:import_results_from_wopwop(cases_directory=os.path.join(case_dir,case))})
    acs_data[case]['geometry_values'] = acs_data[case]['geometry_values'][:,:,0].reshape((8,int(len(acs_data[case]['geometry_values'])/8),3))
    oaspl = (10*np.log10(np.mean(acs_data[case]['function_values'][...,-1]**2,axis = -1).squeeze()/20e-6**2)).reshape((8,int(len(acs_data[case]['function_values'])/8)))

    theta = np.arctan2(acs_data[case]['geometry_values'][:,:,1],acs_data[case]['geometry_values'][:,:,0])
    phi = np.arctan2(acs_data[case]['geometry_values'][...,-1],acs_data[case]['geometry_values'][...,1])*180/np.pi
    
    acs_data[case].update({'theta':theta,'phi':phi,'oaspl':oaspl})
    
# cmap = cm.get_cmap('inferno', 8)
cmap = cm.inferno(np.linspace(0, 1, 8))

leg_labs = [rf'$\phi={i}^\circ$' for i in (phi[:,0]%360).astype(int)]
fig,ax = plt.subplots(1,1, figsize = (3*.95,2/3*3*.95))
plt.subplots_adjust(left = .2,bottom = .225,top = 0.95)
for i in range(8):  
    ax.scatter(acs_data[case]['geometry_values'][i,:,1]/R,acs_data[case]['geometry_values'][i,:,-1]/R,c = cmap[i],cmap = cmap)
ax.set(ylabel = r'z/R', xlabel =r'y/R',xlim = [-20,20],ylim = [-20,20])
ax.grid()
plt.savefig(os.path.join(case_dir,case,f'ff_observers.pdf'),format = 'pdf',bbox_inches='tight')
plt.close()

fig,ax = plt.subplots(1,1, figsize = (3*.95,2/3*3*.95))
plt.subplots_adjust(left = .2,bottom = .225,top = 0.95,right = 0.95)
# ax.plot(np.linalg.norm(acs_data[case]['geometry_values']/R,axis = -1).T,oaspl.T)
for i in range(8):
    ax.plot(np.linalg.norm(acs_data[case]['geometry_values']/R,axis = -1)[i],oaspl[i],color=cmap[i],linestyle = linestyle[i%len(linestyle)])
ax.set(ylabel = r'OASPL, dB', xlabel =r'D/R',xscale = 'log',ylim = [75,140],xlim = [1,30])
ax.legend(leg_labs,loc='upper right',ncol = 2,fontsize=9,borderaxespad=0.25,handletextpad=0.5,handlelength=1.2,columnspacing=0.45,borderpad=0.3)
ax.grid()
plt.savefig(os.path.join(case_dir,case,f'ff_oaspl.pdf'),format = 'pdf',bbox_inches='tight')
plt.close()

# ax.legend(
#     fontsize=8,
#     ncol=2,
#     labelspacing=0.25,
#     handlelength=1.2,
#     handletextpad=0.4,
#     columnspacing=0.8,
#     borderpad=0.3,
#     frameon=False,
# )

