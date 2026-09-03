#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))
from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  
import plot_styles
import matplotlib.colors as mcolors

#%%

# def process_wopwop(cases_directory,cases = 'cases.nam'):
#     f1 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'pressure')
#     f2 = lambda a: extract_wopwop_quant(case_directory=a, prefix = 'spl_spectrum')
#     apply_to_namelist([f1], cases_directory=cases_directory, cases=cases)

#%%

case_dir = os.getcwd()
cases = [i for i in os.listdir(case_dir) if os.path.isdir(i)]

data = {}

rho = 1.225
sos = 343
CT = np.zeros(len(cases))
MT = np.zeros(len(cases))
DL = np.zeros(len(cases))
OASPL= np.zeros(len(cases))
for case_itr,case in enumerate(cases):
    data = np.genfromtxt(os.path.join(case_dir,case,f'{case}.1prop.dat'),names=True)
    CT[case_itr] = data['THRUST']/(rho*np.pi*data['RADIUS']**2*(data['OMEGA']*data['RADIUS'])**2)
    MT[case_itr] = data['OMEGA']*data['RADIUS']/sos
    DL[case_itr] = data['THRUST']/(np.pi*data['RADIUS']**2)

    if not os.path.exists(os.path.join(case_dir,case,f'{case}.1PSU-WOPWOP',f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(case_dir,case,f'{case}.1PSU-WOPWOP'),cases ='cases.nam')

    acs_data = import_results_from_wopwop(cases_directory=os.path.join(case_dir,case,f'{case}.1PSU-WOPWOP'))
    OASPL[case_itr] = np.mean(np.round(10*np.log10(np.mean(acs_data['function_values'][...,-1]**2,axis = -1)/20e-6**2),2))

MT_sort_ind = MT.argsort().reshape((np.sqrt(len(cases))*np.ones(2)).astype(int))
sorted_ind = np.take_along_axis(MT_sort_ind,CT[MT_sort_ind].argsort(axis = -1),axis = -1)

cmap = plt.cm.get_cmap('inferno')

# levels = np.linspace(np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10,int(np.diff((np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10))[0]*1+1))
# levels_c = np.linspace(np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10,int(np.diff((np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10))[0]*1+1))
levels = np.linspace(70,120,51)

fig,ax = plt.subplots(1,1,  figsize = (2.9,2/3*2.9))
plt.subplots_adjust(left = .175,top = .9,right = .9,bottom = .22)
dist = ax.contourf(MT[sorted_ind],CT[sorted_ind],OASPL[sorted_ind],levels = levels,cmap = cmap)
# dist2 = ax.contour(MT[sorted_ind],CT[sorted_ind],DL[sorted_ind],colors = 'k',linestyles = '-.')
# plt.clabel(dist2)
dist2 = ax.contour(MT[sorted_ind],CT[sorted_ind],OASPL[sorted_ind],colors = 'k',linestyles = '-.')
plt.clabel(dist2)
ax.scatter(588.15*(0.4699/2)/343,.009095,marker = '*',s = 100,c = 'white')
cbar = fig.colorbar(dist,pad = .05)
cbar.ax.set_ylabel(r'$RAI \ SPL, \ dB$')
cbar.set_ticks(levels[::10])
ax.set(ylabel =r'$C_T$',xlabel =r'$M_T$' ,ylim = [0.4e-2,0.015],xlim = [0.3,0.8])
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
plt.savefig(os.path.join(case_dir,f'CT_MT_OASPL_{os.path.basename(case_dir)}.pdf'),format = 'pdf',bbox_inches = 'tight',pad_inches=.05)
plt.close()




