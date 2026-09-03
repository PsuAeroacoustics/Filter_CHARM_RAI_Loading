#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))
from pyWopwop.wopwop import *  
from pyWopwop.wopwop_io import *  
import plot_styles
import matplotlib.colors as mcolors

case_dir = os.getcwd()
cases = [i for i in os.listdir(case_dir) if os.path.isdir(i)]

#%%
R = 0.4699/2
N = 5
AR = 7
rho = 1.225
sos = 343

delta_min = 0.25
delta_max = 2
delta = (np.arange(N)/(N-1)*(delta_max-delta_min)+delta_min)

r_min = 0.25
r_max = 3
r = (np.arange(N)/(N-1)*(r_max-r_min)+r_min)[:-1]
#%%
data = {}

S = np.zeros(len(cases))
R = np.zeros(len(cases))
OASPL= np.zeros(len(cases))

for case_itr,case in enumerate(cases):
    data = np.genfromtxt(os.path.join(case_dir,case,f'{case}.1prop.dat'),names=True)
    case_name_split = case.split('_')
    R[case_itr]= r[int(case_name_split[1][-1])-1]
    S[case_itr]= delta[int(case_name_split[2][-1])-1]

    if not os.path.exists(os.path.join(case_dir,case,f'{case}.1PSU-WOPWOP',f'{case}.h5')):
        process_wopwop(cases_directory=os.path.join(case_dir,case,f'{case}.1PSU-WOPWOP'),cases ='cases.nam')

    acs_data = import_results_from_wopwop(cases_directory=os.path.join(case_dir,case,f'{case}.1PSU-WOPWOP'))
    OASPL[case_itr] = np.mean(np.round(10*np.log10(np.mean(acs_data['function_values'][...,-1]**2,axis = -1)/20e-6**2),2))

R_sort_ind = R.argsort().reshape((len(r),len(delta))).astype(int)
sorted_ind = np.take_along_axis(R_sort_ind,S[R_sort_ind].argsort(axis = -1),axis = -1)

cmap = plt.cm.get_cmap('inferno')

# levels = np.linspace(np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10,int(np.diff((np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10))[0]*1+1))
# levels_c = np.linspace(np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10,int(np.diff((np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10))[0]*1+1))
levels = np.linspace(60,100,41)

fig,ax = plt.subplots(1,1,  figsize = (2.9,2/3*2.9))
plt.subplots_adjust(left = .175,top = .9,right = .9,bottom = .22)
dist = ax.contourf(R[sorted_ind],S[sorted_ind],OASPL[sorted_ind],levels = levels,cmap = cmap)
# dist2 = ax.contour(MT[sorted_ind],CT[sorted_ind],DL[sorted_ind],colors = 'k',linestyles = '-.')
# plt.clabel(dist2)
dist2 = ax.contour(R[sorted_ind],S[sorted_ind],OASPL[sorted_ind],colors = 'k',linestyles = '-.')
plt.clabel(dist2)
ax.scatter(.04064/(0.4699/2/AR),(.04064/2)/(0.4699/2/AR),marker = '*',s = 100,c = 'white')
cbar = fig.colorbar(dist,pad = .05)
cbar.ax.set_ylabel(r'$RAISPL, \ dB$')
cbar.set_ticks(levels[::10])
ax.set(ylabel =r'$\Delta/\overline{c}$',xlabel =r'$D_{rod}/\overline{c}$' ,ylim = [0.25,2],xlim = [0.25,2.25])
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
plt.savefig(os.path.join(case_dir,f'ROD_S_R_OASPL_{os.path.basename(case_dir)}.pdf'),format = 'pdf',bbox_inches = 'tight',pad_inches=.05)
plt.close()
