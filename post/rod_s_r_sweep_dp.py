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
cases_name = ["ROD_SEP_KDE_18_6_DN05","ROD_SEP_KDE_18_6_DN05_NPHI_NR5"]

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

for case in cases_name: 
    cases = [i for i in os.listdir(os.path.join(case_dir,case)) if os.path.isdir(os.path.join(case_dir,case,i))]
    S = np.zeros(len(cases))
    R = np.zeros(len(cases))
    OASPL= np.zeros((len(cases)))

    for case_itr,sub_case in enumerate(cases):
        case_name_split = sub_case.split('_')
        R[case_itr]= r[int(case_name_split[1][-1])-1]
        S[case_itr]= delta[int(case_name_split[2][-1])-1]

        if not os.path.exists(os.path.join(case_dir,case,sub_case,f'{sub_case}.1PSU-WOPWOP',f'{sub_case}.h5')):
            process_wopwop(cases_directory=os.path.join(case_dir,case,sub_case,f'{sub_case}.1PSU-WOPWOP'),cases ='cases.nam')

        acs_data = import_results_from_wopwop(cases_directory=os.path.join(case_dir,case,sub_case,f'{sub_case}.1PSU-WOPWOP'))
        OASPL[case_itr] = np.mean(np.round(10*np.log10(np.mean(acs_data['function_values'][...,-1]**2,axis = -1)/20e-6**2),2))

    R_sort_ind = R.argsort().reshape((len(r),len(delta))).astype(int)
    sorted_ind = np.take_along_axis(R_sort_ind,S[R_sort_ind].argsort(axis = -1),axis = -1)
    data.update({case:{'R':R,'S':S,'OASPL':OASPL,'sorted_ind':sorted_ind}})

cmap = plt.cm.get_cmap('inferno')

dOASPL = (data[cases_name[1]]['OASPL'][data[cases_name[1]]['sorted_ind']])-(data[cases_name[0]]['OASPL'][data[cases_name[0]]['sorted_ind']])

levels = np.linspace(-6,0,13)

fig,ax = plt.subplots(1,1,  figsize = (2.9,2/3*2.9))
plt.subplots_adjust(left = .175,top = .9,right = .9,bottom = .22)
dist = ax.contourf(R[sorted_ind],S[sorted_ind],dOASPL,levels = levels,cmap = cmap)
dist2 = ax.contour(R[sorted_ind],S[sorted_ind],dOASPL,colors = 'k',linestyles = '-.')
ax.scatter(.04064/(0.4699/2/AR),(.04064/2)/(0.4699/2/AR),marker = '*',s = 100,c = 'white')
plt.clabel(dist2)
ax.scatter(R[sorted_ind],S[sorted_ind],alpha = 0.5,color = 'white',s = 2)
cbar = fig.colorbar(dist,pad = .05)
cbar.ax.set_ylabel(r'$\Delta \ RAISPL, \ dB$')
# cbar.ax.set_ylabel(r'$\Delta \ OASPL, \ dB$')
cbar.set_ticks(levels[::2])
ax.set(ylabel =r'$\Delta/\overline{c}$',xlabel =r'$D_{rod}/\overline{c}$' ,ylim = [0.25,2],xlim = [0.25,2.25])
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax.scatter(588.15*(0.4699/2)/343,.009095,marker = '*',s = 100,c = 'white')
plt.savefig(os.path.join(case_dir,f'ROD_S_R_dOASPL_{cases_name[0]}.pdf'),format = 'pdf',bbox_inches = 'tight',pad_inches=.05)
plt.close()
