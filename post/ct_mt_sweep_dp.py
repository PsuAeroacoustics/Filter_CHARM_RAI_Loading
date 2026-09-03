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

case_dir = os.getcwd()
cases_name = ["CT_MT_KDE_150_50_DN05_BP","CT_MT_KDE_150_50_DN05_BP_ROD_ONLY"]
rho = 1.225
sos = 343
N_obs = 10
data = {}

for case in cases_name: 
    cases = [i for i in os.listdir(os.path.join(case_dir,case)) if os.path.isdir(os.path.join(case_dir,case,i))]
    CT = np.zeros(len(cases))
    MT = np.zeros(len(cases))
    DL = np.zeros(len(cases))
    OASPL= np.zeros((len(cases),N_obs))
    p_ms = np.zeros((len(cases),N_obs))
    for case_itr,sub_case in enumerate(cases):
        perf_data = np.genfromtxt(os.path.join(case_dir,case,sub_case,f'{sub_case}.1prop.dat'),names=True)
        CT[case_itr] = perf_data['THRUST']/(rho*np.pi*perf_data['RADIUS']**2*(perf_data['OMEGA']*perf_data['RADIUS'])**2)
        MT[case_itr] = perf_data['OMEGA']*perf_data['RADIUS']/sos
        DL[case_itr] = perf_data['THRUST']/(np.pi*perf_data['RADIUS']**2)

        if not os.path.exists(os.path.join(case_dir,case,sub_case,f'{sub_case}.1PSU-WOPWOP',f'{sub_case}.h5')):
            process_wopwop(cases_directory=os.path.join(case_dir,case,sub_case,f'{sub_case}.1PSU-WOPWOP'),cases ='cases.nam')

        acs_data = import_results_from_wopwop(cases_directory=os.path.join(case_dir,case,sub_case,f'{sub_case}.1PSU-WOPWOP'))
        OASPL[case_itr] = np.round(10*np.log10(np.mean(acs_data['function_values'][...,-1].squeeze()**2,axis = -1)/20e-6**2),1)
        p_ms[case_itr] = np.mean(acs_data['function_values'][...,-1].squeeze()**2,axis = -1)
    MT_sort_ind = MT.argsort().reshape((np.sqrt(len(cases))*np.ones(2)).astype(int),order='F')
    sorted_ind = np.take_along_axis(MT_sort_ind,CT[MT_sort_ind].argsort(axis = 0),axis = 0)


    data.update({case:{'CT':CT,'MT':MT,'DL':DL,'OASPL':OASPL,'p_ms':p_ms,'sorted_ind':sorted_ind}})


cmap = plt.cm.get_cmap('inferno')
# cmap = cm.inferno(np.linspace(0, 1, 8))

dOASPL = np.mean((data[cases_name[1]]['OASPL'][data[cases_name[1]]['sorted_ind']])-(data[cases_name[0]]['OASPL'][data[cases_name[0]]['sorted_ind']]),axis = -1)

levels = np.linspace(np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10,int(np.diff((np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10))[0]*1+1))
levels_c = np.linspace(np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10,int(np.diff((np.round(OASPL.min()/10)*10,np.round(OASPL.max()/10)*10))[0]*1+1))
levels = np.linspace(-6,0,13)

fig,ax = plt.subplots(1,1,  figsize = (2.9,2/3*2.9))
plt.subplots_adjust(left = .175,top = .9,right = .9,bottom = .22)
dist = ax.contourf(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],dOASPL,levels = levels,cmap = cmap)
dist2 = ax.contour(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],dOASPL,levels = levels[::2],colors = 'k',linestyles = '-.')
plt.clabel(dist2)
ax.scatter(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],alpha = 0.5,color = 'white',s = 2)
cbar = fig.colorbar(dist,pad = .05)
cbar.ax.set_ylabel(r'$\Delta \ RAI \ SPL, \ dB$')
# cbar.ax.set_ylabel(r'$\Delta \ OASPL, \ dB$')
cbar.set_ticks(levels[::2])
ax.set(ylabel =r'$C_T$',xlabel =r'$M_T$',xlim = [0.3,0.8],ylim = [0.004,0.015])
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax.scatter(588.15*(0.4699/2)/343,.009095,marker = '*',s = 100,c = 'white')
plt.savefig(os.path.join(case_dir,f'CT_MT_dOASPL_{cases_name[0]}.pdf'),format = 'pdf',bbox_inches = 'tight',pad_inches=.05)
plt.close()

dp_ms = (data[cases_name[1]]['p_ms'][data[cases_name[1]]['sorted_ind']]).mean(axis = -1)/(data[cases_name[0]]['p_ms'][data[cases_name[0]]['sorted_ind']]).mean(axis = -1)
# dp_ms = (data[cases_name[1]]['p_ms'][data[cases_name[1]]['sorted_ind']]/data[cases_name[0]]['p_ms'][data[cases_name[0]]['sorted_ind']]).mean(axis = -1)

levels = np.linspace(0,1.2,25)

fig,ax = plt.subplots(1,1,  figsize = (2.9,2/3*2.9))
plt.subplots_adjust(left = .175,top = .9,right = .9,bottom = .22)
dist = ax.contourf(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],dp_ms,levels = levels,cmap = cmap)
dist2 = ax.contour(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],dp_ms,levels = levels[::4],colors = 'k',linestyles = '-.')
plt.clabel(dist2)
ax.scatter(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],alpha = 0.5,color = 'white',s = 2)
cbar = fig.colorbar(dist,pad = .05)
cbar.ax.set_ylabel(r'$\overline{p^2}_{rod} \, / \, \overline{p^2}_{total}$')
# cbar.ax.set_ylabel(r'$\Delta \ OASPL, \ dB$')
cbar.set_ticks(levels[::4])
ax.set(ylabel =r'$C_T$',xlabel =r'$M_T$',xlim = [0.3,0.8],ylim = [0.004,0.015])
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax.scatter(588.15*(0.4699/2)/343,.009095,marker = '*',s = 100,c = 'white')
plt.savefig(os.path.join(case_dir,f'CT_MT_dp_ms_{cases_name[0]}.pdf'),format = 'pdf',bbox_inches = 'tight',pad_inches=.05)
plt.close()


# levels = np.linspace(80,110,31)

# fig,ax = plt.subplots(1,1,  figsize = (4*.95,4*2/3*.95))
# plt.subplots_adjust(left = .15,top = .925,right = .9,bottom = .175)
# dist = ax.contourf(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],data[cases_name[1]]['OASPL'][data[cases_name[1]]['sorted_ind']].mean(axis = -1),levels = levels,cmap = cmap)
# dist2 = ax.contour(data[cases_name[0]]['MT'][data[cases_name[0]]['sorted_ind']],data[cases_name[0]]['CT'][data[cases_name[0]]['sorted_ind']],dOASPL,levels = levels[::2],colors = 'k',linestyles = '-.')
# plt.clabel(dist2)
# cbar = fig.colorbar(dist,pad = .05)
# cbar.ax.set_ylabel(r'$\Delta OASPL, \ dB \ (re: \ 20 \mu Pa)$')
# cbar.set_ticks(levels[::2])
# ax.set(ylabel =r'$C_T$',xlabel =r'$M_T$',xlim = [0.3,0.8],ylim = [0.002,0.015])
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# ax.scatter(588.15*(0.4699/2)/343,.009095,marker = '*',s = 100,c = 'white')

