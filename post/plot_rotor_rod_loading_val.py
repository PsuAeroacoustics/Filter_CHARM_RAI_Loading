#!/usr/bin/env python3

import os
import numpy as np
import plot_styles
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline,make_interp_spline,UnivariateSpline,make_splrep

data_dir = '/Users/danielweitsman/Downloads/'
data_f_name = ['rotor_dfz.csv','rotor_dfx.csv']

data = {}
[data.update({os.path.splitext(case)[0]:np.loadtxt(os.path.join(data_dir,case),delimiter =',').astype(float)}) for case in data_f_name]

dt = (6420/60)**-1/360
t = np.arange(361)*dt
keys = list(data.keys())


spl = [make_splrep(v[:,0]*dt, v[:,1], k=3,s=1e-6) for k,v in data.items()]
dspl = [i.derivative()(t) for i in spl]



fig,ax = plt.subplots(2,1, figsize = (3.5,3.5/1.25))
plt.subplots_adjust(left = .175,bottom = 0.2,right = 0.95,top = 0.92,hspace = 0.3)
for i in range(2):
    ax[i].plot(t/t[-1],spl[i](t),linestyle = '-')
ax[0].set(xlim = [0,1],ylim = [-7e-3,7e-3],ylabel = r'$F_z \ [N]$',xticklabels = [])
ax[1].set(xlim = [0,1],ylim = [-7e-3,7e-3],xlabel = r'Rev. Fraction',ylabel = r'$F_x \ [N]$')
for i in range(2):
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax[i].grid()
plt.savefig(os.path.join(data_dir,'rotor_f.png'),format = 'png',dpi = 600)

fig,ax = plt.subplots(2,1, figsize = (3.5,3.5/1.25))
plt.subplots_adjust(left = .2,bottom = 0.2,right = 0.95,top = 0.92,hspace = 0.3)
for i in range(2):
    ax[i].plot(t/t[-1],dspl[i],linestyle = '-')
ax[0].set(xlim = [0,1],ylim = [-15,15],ylabel = r'$\partial F_z/\partial t \ \ [Ns^{-1}]$',xticklabels = [])
ax[1].set(xlim = [0,1],ylim = [-15,15],xlabel = r'Rev. Fraction',ylabel = r'$\partial F_x/\partial t \ \ [Ns^{-1}]$')
for i in range(2):
    # ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax[i].grid()
plt.savefig(os.path.join(data_dir,'rotor_df.png'),format = 'png',dpi = 600)


