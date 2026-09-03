#!/usr/bin/env python3
import os
# import numpy as np
# import sys
# sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'dependencies'))

from dependencies.pyWopwop.wopwop import *  
from dependencies.pyWopwop.wopwop_io import *  

#%%
case_dir =os.getcwd()
# set to true if rod is positioned below the rotor in its wake
below = True

# loading data file name
loading_fname = 'loading0200.dat'

loading = functional_data(os.path.join(case_dir,loading_fname),endianness='big')
loading.zones[0].data[...,0] = 0.0
# loading.zones[0].data[...,1] = 0.0
loading_fname_split = os.path.splitext(loading_fname)
loading.file_dir = os.path.join(case_dir,f'{loading_fname_split[0]}_mod{loading_fname_split[1]}')
loading.write(ascii=False)

# fig,ax = plt.subplots(1,1, figsize = (2.5,2.5/1.25))
# plt.subplots_adjust(left = .25,bottom = .21,top = 0.875)
# ax.plot(np.gradient(loading.zones[0].data[:,::100,-1],axis = 0))
