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
case_dir = "/Users/danielweitsman/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/charm/rotor_af_interaction/quickROD_NPSI128/quickROD.1PSU-WOPWOP"
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

