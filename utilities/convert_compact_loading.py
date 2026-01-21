#!/usr/bin/env python3

import os
import numpy as np
from dependencies.pyWopwop.wopwop import *  
from dependencies.pyWopwop.wopwop_io import *  

case = 'quickROD_SDOF_GEOM_OAR15_POINT/quickROD.1PSU-WOPWOP'

# loading data file name
loading_fname = 'loading0200_sdof_geom_oar15.dat'
# patch data file name
geometry_fname = 'geometry0200.dat'

case_dir = os.path.join(os.getcwd())
loading_fname_split = os.path.splitext(loading_fname)
geometry_fname_split = os.path.splitext(geometry_fname)

loading = functional_data(os.path.join(case_dir,case,loading_fname),endianness='big')
geometry = patch_data(os.path.join(case_dir,case,geometry_fname),endianness='big')

# determine the index corresponding to each spanwise element 
sort_ind = np.where(np.abs(np.diff(geometry.zones[0].nodes[0][:,0]))>1e-5)[0]+1
lifting_line_nodes = np.asarray([geometry.zones[0].nodes[0][sort_ind[i]:sort_ind[i+1]].mean(axis = 0) for i in range(len(sort_ind)-1)])
lifting_line_norms = np.asarray((np.zeros(len(lifting_line_nodes)),np.zeros(len(lifting_line_nodes)),-np.ones(len(lifting_line_nodes)))).T
dr = np.insert(np.abs(np.diff(lifting_line_nodes,axis = 0))[:,0],len(lifting_line_nodes)-1,np.abs(np.diff(lifting_line_nodes[:2],axis = 0)[0,0]))[:,None]
# compact_loading_data = np.asarray([np.sum(loading.zones[0].data[:,sort_ind[i]:sort_ind[i+1]],axis = 1) for i in range(len(sort_ind)-1)]).transpose(1,0,-1)*np.abs(np.diff(lifting_line_nodes,axis = 0)[1,0])
lifting_line_loads = (np.asarray([np.trapezoid(loading.zones[0].data[:,sort_ind[i]:sort_ind[i+1]]*np.linalg.norm(geometry.zones[0].norms[0][sort_ind[i]:sort_ind[i+1]],axis = -1)[:,None],axis = 1) for i in range(len(sort_ind)-1)]).transpose(1,0,-1)/dr)

compact_loading = functional_data(file_dir = os.path.join(case_dir,case,f'{loading_fname_split[0]}_line{loading_fname_split[1]}'),loads=lifting_line_loads,data_type = 2,dimensions = (len(sort_ind)-1,1),period = loading.zones[0].period, keys = loading.zones[0].keys,endianness = 'big',comments = ' ',structured = 1,node_centered = 1,loading_type = 2,reference_frame = 2)
compact_loading.write(ascii=False)
compact_geometry = patch_data(os.path.join(case_dir,case,f'{geometry_fname_split[0]}_line{geometry_fname_split[1]}'),nodes = lifting_line_nodes,normals = lifting_line_norms,dimensions = (len(sort_ind)-1,1), endianness = 'big',N_zones = 1,structured = 1,data_type = 1,node_centered = 1)
compact_geometry.write(ascii=False)

lifting_point_nodes = lifting_line_nodes.mean(axis =0)
lifting_point_norms = np.asarray([0,0,-1])
lifting_point_loads = np.trapezoid(lifting_line_loads*dr,axis = 1)/np.abs(geometry.zones[0].nodes[0].max(axis =0)-geometry.zones[0].nodes[0].min(axis = 0))[0]

compact_loading = functional_data(file_dir = os.path.join(case_dir,case,f'{loading_fname_split[0]}_point{loading_fname_split[1]}'),loads=lifting_point_loads,data_type = 2,dimensions = (1,1),period = loading.zones[0].period, keys = loading.zones[0].keys,endianness = 'big',comments = ' ',structured = 1,node_centered = 1,loading_type = 2,reference_frame = 2)
compact_loading.write(ascii=False)
compact_geometry = patch_data(os.path.join(case_dir,case,f'{geometry_fname_split[0]}_point{geometry_fname_split[1]}'),nodes = lifting_point_nodes,normals = lifting_point_norms,dimensions = (1,1), endianness = 'big',N_zones = 1,structured = 1,data_type = 1,node_centered = 1)
compact_geometry.write(ascii=False)

