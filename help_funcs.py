#!/usr/bin/env python3
import numpy as np
import os
import h5py
from dependencies.pyWopwop.wopwop import *  
from dependencies.pyWopwop.wopwop_io import *  
from res_funcs import *
import subprocess
from scipy.optimize import differential_evolution
import copy

np.set_printoptions(precision=4, suppress=False)

def read_results_from_h5(case_dir):

    def h5_to_dict(h5_obj):
        """
        Recursively converts an HDF5 file/group into a nested dictionary.

        Args:
            h5_obj (h5py.File or h5py.Group): HDF5 file or group object.

        Returns:
            dict: Nested dictionary representation of the HDF5 structure.
        """
        h5_dict = {}
        for key,value in h5_obj.items():
            if isinstance(value, h5py.Group):
                # Recursively process groups
                h5_dict.update({key:h5_to_dict(value)})
            else:
                if isinstance(value[()], bytes):
                    h5_dict.update({key:value[()].decode()})
                else:
                    h5_dict.update({key:value[()]})
        return h5_dict

    with h5py.File(os.path.join(case_dir, 'saved_params.h5'), 'r') as f:
        saved_params = h5_to_dict(f)
    return saved_params

def write_results_to_h5(saved_params,res_params):
    """
    Exports a nested dictionary to an HDF5 file.

    Args:
        data (dict): The nested dictionary to export.
        file_path (str): Path to the HDF5 file to save.
    """
    def dict_to_h5(h5_group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = h5_group.create_group(key)
                dict_to_h5(subgroup, value)
            else:
                h5_group.create_dataset(key, data=value)
    saved_params.update({'res_params':res_params})
    with h5py.File(os.path.join(saved_params['case_dir'], 'saved_params.h5'), 'w') as f:
        dict_to_h5(f, saved_params)

def run_wopwop(cases = 'cases.nam',parallel = False):
        
    print(f'Running wopwop...')
    if parallel:
            assert subprocess.run(['mpirun','wopwop3',cases],check = True), 'WOPWOP encountered an error'
    else:
        assert subprocess.run(['wopwop3',cases],check = True), 'WOPWOP encountered an error'

def import_charm_data(saved_params,res_params):
    loading = functional_data(os.path.join(saved_params['case_dir'],saved_params['loading_fname']),endianness='big')
    geometry = patch_data(os.path.join(saved_params['case_dir'],saved_params['geometry_fname']),endianness='big')
    loading_baseline = copy.deepcopy(loading)
    saved_params.update({'loading':loading,'loading_baseline':loading_baseline,'geometry':geometry})

def process_patch_geometry(saved_params,res_params):
    
    rng = saved_params['geometry'].zones[0].nodes[0].max(axis =0)-saved_params['geometry'].zones[0].nodes[0].min(axis =0)
    span_axis = np.argmax(rng)
    r = np.linalg.norm(saved_params['geometry'].zones[0].nodes[0][...,1:],axis = -1)
    phi = np.arctan2(saved_params['geometry'].zones[0].nodes[0][...,2],saved_params['geometry'].zones[0].nodes[0][...,1])%(2*np.pi)
    
    V = np.mean(r)*rng[span_axis]
    # select the points that fall within the specified radial extents 
    select_r_ind = (((saved_params['geometry'].zones[0].nodes[0][:,span_axis]-saved_params['geometry'].zones[0].nodes[0][:,span_axis].min())/(saved_params['geometry'].zones[0].nodes[0][:,span_axis].max()-saved_params['geometry'].zones[0].nodes[0][:,span_axis].min())) \
                    >= res_params['r_extents'][0]) & (((saved_params['geometry'].zones[0].nodes[0][:,span_axis]-saved_params['geometry'].zones[0].nodes[0][:,span_axis].min())/(saved_params['geometry'].zones[0].nodes[0][:,span_axis].max()-saved_params['geometry'].zones[0].nodes[0][:,span_axis].min())) <=res_params['r_extents'][1])

    select_phi_ind = (phi>=res_params['phi_extents'][0]*np.pi/180) & (phi<=res_params['phi_extents'][1]*np.pi/180)
    selected_ind = np.where((select_r_ind & select_phi_ind))[0]

    # if res_params['N_res'] is not None:
    #     rng = np.random.default_rng()
    #     del_ind = rng.choice(len(select_ind),len(select_ind)-res_params['N_res'],replace=False, shuffle=False)
    #     select_ind = np.delete(select_ind,del_ind)

    # total treated area
    A_s = r[(select_r_ind & select_phi_ind)].mean()*(phi[selected_ind].max()-phi[selected_ind].min())*np.abs(saved_params['geometry'].zones[0].nodes[0][selected_ind,span_axis].max()-saved_params['geometry'].zones[0].nodes[0][(select_r_ind & select_phi_ind),span_axis].min())

    # sorts the nodes so that they correspond to increasing radial and circumfrential order
    sorted_ind = np.lexsort((saved_params['geometry'].zones[0].nodes[0][selected_ind][:,1],saved_params['geometry'].zones[0].nodes[0][selected_ind][:,0]))
    # number of unique radial and circumfrential points 
    N_pnts_r,N_pnts_phi = len(np.unique(saved_params['geometry'].zones[0].nodes[0][selected_ind][:,0])),len(np.unique(saved_params['geometry'].zones[0].nodes[0][selected_ind][:,1]))

    # number of unique patches
    N_patches = len(res_params['l'])
    # total number of patches
    N_patches_tot  = int(res_params['N_r']*res_params['N_phi'])
    # number of radial and circumfrential points per patch 
    N_r_patch = int(N_pnts_r/res_params['N_r'])
    N_phi_patch = int(N_pnts_phi/res_params['N_phi'])

    ind = np.zeros((N_patches_tot,N_r_patch,N_phi_patch))
    for i in range(res_params['N_phi']):
        for ii in range(int(N_pnts_r/res_params['N_r'])*res_params['N_r']):
            ind[int(ii/N_r_patch)+i*res_params['N_r'],ii%N_r_patch] = (np.arange(ii*N_pnts_phi+i*N_phi_patch,ii*N_pnts_phi+(i+1)*N_phi_patch))
    
    ind = ind.astype(int)
    # patch_geom = saved_params['geometry'].zones[0].nodes[0][(select_r_ind & select_phi_ind)][sorted_ind][ind]

    if res_params['indexing'] =='r':
        patch_types = np.asarray([np.roll(np.tile(np.arange(N_patches),int(np.ceil(res_params['N_r']/N_patches)))[:res_params['N_r']],-int(N_patches/2)*i) for i in range(res_params['N_phi'])]).flatten()
    else:
        patch_types = np.arange(N_patches_tot)

    dphi = np.abs(np.diff(phi[selected_ind][sorted_ind][ind],axis = -1)).mean(axis = 1).sum(axis = -1)
    R = r[selected_ind][sorted_ind][ind].mean(axis = -1).mean(axis = -1)
    dr = np.abs(np.diff(saved_params['geometry'].zones[0].nodes[0][selected_ind][sorted_ind][ind][...,span_axis],axis = 1)).sum(axis = 1).mean(axis = -1)
    A_patch = R*(dphi+dphi/N_phi_patch)*(dr+dr/N_r_patch)

    point_filt_ind = selected_ind[sorted_ind][ind.reshape(N_patches_tot,N_r_patch*N_phi_patch)]
    patch_filt_ind = np.ones(N_patches_tot,dtype=bool)



    saved_params.update({'phi':phi,'r':r,'V':V,'patch_types':patch_types,'point_filt_ind':point_filt_ind,'A_patch':A_patch,'A_s':A_s,'patch_filt_ind':patch_filt_ind,'N_r_patch':N_r_patch,'N_phi_patch':N_phi_patch,'N_patches_tot':N_patches_tot,'N_patches':N_patches,'N_pnts_r':N_pnts_r,'N_pnts_phi':N_pnts_phi})

def flatten_list(x):
    def flatten(lst):
        for x in lst:
            if isinstance(x, list):
                yield from flatten(x)
            else:
                yield x
    return np.asarray(list(flatten(x)))

def unpack_x0(x0,res_params):
    cntr = 1
    a = x0[0]
    l = []
    dist = []
    for i in range(len(res_params['l'])):
        l.append(list(x0[cntr+np.arange(len(res_params['l'][i]))]))
        cntr+=len(res_params['l'][i])
    if res_params['staggered']: 
        for i in range(res_params['N_phi']):
            dist.append(list(x0[cntr+np.arange(5)]))
            cntr+=5
    res_params.update({'a':a,'l':l,'dist':dist})


def optimize_treatment(x0,saved_params,res_params):
    
    unpack_x0(x0,res_params)
    apply_treatment(saved_params,res_params)
    oaspl_treated = np.round(10*np.log10(np.mean(saved_params['function_values'].squeeze()[:,:,-1]**2,axis = 1)/20e-6**2),1)

    V_res = 0
    for i in range(saved_params['N_patches']):
        N_res = (res_params['OAR']/len(res_params['l'][i])*saved_params['A_patch']/(np.pi*res_params['a']**2)).astype(int)
        if saved_params['N_patches'] ==1:
            V_res += (N_res[:,None]*np.pi*res_params['a']**2*res_params['l'][i]).sum(axis = -1)[(saved_params['patch_types']==i) & saved_params['patch_filt_ind']].sum()
            # V_res += (N_res*np.pi*res_params['a']**2*res_params['l'][i])[(saved_params['patch_types']==i) & saved_params['patch_filt_ind']].sum()

    mu = 100
    L = flatten_list(res_params['l'])
    constraint = mu/2*np.max((0,-(1-V_res/saved_params['V'])))**2+mu/2*np.sum(np.max((np.zeros(len(L)),-(0.25-res_params['a']/L)),axis = 0))**2
    residual = np.mean(10**(oaspl_treated/10)/(10**(saved_params['oaspl_baseline']/10)))+constraint

    print(f'Inputs: {x0}')
    print(f"Volume fraction: {V_res/saved_params['V']:.3e}")
    print(f'Delta OASPL: {np.round(np.mean(saved_params['oaspl_baseline']-oaspl_treated),2)}')
    print(f'Residual: {residual:.3e}')

    return residual

def apply_treatment(saved_params,res_params):    
    
    if res_params['staggered']:
        saved_params['patch_filt_ind'] = get_res_dist(res_params)

    get_Z_smeared(saved_params,res_params)

    # fig,ax = plt.subplots(2,1, figsize = (3,3))
    # plt.subplots_adjust(bottom = 0.15,left = 0.175,top =0.95,right = 0.95)
    # ax[0].plot(saved_params['f'],np.real(saved_params['Z_smeared']))
    # ax[1].plot(saved_params['f'],np.imag(saved_params['Z_smeared']))
    # ax[0].set_xticklabels([])
    # ax[0].set_ylabel(r'$\mathrm{Resistance}, \ \overline{\theta}$')
    # ax[0].set_xlim([0,5e3])
    # ax[0].set_ylim([0,10])
    # ax[0].grid()
    # ax[-1].set_ylabel(r'$\mathrm{Reactance}, \ \overline{\chi}$')
    # ax[-1].set_xlabel(r'Frequency [Hz]')
    # ax[-1].grid()
    # ax[-1].set_xlim([0,5e3])
    # ax[-1].set_ylim([-5, 5])
    # R = (saved_params['Z_smeared']-1)/(saved_params['Z_smeared']+1)
    # fig,ax = plt.subplots(2,1, figsize = (3,3))
    # plt.subplots_adjust(bottom = 0.15,left = 0.175,top =0.95,right = 0.95)
    # ax[0].plot(saved_params['f'],np.abs(R))
    # ax[1].plot(saved_params['f'],np.angle(R))
    # ax[0].set_xticklabels([])
    # ax[0].set_ylabel(r'$\mathrm{Resistance}, \ \overline{\theta}$')
    # ax[0].set_xlim([0,5e3])
    # # ax[0].set_ylim([0,10])
    # ax[0].grid()
    # ax[-1].set_ylabel(r'$\mathrm{Reactance}, \ \overline{\chi}$')
    # ax[-1].set_xlabel(r'Frequency [Hz]')
    # ax[-1].grid()
    # ax[-1].set_xlim([0,5e3])
    # # ax[-1].set_ylim([-5, 5])

    filt_resp = get_filt_resp(saved_params['Z_smeared'])
    # filt_resp = get_filt_resp(1j*np.imag(saved_params['Z_smeared']))
    if saved_params['mag']:
        filt_resp = np.abs(filt_resp)
    elif saved_params['phase']:
        filt_resp = filt_resp/np.abs(filt_resp)

    for i in range(saved_params['N_patches']):
        filt_ind = (saved_params['point_filt_ind'][(saved_params['patch_types']==i) & saved_params['patch_filt_ind']]).flatten()
        # filt_data = apply_filt(saved_params['loading'].zones[0].data[:,filt_ind,1:],filt_resp[:,i])
        saved_params['loading'].zones[0].data[:,filt_ind,1:] = apply_filt(saved_params['loading_baseline'].zones[0].data[:,filt_ind,1:],filt_resp[:,i])
    saved_params['loading'].write(ascii=False)
    run_wopwop(parallel=saved_params['parallel'])
    process_wopwop(cases_directory=saved_params['case_dir'],cases = 'cases.nam')
    saved_params.update(import_results_from_wopwop(cases_directory=saved_params['case_dir']))


def filter_rod_loads(saved_params,res_params):
    
    dt = saved_params['loading'].zones[0].period/saved_params['loading'].zones[0].N_time_steps
    f = np.arange(1,int(saved_params['loading'].zones[0].N_time_steps/2)+1)*1/saved_params['loading'].zones[0].period
    
    saved_params['loading'].zones[0].data[...,0] = 0.0
    saved_params['loading'].file_dir = os.path.join(saved_params['case_dir'],saved_params['loading_fname_mod'])
    saved_params['loading'].write(ascii=False)

    run_wopwop(parallel=saved_params['parallel'])
    process_wopwop(cases_directory=saved_params['case_dir'],cases = 'cases.nam')
    acs_data = import_results_from_wopwop(cases_directory=saved_params['case_dir'])
    oaspl_baseline = np.round(10*np.log10(np.mean(acs_data['function_values'].squeeze()[...,-1]**2,axis = 1)/20e-6**2),1)
    saved_params.update({'f':f,'dt':dt,'oaspl_baseline':oaspl_baseline,'baseline_function_values':acs_data['function_values']})

    if saved_params['optimize']:

        x0 = flatten_list([res_params['a'],res_params['l'],res_params['dist']])

        lb,ub = [[res_params['a_bounds'][i]]+[res_params['l_bounds'][i]]*len(flatten_list([res_params['l']]))+[i]*len(flatten_list([res_params['dist']])) for i in [0,1]]
        bounds = Bounds(lb = lb, ub = ub,keep_feasible=True)

        sol = differential_evolution(optimize_treatment,x0 = x0,bounds = bounds, polish=False,maxiter = int(res_params['maxiter']/(15*len(x0))),args = (saved_params,res_params))        
        unpack_x0(sol.x,res_params)
        print(f"Done! Minimizer: a={res_params['a']}, l={res_params['l']}")

        run_wopwop(parallel=saved_params['parallel'])
        process_wopwop(cases_directory=saved_params['case_dir'],cases = 'cases.nam')
        saved_params.update(import_results_from_wopwop(cases_directory=saved_params['case_dir']))

    else:
        apply_treatment(saved_params,res_params)
    
    saved_params['function_values'] = np.flip(saved_params['function_values'],axis = 0)
    saved_params['baseline_function_values'] = np.flip(saved_params['baseline_function_values'],axis = 0)




