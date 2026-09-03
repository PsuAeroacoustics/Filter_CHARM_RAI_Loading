#!/usr/bin/env python3

import argparse
import json5
from help_funcs import *
from plot_results import *

#%%

def main():
    parser = argparse.ArgumentParser("rotor_gust_interaction",description='Simulates a gust interacting with a hovering rotor, only the positive half of the gust profile is considered.')
    parser.add_argument(
        "-loading_fname",
        type= str,
        required=False,
		default="loading0200.dat",
		help="Name of rod loading functional data file, defaults to 'loading0200.dat'",
    )
    parser.add_argument(
        "-geometry_fname",
        type= str,
        required=False,
		default="geometry0200.dat",
		help="Name of rod geometry patch file, defaults to 'geometry0200.dat'",
    )
    parser.add_argument(
        "-resonator_fname",
        type= str,
        required=False,
		default="res_params.json5",
		help="Name of json5 file that specifies the impedance patch parameters.'",
    )
    parser.add_argument(
        "-loading_fname_mod",
        type= str,
        required=False,
		default="loading0200_mod.dat",
		help="Name of modified rod loading functional data file that should already be referenced in the namelist file under the appropriate container, defaults to 'loading0200_mod.dat'",
    )

    parser.add_argument(
        "-parallel",
        action='store_true',
        help="Include to run wopwop using mpi.",
        default=False,
        required=False
    )
    parser.add_argument(
        "-optimize",
        action='store_true',
        help="Include to optimize treatment.",
        default=False,
        required=False
    )
    parser.add_argument(
        "-plot",
        action='store_true',
        help="Include to plot results.",
        default=False,
        required=False
    )
    parser.add_argument(
          '-o','--observers',                
        nargs='+',
        help="Indicies of observers to plot.",
		required=False,
        type=int,
    )
    parser.add_argument(
        "-mag",
        action='store_true',
        help="Filter the rod loads only with the magnitude response of the impedance patches.",
        default=False,
        required=False
    )
    parser.add_argument(
        "-phase",
        action='store_true',
        help="Filter the rod loads only with the phase response of the impedance patches.",
        default=False,
        required=False
    )
    args = parser.parse_args()

    saved_params = {'case_dir':os.getcwd()}
    # if os.path.exists(os.path.join(saved_params['case_dir'],'saved_params.h5')):
    #     saved_params.update(read_results_from_h5(saved_params['case_dir']))
    saved_params.update(vars(args).copy())

    with open(os.path.join(saved_params['case_dir'],saved_params['resonator_fname']), "r") as f:
        res_params = json5.load(f)
    
    import_charm_data(saved_params,res_params)
    process_patch_geometry(saved_params,res_params)
    filter_rod_loads(saved_params,res_params)

    if saved_params['observers'] is None:
         saved_params['observers'] = np.arange(len(saved_params['function_values']))
    
    if saved_params['plot']:
        [f(saved_params,res_params) for f in [plot_ptseries,plot_psd_ptseries,plot_geom]]

    
    with open(os.path.join(saved_params['case_dir'],saved_params['resonator_fname']), "w") as f:
        json5.dump(res_params,f,indent=4)
    
    # removes the loading and geometry objects from saved parameters so that they are not saved redundently in saved_params.h5 
    [saved_params.pop(key, None) for key in ['loading','loading_baseline','geometry']]
    write_results_to_h5(saved_params,res_params)

if __name__ == "__main__":
	main()
	print("exiting main.py")