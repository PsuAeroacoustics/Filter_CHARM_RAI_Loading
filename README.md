# CDI CHARM Rotor-Airframe Interaction Noise Utility 

This code was developed to assess the effectiveness of using open-closed resonator cavities embedded in the airframe to reduce rotor-airframe interaction noise. CDI CHARM was used as the aerodynamics solver, but this methodology is general so it can be adapted for other programs. It only requires a complete PSU-WOPWOP case (namelist, functional loading files, and geometry patch). 

This workflow reads the non-compact loading data on the surface of the airframe, which in this case is taken to be a cylindrical rod. Modifies the loads in accordance with the estimated acoustic response (complex reflection coefficient) of the resonator cavities, and writes out the modified loading file. It then runs PSU-WOPWOP and imports and plots the results. The workflow also includes a differential evolution global optimizer (scipy) to determine the dimensions of the resonators cavities that yield the maximum predicted noise reduction. All this functionality is controlled through the CLI. 

## Requirements

This code utilizes two other repositories that need to be setup and installed first. These are provided as git submodules and are largely transparent to the user. If you are cloning a fresh repository run:
```
	git clone --recurse-submodules https://github.com/DanWeitsman/rotor_gust_interaction
```

If the repository is already cloned, but submodules were not cloned, run the following command to acquire the submodules:
```
	git submodule update --init --recursive
```

In addition the following dependencies are required. 

- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `json5`
- `f90nml`
- A separately installed PSU-WOPWOP executable for running acoustic cases

These can be installed as follows
```bash
pip install -r requirements.txt
```

Before running the script, the local path to the location of the repository containing `fliter_optimize_rod_loads.py` must be added to your system PATH environment variable so that the command can be executed from any PSU-WOPWOP case directory. This can be accomplished by adding the following lines to your `.bashrc` or `.zshrc` file:

```bash
export PATH="/path/to/fliter_optimize_rod_loads.py:$PATH"
```

After updating the file, reload your shell configuration:

```bash
source ~/.bashrc
```
or 
```bash
source ~/.zshrc
```

The submodules in the `dependencies` folder also needs to be added as a PATH variable. 

This only needs to be done once. Thereafter, pyPostAcs.py can be executed from any directory containing the desired acoustic datasets.

### Input data

Run the main script from an OpenWOPWOP case directory containing at least:

- A CHARM/OpenWOPWOP functional loading file, such as `loading0200.dat`.
- A CHARM/OpenWOPWOP patch geometry file, such as `geometry0200.dat`.
- A `cases.nam` OpenWOPWOP namelist that references the loading file to analyze.
- A JSON5 resonator configuration file.


### Packaged example case

The repository includes a complete rotor-airframe-interaction example in
[example_case/rotor_airframe_interaction](example_case/rotor_airframe_interaction).
It contains the PSU-WOPWOP namelists, baseline loading and geometry files,
an example resonator configuration in `res_params.json5`,
and saved acoustic result files. The case is also useful for inspecting the
expected input and output file layout before preparing a new case.

The code reads in the non-compact loading data on the surface of the rod. Modifies it in accordance with the estimated acoustic response (complex reflection coefficient) of the resonator cavities, and writes out the modified rod loads before rerunning PSU-WOPWOP and plotting the output. 

To run this workflow, navigate to the example case directory and execute `fliter_optimize_rod_loads.py` with the default CL arguments:

```bash
cd example_case/rotor_airframe_interaction
python ../../fliter_optimize_rod_loads.py -plot
```

## Command-line reference

Run:

```bash
python fliter_optimize_rod_loads.py --help
```

| Option | Default | Description |
| --- | --- | --- |
| `-loading_fname` | `loading0200.dat` | Input functional loading file. |
| `-geometry_fname` | `geometry0200.dat` | Input patch geometry file. |
| `-resonator_fname` | `sdof_geom_params.json5` | JSON5 treatment configuration. |
| `-loading_fname_mod` | `loading0200_mod.dat` | Output loading file written before the PSU-WOPWOP run. Ensure `cases.nam` references this file where required. |
| `-parallel` | off | Runs PSU-WOPWOP through `mpirun`. |
| `-optimize` | off | Optimizes treatment parameters within the configured bounds. |
| `-plot` | off | Generates the treatment geometry plots. |
| `-o`, `--observers` | all | Observer indices to use for plotting; accepts one or more integers. |
| `-mag` | off | Uses only the magnitude of the impedance-derived filter response. |
| `-phase` | off | Uses only the phase-normalized impedance-derived filter response. |

`-mag` and `-phase` select alternate filter-response modes. With neither option,
the full complex response is applied. If both are supplied, `-mag` takes
precedence because it is evaluated first in the script.

### What the pipeline does

The main script performs the following operations:

1. Loads the functional loading and patch geometry files.
2. Detects the span and azimuth coordinates of the geometry and selects the
	 region described by `r_extents` and `phi_extents`.
3. Divides the selected region into `N_r * N_phi` patches and assigns each patch
	 its resonator length set.
4. Runs PSU-WOPWOP to establish the baseline response.
5. Computes each resonator's complex impedance and complex reflection coefficient using the 		Zwikker-Kosten TL model. 
6. Convolves original loads with the complex reflection coefficient.
7. Writes the modified loading file and reruns PSU-WOPWOP.
8. Optionally optimizes resonator radius, lengths, and staggered distributions
	 with SciPy differential evolution.
9. Processes and writes out results to `saved_params.h5` and, when requested, plots output.

## Configuration

| Key | Meaning |
| --- | --- |
| `phi_extents` | Azimuth limits in degrees. The treatment is applied between these values. |
| `r_extents` | Nondimensional spanwise extents of the treatment, between `0` and `1`. |
| `N_res` | Total resonator count. In the current main workflow, `null` selects all points in the selected region. |
| `N_r` | Number of radial patch divisions. |
| `N_phi` | Number of circumferential patch divisions. |
| `a_bounds` | Minimum and maximum resonator radius in meters for optimization. |
| `l_bounds` | Minimum and maximum resonator length in meters for optimization. |
| `OAR` | Open-area ratio used to combine resonator impedances. |
| `staggered` | Enables the staggered patch treatment and its distribution parameters. |
| `a` | Resonator radius in meters for a non-optimized run or initial optimization value. |
| `l` | Nested resonator-length lists in meters, one list per unique impedance patch. |
| `dist` | Five-element distribution parameters per circumferential section when `staggered` is `true`. |
| `x0` | Example notation for optimization variables. The main script initializes from `a`, `l`, and `dist` instead. |

### Patch consistency

The selected geometry must contain regularly ordered points that can be divided
into the requested `N_r` by `N_phi` grid. `N_r * N_phi` is the total number of
geometric patches. The number of nested lists in `l` is the number of unique
impedance types. Before running, verify that:


## Files produced

Depending on the selected options, the case directory can contain:

- The modified loading file named by `-loading_fname_mod`.
- OpenWOPWOP output files and imported acoustic result data.
- `saved_params.h5`, which is overwritten with the most recent run's summary.
- `geom_<configuration>.png` and `geom_<configuration>.pdf` when `-plot` is used.
- `geom_<configuration>_compact.pdf` when `-plot` is used.
- Additional response plots such as `p_tseries_<configuration>.pdf` and
	`psd_p_tseries_<configuration>.pdf` when the corresponding plotting functions
	are called.

The pipeline also writes the final resonator parameters back to the JSON5 file.
Use a copy of the configuration when the original file must remain unchanged.

## Repository guide

### Core processing

- [fliter_optimize_rod_loads.py](fliter_optimize_rod_loads.py): command-line entry
	point for baseline filtering, treatment, optimization, and result persistence.
- [help_funcs.py](help_funcs.py): CHARM/OpenWOPWOP data import, case execution,
	geometry processing, and HDF5 serialization helpers.
- [res_funcs.py](res_funcs.py): resonator impedance, smeared impedance, filter
	response, FFT filtering, and optimization helpers.
- [plot_results.py](plot_results.py): treatment geometry, time-series, and PSD
	plotting functions.

### Standalone utilities

- [modify_rod_loads.py](modify_rod_loads.py): simple loading-file modification
	utility that zeros the first loading component.
- [utilities/convert_compact_loading.py](utilities/convert_compact_loading.py):
	compact-loading conversion utility.
- [utilities/sweep_params.py](utilities/sweep_params.py): parameter-sweep setup
	calculations for rotor and rod cases.

### Post-processing and dependencies

- [post/](post/): case-specific CHARM/OpenWOPWOP validation, spectra, observer,
	loading, and sweep plotting scripts. Most scripts contain case paths and
	observer selections that should be edited before use.
- [dependencies/pyWopwop/](dependencies/pyWopwop/): Python readers and writers for
	OpenWOPWOP data and case utilities.
- [dependencies/resonator/](dependencies/resonator/): resonator and facesheet
	impedance models plus validation examples/data.
- [dependencies/pyPostAcs/](dependencies/pyPostAcs/): acoustic post-processing
	support scripts and examples.

The post-processing scripts do not share a common CLI. Many define case paths and
observer selections near the top of the file, so inspect and update those values
before running them.
