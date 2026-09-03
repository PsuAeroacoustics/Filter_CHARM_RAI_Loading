# charm_post

Scripts for analyzing CDI CHARM rotor and rod-loading output with OpenWOPWOP.
The primary workflow models acoustic resonators over selected geometry patches,
filters the loading in the frequency domain, and compares untreated and treated
acoustic predictions.

The repository is organized as a collection of research scripts rather than an
installable Python package. In addition to the primary treatment workflow, it
contains resonator models, OpenWOPWOP readers and writers, plotting utilities,
validation scripts, and parameter-sweep helpers.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Basic workflow](#basic-workflow)
- [Configuration](#configuration)
- [Command-line reference](#command-line-reference)
- [Files produced](#files-produced)
- [Repository guide](#repository-guide)
- [Troubleshooting](#troubleshooting)

## Requirements

### Python

The scripts use Python 3 and the following packages:

- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `json5`
- `f90nml`

The bundled [pyWopwop environment](dependencies/pyWopwop/environment.yml)
provides `matplotlib`, `numpy`, and `f90nml`; install the remaining packages in
the same environment. `json5` is required by the main optimization/filtering
script, while `scipy` and `h5py` are used throughout the repository.

### OpenWOPWOP

The treatment workflow invokes the command-line executable `wopwop3`. It must be
installed and available on `PATH`. A case must also contain a valid OpenWOPWOP
`cases.nam` file and the files referenced by that case. OpenWOPWOP is not bundled
with this repository.

For parallel execution, `mpirun` must also be available and able to launch
`wopwop3`.

### Input data

Run the main script from an OpenWOPWOP case directory containing at least:

- A CHARM/OpenWOPWOP functional loading file, such as `loading0200.dat`.
- A CHARM/OpenWOPWOP patch geometry file, such as `geometry0200.dat`.
- A `cases.nam` OpenWOPWOP namelist that references the loading file to analyze.
- A JSON5 resonator configuration file.

The bundled `pyWopwop` readers read these binary files as big-endian data. Keep
the case files together and run the workflow from the case directory so relative
paths in `cases.nam` resolve correctly.

## Installation

From the repository root, create or activate a Python environment and install the
Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib h5py json5 f90nml
```

Verify the Python imports and the external solver independently:

```bash
python -c "import numpy, scipy, matplotlib, h5py, json5, f90nml; print('Python dependencies OK')"
wopwop3 --help
```

The scripts are not installed as a package. The main script can be called with an
absolute path from a case directory, as shown below. Scripts that import modules
from the repository root should be run from the repository root or with the root
directory on `PYTHONPATH`.

## Basic workflow

1. Prepare an OpenWOPWOP case directory with `cases.nam`, the baseline loading,
   and the matching patch geometry file.
2. Copy [sdof_geom_params_example.json5](sdof_geom_params_example.json5) to a
   case-specific filename and edit its treatment parameters.
3. Run the main script from that case directory:

   ```bash
   python /path/to/charm_post/fliter_optimize_rod_loads.py \
       -loading_fname loading0200.dat \
       -geometry_fname geometry0200.dat \
       -resonator_fname sdof_geom_params.json5 \
       -loading_fname_mod loading0200_mod.dat \
       -plot \
       -o 0 4 8
   ```

4. Inspect the generated OpenWOPWOP results and `saved_params.h5`. The HDF5 file
   contains the run arguments, derived geometry and filter data, resonator
   parameters, and both acoustic responses.

The filename `fliter_optimize_rod_loads.py` is intentionally preserved for
compatibility with the existing scripts; `fliter` is a historical spelling of
`filter` in this repository.

### What the pipeline does

The main script performs the following operations:

1. Loads the functional loading and patch geometry files.
2. Detects the span and azimuth coordinates of the geometry and selects the
	 region described by `r_extents` and `phi_extents`.
3. Divides the selected region into `N_r * N_phi` patches and assigns each patch
	 its resonator length set.
4. Removes the loading's first data component, writes the modified loading file,
   and runs OpenWOPWOP to establish the baseline response.
5. Computes each resonator's complex impedance, combines the resonators using the
   open-area ratio, converts impedance to a reflection response, and applies that
   response to the selected loading histories with an FFT/IFFT.
6. Writes the treated loading, runs OpenWOPWOP again, and imports the treated
   response.
7. Optionally optimizes resonator radius, lengths, and staggered distributions
	 with SciPy differential evolution.
8. Writes `saved_params.h5` and, when requested, treatment geometry plots.

Overall sound pressure levels use a 20 microPa reference. During optimization,
the objective reduces treated acoustic level relative to baseline while applying
penalties for invalid volume fractions and radius-to-length ratios.

## Configuration

Start with [sdof_geom_params_example.json5](sdof_geom_params_example.json5):

| Key | Meaning |
| --- | --- |
| `phi_extents` | Azimuth limits in degrees. The treatment is applied between these values. |
| `r_extents` | Normalized span limits, generally between `0` and `1`. |
| `N_res` | Total resonator count. In the current main workflow, `null` selects all points in the selected region. |
| `N_r` | Number of radial patch divisions. |
| `N_phi` | Number of circumferential patch divisions. |
| `a_bounds` | Minimum and maximum resonator radius in metres for optimization. |
| `l_bounds` | Minimum and maximum resonator length in metres for optimization. |
| `OAR` | Open-area ratio used to combine resonator impedances. |
| `staggered` | Enables the staggered patch treatment and its distribution parameters. |
| `a` | Resonator radius in metres for a non-optimized run or initial optimization value. |
| `l` | Nested resonator-length lists in metres, one list per unique impedance patch. |
| `dist` | Five-element distribution parameters per circumferential section when `staggered` is `true`. |
| `x0` | Example notation for optimization variables. The main script initializes from `a`, `l`, and `dist` instead. |

### Patch consistency

The selected geometry must contain regularly ordered points that can be divided
into the requested `N_r` by `N_phi` grid. `N_r * N_phi` is the total number of
geometric patches. The number of nested lists in `l` is the number of unique
impedance types. Before running, verify that:

- `N_r` and `N_phi` match the intended geometry discretization.
- Every length in `l` is positive and uses metres.
- `l` contains the intended number of unique patch types.
- `phi_extents` and `r_extents` select at least one geometry point.
- `dist` contains one five-value list for each circumferential section when
	staggering is enabled.

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
| `-loading_fname_mod` | `loading0200_mod.dat` | Output loading file written before the OpenWOPWOP run. Ensure `cases.nam` references this file where required. |
| `-parallel` | off | Runs OpenWOPWOP through `mpirun`. |
| `-optimize` | off | Optimizes treatment parameters within the configured bounds. |
| `-plot` | off | Generates the treatment geometry plots. |
| `-o`, `--observers` | all | Observer indices to use for plotting; accepts one or more integers. |
| `-mag` | off | Uses only the magnitude of the impedance-derived filter response. |
| `-phase` | off | Uses only the phase-normalized impedance-derived filter response. |

`-mag` and `-phase` select alternate filter-response modes. With neither option,
the full complex response is applied. If both are supplied, `-mag` takes
precedence because it is evaluated first in the script.

### Optimization example

```bash
python fliter_optimize_rod_loads.py \
		-optimize \
		-resonator_fname sdof_geom_params.json5 \
		-loading_fname_mod loading0200_optimized.dat
```

The root-level [opt.sh](opt.sh) contains a collection of optimization commands
for several radial/circumferential and staggered configurations. Update filenames
and execute it with `bash opt.sh` only after confirming that each referenced case
file exists and that `cases.nam` points to the selected modified loading file.

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

- [filter_rod_loads.py](filter_rod_loads.py): older hard-coded filtering workflow
	for a single case and SDOF geometry configuration.
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

## Troubleshooting

### `ModuleNotFoundError`

Activate the environment used to install the dependencies and run the command
from the repository root. If importing a script from another directory, add this
repository to `PYTHONPATH`.

### `FileNotFoundError` for loading, geometry, or JSON5 files

The main script resolves these names relative to the current working directory.
Change into the case directory first, or pass paths that are valid from the
directory where the command is launched.

### `wopwop3: command not found`

Install OpenWOPWOP and add its executable directory to `PATH`. For `-parallel`,
also verify `mpirun` and the MPI installation.

### OpenWOPWOP uses the wrong loading file

The script writes the file named by `-loading_fname_mod`, but OpenWOPWOP follows
the references in `cases.nam`. Update the relevant namelist entry before running.

### No points are selected or patch processing fails

Check the azimuth and normalized span limits, the geometry ordering, and the
`N_r`/`N_phi` values. The selected geometry must contain regularly arranged radial
and circumferential points compatible with the requested patch grid.

## Reproducibility notes

- Record the exact input loading, geometry, `cases.nam`, and JSON5 configuration
	files for each study.
- Preserve the generated `saved_params.h5` and modified loading file together.
- Record the OpenWOPWOP executable version and Python environment packages.
- Run one configuration per case directory when possible, since generated files
	such as `saved_params.h5` and the modified loading file are overwritten.
