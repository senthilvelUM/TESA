# TESA Tutorial

This tutorial walks you through a complete TESA analysis using the included sample data. By the end, you will have computed effective elastic, thermal expansion, and thermal conductivity properties for a two-phase quartz-plagioclase microstructure, along with microscale stress, strain, heat flux, and temperature gradient fields.

## Prerequisites

1. Python 3.9 or later
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 1. Input Data

TESA requires two types of input: an EBSD map and one property file per crystal phase.

### EBSD File

The included sample file `EBSD_maps/Tutorial.ang` is a 50x50 EBSD map of a two-phase quartz-plagioclase microstructure. Each row contains six columns:

| Column | Content |
|--------|---------|
| 1 | phi1 (Euler angle, radians) |
| 2 | Phi (Euler angle, radians) |
| 3 | phi2 (Euler angle, radians) |
| 4 | X coordinate |
| 5 | Y coordinate |
| 6 | Phase ID (1 or 2) |

The first few lines look like this:

```
#TestSample1 (Phase1-Quartz; Phase2-Plagioclase)
#Three Euler angles, x- and y- coordinates, phase
4.1452123   0.3929915   2.2051931   0   0   2
4.1452123   0.3929915   2.2051931   2   0   2
4.1452123   0.3929915   2.2051931   4   0   2
```

Lines starting with `#` are header comments and are ignored by the parser. The column numbers in run_tesa.py (`euler_col`, `xy_col`, `phase_col`) are 1-indexed and tell TESA where to find each quantity. For this file: `euler_col=1`, `xy_col=4`, `phase_col=6`.

TESA also supports `.ctf` (Oxford Instruments) and `.txt` (generic tab-delimited) formats.

**Important notes about EBSD data:**
- The EBSD data must be on a **square grid** (equal step size in x and y).
- All data points within a grain must have the **same crystallographic orientation** (Euler angles). This is the standard output when exporting grain-averaged orientations from OIM or Channel 5 software.

### Crystal Phase Property Files

Each phase referenced in the EBSD map needs a property file. The tutorial uses two files in `property_files/`:

- **Quartz.txt** — alpha-Quartz single-crystal properties
- **Plagioclase.txt** — Plagioclase (An25) single-crystal properties

Property files use an Abaqus-style `*keyword` format. Here is a shortened example:

```
*phase
Quartz

*density
2650

*stiffness_matrix
  88.2e+9    6.5e+9   12.4e+9  18.8e+9  00.0e+0  00.0e+0
   6.5e+9   88.2e+9   12.4e+9 -18.8e+9  00.0e+0  00.0e+0
  12.4e+9   12.4e+9  107.2e+9  00.0e+0  00.0e+0  00.0e+0
  18.8e+9  -18.8e+9   00.0e+0  58.5e+9  00.0e+0  00.0e+0
  00.0e+0   00.0e+0   00.0e+0  00.0e+0  58.5e+9  18.8e+9
  00.0e+0   00.0e+0   00.0e+0  00.0e+0  18.8e+9  40.9e+9

*thermal_expansion
8.0e-6
8.0e-6
14.0e-6
0.0e+0
0.0e+0
0.0e+0

*thermal_conductivity
6.15 0.0 0.0
0.0 6.15 0.0
0.0 0.0 10.17
```

The available keywords are:

| Keyword | Description | Units |
|---------|-------------|-------|
| `*phase` | Phase/material name | — |
| `*density` | Scalar density | kg/m^3 |
| `*stiffness_matrix` | 6x6 elastic stiffness matrix | Pa |
| `*thermal_expansion` | 6-component thermal expansion array | 1/K |
| `*thermal_conductivity` | 3x3 thermal conductivity matrix | W/(m K) |

Comments (`#`) and blank lines are ignored. Use `# source: [n]` to document data provenance, with full references at the bottom of the file.

If a keyword is omitted from the property file, TESA uses defaults: `*density` = 1.0, `*stiffness_matrix` = identity, `*thermal_expansion` = zeros, `*thermal_conductivity` = identity. If you do not have thermal expansion or thermal conductivity data for a phase, you can omit those keywords and skip the corresponding analysis (`run_heat_conduction = False`).

## 2. Global Settings

The `settings` dictionary at the top of `run_tesa.py` controls options that apply to all jobs. Here are the key groups:

### Output verbosity

```python
"verbose_console": "high",   # "none", "medium", or "high"
"verbose_log": "high",       # "none", "medium", or "high"
```

- `"none"` — banner and errors only
- `"medium"` — key results (effective properties, mesh statistics)
- `"high"` — full details (recommended for the tutorial)

### Figure display

```python
"show_figures": True,           # True = display on screen, False = save to files only
"figure_pause": 1.0,            # Seconds each figure is displayed (if show_figures=True)
"figure_dpi": 150,              # Resolution of saved figures (dots per inch)
"figure_fontsize": 16,          # Font size for all non-title text (colorbar labels, tick labels, axis labels)
"figure_title_fontsize": 14,    # Font size for figure titles
```

Set `show_figures` to `False` if running on a remote server or if you prefer to review saved figures after the run completes.

### Microstructure plot colors

```python
"phase_colors": ['red', 'lime', 'tab:blue', ...],
"grain_colormap": "tab20",
```

These control the colors used for phase maps and grain maps. Any Matplotlib colormap or named color can be used.

### Field plot settings

```python
"field_colormap": "turbo",         # Colormap for stress, strain, heat flux plots
"field_plot_style": "default",     # "default" auto-selects based on mesh type
"show_grain_boundaries": True,     # Overlay grain boundaries on field plots
```

The `"default"` field plot style uses grain-aware RBF interpolation for Type 1 meshes (smooth within grains, discontinuous at boundaries) and smooth interpolation for Types 2/3.

### Reproducibility

```python
"random_seed": 10,   # Fixed integer for reproducible mesh; None for random
```

Setting a fixed seed ensures the mesh is identical across runs.

## 3. Job Configuration

Each job in the `jobs` list defines a complete analysis. The tutorial job in `run_tesa.py` is configured as follows:

### EBSD data

```python
"ebsd_file": "EBSD_maps/Tutorial.ang",
"study_name": "tutorial_analyses",
"euler_col": 1,
"xy_col": 4,
"phase_col": 6,
"ref_frame_angle": 90,
```

- `ebsd_file` — path to the EBSD data file
- `study_name` — results are saved in a subfolder named `study_{study_name}`
- `euler_col`, `xy_col`, `phase_col` — 1-indexed column positions (see the EBSD file table above)
- `ref_frame_angle` — rotation from EBSD to mesh coordinates (degrees). This depends on the EBSD acquisition system and reference frame convention:

  | System | Condition | Angle |
  |--------|-----------|-------|
  | OIM (EDAX) | RD coincident with -Y_pixel (upward on screen) | 90 |
  | OIM (EDAX) | RD coincident with +Y_pixel (downward on screen) | -90 |
  | HKL (Oxford) | Y0 coincident with +Y_pixel (upward on screen) | 180 |
  | HKL (Oxford) | Y0 coincident with -Y_pixel (downward on screen) | 0 |

  For `.ang` files exported from EDAX OIM software, use 90 (the default).

### Phase properties

```python
"phase_properties": {
    1: "property_files/Quartz.txt",
    2: "property_files/Plagioclase.txt",
},
```

Maps each phase ID (as it appears in the EBSD file) to the corresponding property file.

### Grain preprocessing

```python
"remove_small_grains": False,
"min_grain_pixels": 10,
```

When `remove_small_grains` is `True`, grains with fewer than `min_grain_pixels` data points are merged into a neighboring grain. This is useful for cleaning up noisy EBSD data with spurious small grains.

### Mesh generation

```python
"reuse_mesh": True,
"mesh_type": 1,
"target_elements": 10000,
```

- `mesh_type` — the type of finite element mesh:

  | Type | Description | Best for |
  |------|-------------|----------|
  | **1** | Conforming non-uniform — adapts to grain boundaries | Accurate microscale fields |
  | **2** | Non-conforming hexagonal grid | Fast meshing, large maps |
  | **3** | Non-conforming rectangular grid | Fastest, parametric studies |

- `target_elements` — approximate number of triangular elements. Higher values produce a finer mesh and more accurate results, but take longer to generate and solve. The actual count varies by approximately 10-15%.
- `reuse_mesh` — when `True`, the mesh is saved after generation and reused on subsequent runs, skipping Stages 1 and 2. This is useful when running multiple studies on the same EBSD map.

#### Type 1 mesh settings (advanced)

These settings control the conforming mesh generator and normally do not need to be changed:

```python
"mesh_convergence": [10, 0.2, 0.8, 15],
"mesh_floor_ratio": 0.2,
"junction_refine_ratio": 0.7,
"advanced_mesh_params": [0.01, 0.5, 0.3],
```

- `mesh_convergence` — `[min_iter, q_worst_avg_target, q_mean_target, max_iter]`. Controls the iterative mesh optimization. The mesh iterates until both quality targets are met (after at least `min_iter` iterations) or `max_iter` is reached.
- `mesh_floor_ratio` — minimum element size as a fraction of the maximum. Prevents excessively small elements in narrow grain regions. Set to 0 to disable.
- `junction_refine_ratio` — refines the mesh near triple junction points. The value is a radius in multiples of the maximum element size. Set to 0 to disable.
- `advanced_mesh_params` — `[K, R, g]` controlling curvature refinement, boundary refinement ratio, and size transition rate.

### Thermoelastic analysis

```python
"run_thermoelastic": True,
"macro_mechanical_field_type": "stress",
"macro_mechanical_field": [100e6, 0, 0, 0, 0, 0],
"macro_temperature_field": 0,
```

- `run_thermoelastic` — enables computation of effective elastic stiffness (C), thermal expansion (alpha), and thermal stress (beta) tensors
- `macro_mechanical_field_type` — `"stress"` or `"strain"` to specify the applied macroscale loading for microscale field evaluation, or `"none"` to skip field computation
- `macro_mechanical_field` — the 6-component applied field: [sigma_11, sigma_22, sigma_33, sigma_23, sigma_13, sigma_12] for stress, or the corresponding strain components
- `macro_temperature_field` — applied temperature change (scalar)

In this tutorial, a uniaxial stress of 100 MPa is applied in the x1 direction with no temperature change.

The stress and strain components use the coordinate system: **1** = horizontal (right), **2** = vertical (up), **3** = out-of-plane. This is the same coordinate system as the EBSD map and the mesh.

### Heat conduction analysis

```python
"run_heat_conduction": True,
"macro_thermal_field_type": "temperature_gradient",
"macro_thermal_field": [0, -1, 0],
```

- `run_heat_conduction` — enables computation of the effective thermal conductivity tensor (kappa)
- `macro_thermal_field_type` — `"temperature_gradient"` or `"heat_flux"` for microscale field evaluation, or `"none"` to skip
- `macro_thermal_field` — the 3-component applied field: [dT/dx1, dT/dx2, dT/dx3] or [q1, q2, q3]

In this tutorial, a unit temperature gradient is applied in the negative x2 direction.

### Wave speed analysis

```python
"wave_speed_plots": ["VP", "VS1"],
"wave_speed_plot_type": "both",
"wave_speed_sphere_elev": 30,
"wave_speed_sphere_azim": 30,
```

- `wave_speed_plots` — `"none"` to skip, `"all"` for all 8 fields, or a list of specific field names: `["VP", "VS1", "VS2", "VSH", "VSV", "AVS", "DTS", "DTP"]`.
- `wave_speed_plot_type` — `"lambert"` for 2D Lambert azimuthal projections, `"sphere"` for 3D sphere plots, or `"both"`

## 4. Running the Pipeline

Once `run_tesa.py` is configured, run it from the project root:

```
python run_tesa.py
```

Use `python3 run_tesa.py` if your system requires it.

The pipeline processes each job through four stages:

| Stage | What it does |
|-------|--------------|
| **Stage 1** | Loads the EBSD data, reads phase property files, identifies grains, and plots the microstructure |
| **Stage 2** | Generates the finite element mesh with periodic boundary conditions |
| **Stage 3** | Solves the AEH homogenization problem — computes characteristic functions and effective properties |
| **Stage 4** | Saves result files and generates all plots (characteristic functions, microscale fields, wave speeds) |

With `verbose_console` set to `"high"`, you will see detailed progress for each stage, including mesh quality metrics, solver timings, and effective property summaries.

## 5. Reviewing Results

Results are saved in a folder structure under `results/`:

```
results/Tutorial/
├── input_data/                         — Copy of EBSD + property files
├── microstructure/                     — Phase maps, Euler angle plots, grain boundaries
├── mesh/                               — Mesh figures, convergence plots
│   └── diagnostics/                    — Mesh statistics, element quality data
├── saved_mesh/                         — Saved mesh for reuse (mesh.pkl + metadata)
│
└── study_tutorial_analyses/            — Results for this study
    ├── AEH_characteristic_functions/   — Chi, psi, phi plots
    ├── homogenized_properties/         — Effective property text files
    │   ├── AEH/
    │   ├── Voigt/
    │   ├── Reuss/
    │   ├── Hill/
    │   └── GeoMean/
    ├── wave_speed_plots/               — Wave speed Lambert and sphere plots
    ├── microscale_fields/              — Stress, strain, heat flux, temp gradient plots
    │   ├── stresses/
    │   ├── strains/
    │   ├── heat_flux/
    │   └── temp_gradient/
    └── log.md                          — Analysis log
```

### Microstructure

Stage 1 parses the EBSD data into grains colored by phase. Grain boundaries are smoothed using RBF (Radial Basis Function) interpolation to produce realistic contours from the pixelated EBSD data.

<p align="center">
  <img src="images/grains_phase.png" height="350" alt="Grain map colored by phase">
  <img src="images/grain_overlay_comparison.png" height="350" alt="RBF-smoothed grain boundaries">
</p>

### Mesh

Stage 2 generates a conforming non-uniform mesh (Type 1) that adapts element size to grain boundary curvature and junction points. The mesh size function controls local refinement — smaller elements near boundaries, larger in grain interiors.

<p align="center">
  <img src="images/mesh_size_function.png" height="350" alt="Mesh size function">
  <img src="images/final_mesh.png" height="350" alt="Final conforming mesh">
</p>

The final mesh is overlaid on the original grain boundaries and the phase map to verify alignment:

<p align="center">
  <img src="images/final_mesh_on_original_GB.png" height="350" alt="Final mesh on original grain boundaries">
  <img src="images/final_mesh_on_original_phase_map.png" height="350" alt="Final mesh on original phase map">
</p>

Mesh quality convergence is tracked using three metrics: q_mean (bulk quality), q_worst_avg (mean quality of the worst 0.5% of elements), and q_min (absolute worst element).

<p align="center">
  <img src="images/mesh_quality_convergence.png" width="500" alt="Mesh quality convergence">
</p>

### AEH Characteristic Functions

Stage 3 solves for the characteristic functions (chi, psi, phi) on the periodic unit cell. These encode how the microstructure redistributes applied macroscale fields.

**Chi** (elastic characteristic functions) — there are 18 chi fields (6 independent strain components x 3 displacement components). Each describes the microscale displacement response to a unit macroscale strain. Below is chi_11 (u1 component), showing how the microstructure redistributes a uniaxial strain in the x1 direction.

<p align="center">
  <img src="images/chi_11_u1.png" width="450" alt="Chi_11 u1 elastic characteristic function">
</p>

**Psi** (thermal stress characteristic function) — there are 3 psi fields (one per displacement component). These describe the microscale displacement response to a unit temperature change, capturing the coupling between thermal and mechanical behavior. Below is psi (u1 component).

<p align="center">
  <img src="images/psi_u1.png" width="450" alt="Psi u1 thermal stress characteristic function">
</p>

**Phi** (thermal conductivity characteristic functions) — there are 3 phi fields (one per spatial direction). These describe the microscale temperature response to a unit macroscale temperature gradient. Below is phi_1.

<p align="center">
  <img src="images/phi_1.png" width="450" alt="Phi_1 thermal characteristic function">
</p>

### Thermoelastic Microscale Fields

Under the applied uniaxial stress of 100 MPa in x1, TESA computes the microscale stress and strain fields. The fields are smooth within grains and discontinuous at grain boundaries, reflecting the anisotropic single-crystal properties and crystallographic misorientation between grains.

Below are the normal stress sigma_11 and the normal strain epsilon_11 in the loading direction. The stress varies significantly across grains due to elastic anisotropy — even though the macroscale stress is uniform, each grain experiences a different local stress state depending on its orientation and its neighbors.

<p align="center">
  <img src="images/sigma_11.png" height="300" alt="Sigma_11 normal stress field">
  <img src="images/epsilon_11.png" height="300" alt="Epsilon_11 normal strain field">
</p>

The maximum shear stress (tau_max) highlights stress concentrations near grain boundaries and triple junctions, where crystallographic mismatch between grains is largest.

<p align="center">
  <img src="images/tau_max.png" width="450" alt="Maximum shear stress field">
</p>

TESA computes all 6 stress and strain components (sigma_11 through sigma_12, epsilon_11 through epsilon_12), plus 3 principal stresses, 3 principal strains, and the maximum shear stress — 19 field plots in total. These are saved in `microscale_fields/stresses/` and `microscale_fields/strains/`.

### Heat Conduction Microscale Fields

Under the applied temperature gradient of [0, -1, 0] (unit gradient in the negative x2 direction), TESA computes the microscale heat flux and temperature gradient fields. Below are the heat flux magnitude contour, the heat flux vector field, and the temperature gradient magnitude.

<p align="center">
  <img src="images/q_magnitude_2D.png" height="300" alt="Heat flux magnitude">
  <img src="images/q_vector_2D.png" height="300" alt="Heat flux vectors">
</p>

<p align="center">
  <img src="images/grad_T_magnitude_2D.png" width="450" alt="Temperature gradient magnitude">
</p>

The vector arrows show the direction and relative magnitude of the in-plane heat flux at each point. The heat flux is smooth within grains but changes direction and magnitude across grain boundaries due to the anisotropic thermal conductivity of each crystal.

TESA computes 3 heat flux components, 3 temperature gradient components, magnitudes, and vector fields — 12 field plots in total. These are saved in `microscale_fields/heat_flux/` and `microscale_fields/temp_gradient/`.

### Wave Speeds

TESA computes seismic wave speeds (phase velocities) as a function of propagation direction from the effective stiffness tensor. Wave speeds are plotted as Lambert azimuthal equal-area projections (2D) and as 3D sphere plots.

Below are the P-wave velocity (VP) shown as both a Lambert projection and a 3D sphere, and the S1-wave velocity (VS1) as a 3D sphere. These are computed from the AEH effective stiffness.

<p align="center">
  <img src="images/VP_lambert.png" height="300" alt="VP P-wave velocity Lambert projection">
  <img src="images/VP_sphere.png" height="300" alt="VP P-wave velocity sphere plot">
</p>

<p align="center">
  <img src="images/VS1_sphere.png" width="450" alt="VS1 S-wave velocity sphere plot">
</p>

TESA computes 8 wave speed fields: VP, VS1, VS2, VSH, VSV, AVS, DTS, and DTP. These are evaluated for the AEH effective stiffness as well as the Voigt, Reuss, Hill, and Geometric Mean bounds, plus the single-crystal stiffness of each phase. All plots are saved in `wave_speed_plots/`.

### Effective Properties

Effective properties are computed using AEH along with analytical bounds (Voigt, Reuss, Hill, Geometric Mean). Results are saved as text files in the `homogenized_properties/` subfolders. Example output for thermal conductivity:

```
Effective conductivity kappa_AEH (W/(m K)):
      4.0204     -0.0091      0.1488
     -0.0091      3.7421     -0.0630
      0.1488     -0.0630      5.1362
```

## 6. Next Steps

Now that you have completed the tutorial, here are some things to try:

- **Different mesh types** — Change `mesh_type` to 2 or 3 to compare the non-conforming mesh results with the conforming mesh. Types 2 and 3 are faster to generate but produce approximate microscale fields.
- **Multiple studies** — Run different analyses on the same EBSD map by creating additional jobs with different `study_name` values. Each study gets its own results subfolder while sharing the mesh and microstructure data.
- **Mesh reuse** — With `reuse_mesh` set to `True`, the mesh is saved after the first run. Subsequent runs skip Stages 1 and 2, going directly to the solver. This is useful for running parametric studies with different loading conditions.
- **Your own EBSD data** — Replace `Tutorial.ang` with your own EBSD file. Update the column numbers (`euler_col`, `xy_col`, `phase_col`) and create property files for your phases.
- **Larger maps** — Increase `target_elements` for finer meshes and more accurate results. For large EBSD maps, consider starting with a Type 2 or 3 mesh for quick results before running a Type 1 mesh.
