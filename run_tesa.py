"""
TESA Toolbox — Thermal and Elastic Scale-bridging Analysis of Polycrystalline Microstructures

Main pipeline script. Configure your analysis below and run:
    python run_tesa.py or python3 run_tesa.py
"""

from tesa.run_pipeline import run_job

# ── Global Settings ─────────────────────────────────────────────────────
# These apply to all jobs.
settings = {
    # ── Output verbosity ──────────────────────────────────────────
    "verbose_console": "high",          # "none" = banner + errors only, "medium" = key results, "high" = all details
    "verbose_log": "high",              # "none" = header only, "medium" = summary tables, "high" = full console mirror

    # ── Figure display ────────────────────────────────────────────
    "show_figures": True,               # True = display figures on screen, False = save to files only
    "figure_pause": 1.0,               # Seconds to display each figure on screen (if show_figures=True)
    "figure_dpi": 150,                  # Resolution of saved figures (dots per inch)

    # ── Microstructure plot colors ────────────────────────────────
    "phase_colors": ['red', 'lime', 'tab:blue', 'tab:orange', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
                                        # One color per phase (cycles if more phases than colors)
    "phase_colors_alpha": 0.9,          # Opacity of phase-colored fills (0.0=transparent, 1.0=opaque)
    "grain_colormap": "tab20",          # Matplotlib colormap for grain shading (e.g., "tab20", "Set3", "Paired")
    "grain_colormap_alpha": 0.9,        # Opacity of grain-colored fills (0.0=transparent, 1.0=opaque)

    # ── Mesh overlay lines ───────────────────────────────────────
    "mesh_overlay_alpha": 0.5,          # Opacity of mesh wireframe on overlay figures (0.0=invisible, 1.0=opaque)
    "mesh_overlay_linewidth": 0.3,      # Line thickness of mesh wireframe on overlay figures

    # ── Field plot settings ───────────────────────────────────────
    "field_colormap": "turbo",          # Colormap for field plots: "rainbow", "turbo", "jet", "viridis"
    "field_plot_style": "default",      # "default" = rbf for Type 1, smooth for Types 2/3. Or "element" for Types 2/3.
    "field_fontsize": 16,               # Font size for colorbar labels and tick labels in field plots
    "show_grain_boundaries": True,      # True = overlay grain boundaries on field plots, False = suppress
    "arrow_length": 1.65,               # Arrow length scale factor (1.0=one grid cell, 0.5=shorter, 2.0=longer)
    "arrow_stem_width": 0.0012,          # Shaft width as fraction of plot width
    "arrow_alpha": 0.9,                 # Opacity of vector arrows (0.0=invisible, 1.0=opaque)

    # ── Reproducibility ───────────────────────────────────────────
    "random_seed": 10,                  # Fixed integer for reproducible mesh; None for a different mesh each run
}

# ── Job Configuration ───────────────────────────────────────────────────
# Each job specifies an EBSD file, its column layout, and phase property files.
# Results for each job go in a separate folder under results/.
#
# EBSD data (column numbers are 1-indexed):
#   euler_col       : first Euler angle column (phi1; Phi and phi2 are the next two)
#   xy_col          : X-coordinate column (Y-coordinate is the next column)
#   phase_col       : phase ID column
#   ref_frame_angle : EBSD-to-mesh coordinate system rotation (degrees, typically 90)
#
# Grain preprocessing:
#   remove_small_grains       : (optional) True = merge grains with <= min_grain_pixels data points
#                               into a neighboring grain; False = keep all grains (default)
#   min_grain_pixels          : (optional) minimum data points a grain must have to be kept (default 10)
#
# Mesh settings:
#   reuse_mesh           : True = save mesh after generation and load on subsequent runs (skips regeneration)
#   mesh_type            : 1=conforming non-uniform, 2=non-conforming hexgrid, 3=non-conforming rectangular
#                          Type 1: Use for accurate microscale stress/strain/heat flux fields
#                                  (mesh conforms to grain boundaries, slower to generate)
#                          Types 2/3: Use for rapid evaluation of effective/homogenized properties
#                                     (structured grid, fast, but microscale fields are approximate)
#   advanced_mesh_params : [K, R, g] (Type 1 only) — advanced controls for element size distribution.
#                          These normally do not need to be changed from their defaults.
#                            K = curvature refinement (default 0.01). Controls element refinement near
#                                high-curvature grain boundaries. Smaller K = finer elements at curved
#                                boundaries. Effect is limited to a thin strip near boundaries; for
#                                typical EBSD maps with smooth RBF boundaries, K has minimal effect.
#                            R = boundary refinement ratio (default 0.5). Controls how much finer
#                                elements are near grain boundaries vs grain interiors. At R=1.0,
#                                element size equals the local grain width. Higher R = more refinement
#                                near boundaries of narrow grains. Lower R = more uniform sizing.
#                            g = size transition rate (default 0.3). Controls how quickly element size
#                                can change from one region to the next. Lower g = more uniform mesh
#                                (elements change size slowly). Higher g = more variation (small elements
#                                near boundaries, large elements in interiors). Range: 0.1 (nearly
#                                uniform) to 0.5+ (strongly graded). Note: narrow grain boundaries will
#                                always have finer spacing than wide grain boundaries.
#   target_elements      : target number of elements (all mesh types). The mesh size function
#                          h(x,y) is scaled to produce approximately this many elements.
#                          Actual count varies ±10-15% due to boundary effects, element shape
#                          variation, and mesh optimization. Higher values = finer mesh, more
#                          accurate results, longer computation time.
#   mesh_floor_ratio      : (Type 1 only) minimum element size as a fraction of the maximum.
#                          Default 0.25 = smallest element ≥ 25% of largest (max/min ≤ 4.0).
#                          Prevents excessively small elements in narrow grain regions.
#                          Set to 0 to disable the floor entirely.
#   junction_refine_ratio : (Type 1 only) refine mesh near junction (triple) points.
#                          Radius in multiples of h_max. Within this radius, element size is
#                          capped at h_floor, ensuring fine mesh at stress concentrations.
#                          Default 0.7. Set to 0 to disable.
#   mesh_convergence     : [min_iter, q_worst_avg_target, q_mean_target, max_iter] (Type 1 only)
#                          q_worst_avg = mean quality of worst 0.5% of elements (convergence metric)
#
# Analysis settings:
#   element_homogenization : Element-level averaging for non-conforming meshes (Types 2, 3):
#                            1=Nearest, 2=Voigt, 3=Reuss, 4=Hill, 5=Geometric Mean
#
# Thermoelastic analysis:
#   run_thermoelastic             : True/False — compute effective C, α, β (solves chi, psi)
#   macro_mechanical_field_type   : "none" = skip fields, "stress" or "strain"
#   macro_mechanical_field        : [σ11, σ22, σ33, σ23, σ13, σ12] or [ε11, ε22, ε33, ε23, ε13, ε12]
#   macro_temperature_field       : scalar ΔT
#
# Heat conduction analysis:
#   run_heat_conduction           : True/False — compute effective κ (solves phi)
#   macro_thermal_field_type      : "none" = skip fields, "temperature_gradient" or "heat_flux"
#   macro_thermal_field           : [∂T/∂x1, ∂T/∂x2, ∂T/∂x3] or [q1, q2, q3]
#
# Wave speed settings:
#   wave_speed_plots    : "none" = skip, "all" = all 8 fields, or list of field names
#                          ["VP", "VS1", "VS2", "VSH", "VSV", "AVS", "DTS", "DTP"]
#   wave_speed_plot_type : "lambert" = 2D disk, "sphere" = 3D, "both" = both
jobs = [
    {
        # ── EBSD data ─────────────────────────────────────────────────
        "ebsd_file": "EBSD_maps/Tutorial.ang",
        "study_name": "tutorial_analyses",                          # Results subfolder name (default: "default")
        "euler_col": 1,
        "xy_col": 4,
        "phase_col": 6,  # Phase ID column (varies by EBSD format: .ang files typically use 6 or 8)
        "ref_frame_angle": 90,
        
        # ── Phase properties ──────────────────────────────────────────
        "phase_properties": {
            1: "property_files/Quartz.txt",
            2: "property_files/Plagioclase.txt",
        },
        
        # ── Grain preprocessing ───────────────────────────────────────
        "remove_small_grains": False,                       # True = absorb grains with <= min_grain_pixels data points into neighbors
        "min_grain_pixels": 10,                             # Minimum data points per grain; grains at or below this are merged (requires remove_small_grains=True)
        
        # ── Mesh generation ───────────────────────────────────────────
        "reuse_mesh": True,                                 # Save mesh after generation; load saved mesh if available
        "mesh_type": 1,                                     # 1=conforming non-uniform, 2=non-conforming hexgrid, 3=non-conforming rectangular
        "target_elements": 10000,                           # Target number of elements (all mesh types; actual count ±10-15%)
        
        # ── Type 1 mesh settings (conforming non-uniform) ────────────
        "mesh_convergence": [10, 0.2, 0.8, 20],             # [min_iter, q_worst_avg, q_mean, max_iter]. Recommended [25, 0.2, 0.8, 50]
        "mesh_floor_ratio": 0.2,                           # h_min = floor × h_max. Limits max/min element size ratio.
                                                            # 0.2 = smallest element ≥ 20% of largest (ratio ≤ 5.0). 0 = no floor.
        "junction_refine_ratio": 0.7,                       # Refine mesh near junction (triple) points.
                                                            # Value = radius in multiples of h_max. 0 = disabled.
        "advanced_mesh_params": [0.01, 0.5, 0.3],           # [K, R, g] — advanced controls (normally do not need to be changed)
        
        # ── Analysis settings ─────────────────────────────────────────
        "element_homogenization": 4,                        # Non-conforming meshes only (Types 2, 3):
                                                            # 1=Nearest, 2=Voigt, 3=Reuss, 4=Hill, 5=GeoMean
        # ── Thermoelastic analysis ────────────────────────────────────
        "run_thermoelastic": True,                         # Effective C, α, β (solves chi, psi)
        "macro_mechanical_field_type": "stress",            # "none" = skip fields, "stress" or "strain"
        "macro_mechanical_field": [100e6, 0, 0, 0, 0, 0],   # [σ11, σ22, σ33, σ23, σ13, σ12] or [ε11, ε22, ε33, ε23, ε13, ε12]
        "macro_temperature_field": 0,                       # ΔT (scalar)
        
        # ── Heat conduction analysis ──────────────────────────────────
        "run_heat_conduction": True,                        # Effective κ (solves phi)
        "macro_thermal_field_type": "temperature_gradient", # "none" = skip fields, "temperature_gradient" or "heat_flux"
        "macro_thermal_field": [0, -1, 0],                  # [∂T/∂x1, ∂T/∂x2, ∂T/∂x3] or [q1, q2, q3]
        
        # ── Wave speed settings ──────────────────────────────────────
        "wave_speed_plots": "VP",                        # "none" = skip, "all" = all 8 fields, or list:
                                                            # ["VP", "VS1", "VS2", "VSH", "VSV", "AVS", "DTS", "DTP"]
        "wave_speed_plot_type": "both",                     # "lambert" = 2D disk, "sphere" = 3D, "both" = both
        "wave_speed_sphere_elev": 30,                         # Sphere plot elevation above x-y plane (degrees)
        "wave_speed_sphere_azim": 30,                         # Sphere plot azimuth from +x axis in x-y plane (degrees)
    },
    # ── Additional jobs ──────────────────────────────────────────────────
    # To add more jobs, copy the job dictionary above, paste it here,
    # and modify the settings as needed. Each job runs independently
    # through the full pipeline. Use a different "study_name" for each.
    #
    # { ... },  # Job 2
    # { ... },  # Job 3
]

# ── Pipeline ───────────────────────────────────────────────────────────
# Process each job sequentially through the four-stage pipeline.

ms_list = []

for job_num, job in enumerate(jobs, start=1):
    ms, run_dir = run_job(job, job_num, len(jobs), settings)
    ms_list.append(ms)
