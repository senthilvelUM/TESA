"""
Save and load mesh (Microstructure object) for reuse across analyses.

After Stage 2 (mesh generation), the ms object is saved as a pickle file
along with a JSON metadata file for verification. On subsequent runs,
the saved mesh can be loaded instantly, skipping the expensive mesh
generation step.

The metadata file stores the EBSD file checksum and mesh parameters so
that loading a mismatched mesh is detected and prevented.
"""

import os
import json
import pickle
import hashlib
import sys
from datetime import datetime
import numpy as np


def _compute_file_checksum(filepath):
    """
    Compute SHA-256 checksum of a file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    checksum : str
        Hex digest prefixed with ``'sha256:'``.
    """
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _get_mesh_metadata(ms, job):
    """
    Build metadata dict from current job and Microstructure object.

    The metadata is saved alongside the pickle file and used to verify
    that a saved mesh matches the current job parameters on reload.

    Parameters
    ----------
    ms : Microstructure
        Microstructure object after mesh generation.
    job : dict
        Job dictionary with EBSD file path and mesh parameters.

    Returns
    -------
    metadata : dict
        Dictionary with keys: ebsd_file, ebsd_checksum, mesh_type,
        advanced_mesh_params, mesh_convergence, target_elements,
        n_elements, n_nodes, n_grains, timestamp, python_version.
    """
    ebsd_file = job["ebsd_file"]
    return {
        "ebsd_file": ebsd_file,
        "ebsd_checksum": _compute_file_checksum(ebsd_file),
        "mesh_type": job["mesh_type"],
        "advanced_mesh_params": list(job.get("advanced_mesh_params", [0.01, 0.5, 0.3])),
        "mesh_convergence": list(job.get("mesh_convergence", [20, 0.1, 0.80, 50])),
        "target_elements": job.get("target_elements", 2000),
        "n_elements": int(ms.NumberElements) if ms.NumberElements is not None else 0,
        "n_nodes": int(ms.NumberNodes) if ms.NumberNodes is not None else 0,
        "n_grains": len([g for g in ms.Grains if g is not None and np.asarray(g).size > 0])
                    if ms.Grains is not None else 0,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def save_mesh(ms, run_dir, job):
    """
    Save Microstructure object and metadata after mesh generation.

    Parameters
    ----------
    ms : Microstructure
        The ms object after Stage 2 (create_mesh).
    run_dir : str
        Results directory for this job.
    job : dict
        Job dictionary with EBSD file path and mesh parameters.

    Returns
    -------
    pkl_path : str
        Path to the saved pickle file.
    """
    # Remove non-serializable attributes (functions stored on ms during plotting)
    _non_serializable = {}
    for attr in list(vars(ms)):
        val = getattr(ms, attr)
        if callable(val) and not isinstance(val, type):
            _non_serializable[attr] = val
            delattr(ms, attr)

    # Save pickle to saved_mesh/ subfolder
    save_dir = os.path.join(run_dir, "saved_mesh")
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, "mesh.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(ms, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Restore non-serializable attributes
    for attr, val in _non_serializable.items():
        setattr(ms, attr, val)

    # Save metadata
    metadata = _get_mesh_metadata(ms, job)
    json_path = os.path.join(save_dir, "mesh_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Report file size
    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"  Mesh saved for reuse (saved_mesh/mesh.pkl, {size_mb:.1f} MB)")

    return pkl_path


def load_mesh(run_dir, job):
    """
    Load a saved Microstructure object if it matches the current job.

    Checks the metadata file to verify that the saved mesh was generated
    from the same EBSD file with the same mesh parameters. If any mismatch
    is detected, returns None (caller should regenerate the mesh).

    Parameters
    ----------
    run_dir : str
        Results directory for this job.
    job : dict
        Job dictionary with EBSD file path and mesh parameters.

    Returns
    -------
    ms : Microstructure or None
        The loaded ms object, or None if no valid saved mesh exists.
    """
    save_dir = os.path.join(run_dir, "saved_mesh")
    pkl_path = os.path.join(save_dir, "mesh.pkl")
    json_path = os.path.join(save_dir, "mesh_metadata.json")

    # Check if files exist
    if not os.path.exists(pkl_path) or not os.path.exists(json_path):
        return None

    # Load and verify metadata
    try:
        with open(json_path, 'r') as f:
            saved = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("  WARNING: mesh_metadata.json is corrupted — regenerating mesh")
        return None

    # Compute current EBSD file checksum
    ebsd_file = job["ebsd_file"]
    if not os.path.exists(ebsd_file):
        print(f"  WARNING: EBSD file not found: {ebsd_file} — regenerating mesh")
        return None

    current_checksum = _compute_file_checksum(ebsd_file)

    # Verify each field
    mismatches = []

    if saved.get("ebsd_checksum") != current_checksum:
        mismatches.append(f"EBSD file checksum differs (file may have been modified)")

    if saved.get("mesh_type") != job["mesh_type"]:
        mismatches.append(f"mesh_type: saved={saved.get('mesh_type')}, current={job['mesh_type']}")

    # Compare advanced_mesh_params (as lists, with tolerance for floats)
    saved_params = saved.get("advanced_mesh_params", saved.get("mesh_params", []))
    current_params = list(job.get("advanced_mesh_params", [0.01, 0.5, 0.3]))
    if len(saved_params) != len(current_params) or \
       any(abs(s - c) > 1e-10 for s, c in zip(saved_params, current_params)):
        mismatches.append(f"advanced_mesh_params: saved={saved_params}, current={current_params}")

    saved_conv = saved.get("mesh_convergence", [])
    current_conv = list(job.get("mesh_convergence", [20, 0.1, 0.80, 50]))
    if len(saved_conv) != len(current_conv) or \
       any(abs(s - c) > 1e-10 for s, c in zip(saved_conv, current_conv)):
        mismatches.append(f"mesh_convergence: saved={saved_conv}, current={current_conv}")

    saved_te = saved.get("target_elements")
    current_te = job.get("target_elements", 2000)
    if saved_te is not None and saved_te != current_te:
        mismatches.append(f"target_elements: saved={saved_te}, current={current_te}")

    # Report mismatches
    if mismatches:
        print("  Saved mesh found but parameters differ:")
        for m in mismatches:
            print(f"    {m}")
        print("  Regenerating mesh...")
        return None

    # Load pickle
    try:
        with open(pkl_path, 'rb') as f:
            ms = pickle.load(f)
    except Exception as e:
        print(f"  WARNING: Failed to load mesh.pkl ({e}) — regenerating mesh")
        return None

    # Report success
    n_elem = saved.get("n_elements", "?")
    n_nodes = saved.get("n_nodes", "?")
    n_grains = saved.get("n_grains", "?")
    print(f"  Loaded saved mesh: {n_elem} elements, {n_nodes} nodes, {n_grains} grains")
    print(f"  (saved_mesh/mesh.pkl verified — EBSD checksum and mesh parameters match)")

    return ms
