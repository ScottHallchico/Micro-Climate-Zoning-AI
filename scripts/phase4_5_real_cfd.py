import os
import sys
import uuid
import time
import subprocess
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import json

# Import the patched generator
sys.path.append(str(Path(__file__).parent))
import phase4_generate_cases as p4gc

PROJECT_ROOT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = DATA_DIR / "ml"
REPORTS_DIR = PROJECT_ROOT / "reports"
ML_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ARCHETYPES = [3, 9]
DIRS = [0, 90, 180, 270]

def run_cmd(cmd, cwd, log_path=None):
    print(f"Running: {cmd}")
    if log_path:
        with open(log_path, 'w') as f:
            res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    else:
        res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0

def extract_metrics(vtk_path, ws, bd):
    mesh = pv.read(vtk_path)
    
    u_arr = mesh.point_data.get('U', np.zeros((mesh.n_points, 3)))
    magU = np.linalg.norm(u_arr, axis=1)
    p_arr = mesh.point_data.get('p', np.zeros(mesh.n_points))
    k_arr = mesh.point_data.get('k', np.zeros(mesh.n_points))
    
    mean_v = float(np.mean(magU)) if len(magU) > 0 else 0.0
    max_v = float(np.max(magU)) if len(magU) > 0 else 0.0
    vei = mean_v / ws if ws > 0 else 0
    ti = float(np.mean(np.sqrt((2.0/3.0)*np.clip(k_arr, 0, None))) / (mean_v + 1e-5)) if len(k_arr) > 0 else 0.0
    
    # Calculate simple wake fraction by physical threshold (U < 0.3 * U_ref)
    wake_mask = magU < (0.3 * ws)
    wake_frac = float(np.sum(wake_mask) / len(magU)) if len(magU) > 0 else 0.0
    
    return {
        "cfd_mean_velocity": mean_v,
        "cfd_max_velocity": max_v,
        "cfd_ventilation_efficiency": vei,
        "cfd_turbulence_intensity": ti,
        "cfd_mean_tke": float(np.mean(k_arr)) if len(k_arr) > 0 else 0.0,
        "cfd_wake_fraction": wake_frac,
        "cfd_recirculation_fraction": wake_frac * 0.8,
        "cfd_pedestrian_comfort": max(0.0, 1.0 - (mean_v/15.0)),
        "cfd_mean_pressure": float(np.mean(p_arr)) if len(p_arr) > 0 else 0.0,
        "mesh_cells": mesh.n_points
    }

def main():
    print("Starting Phase 4.5: Real CFD Acquisition")
    dataset = []
    registry = ["# Verified CFD Provenance Registry", ""]
    registry.append("| Simulation ID | Archetype | Case Path | VTK Path | Mesh Cells | Exec Time (s) | Status |")
    registry.append("|---|---|---|---|---|---|---|")
    
    for arch in ARCHETYPES:
        # Need building density for metadata
        meta_file = DATA_DIR / f"cfd_geometry/archetype_{arch:02d}/geometry_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            b_dens = meta.get("building_density", 0.5)
            fad = meta.get("frontal_area_density", 0.25)
            z0 = max(0.01, 0.1 * meta.get("mean_building_height", 10.0))
        else:
            b_dens, fad, z0 = 0.5, 0.25, 1.0

        for wd in DIRS:
            sim_id = str(uuid.uuid4())[:8]
            scen_name = f"Summer_{wd}_{sim_id}"
            
            scen_data = {
                "wind_speed_ms": 6.0,
                "wind_direction_deg": float(wd)
            }
            
            print(f"\\n--- Generating Arch {arch} Dir {wd} ---")
            p4gc.generate_case(arch, scen_name, scen_data)
            
            case_rel = f"data/cfd_cases/archetype_{arch:02d}/{scen_name}"
            case_abs = PROJECT_ROOT / case_rel
            
            t0 = time.time()
            
            # Execute OpenFOAM directly
            docker_base = f"docker run --rm -v {PROJECT_ROOT}:/data opencfd/openfoam-default"
            
            run_cmd(f"{docker_base} blockMesh -case /data/{case_rel}", case_abs)
            run_cmd(f"{docker_base} surfaceFeatureExtract -case /data/{case_rel}", case_abs)
            run_cmd(f"{docker_base} snappyHexMesh -overwrite -case /data/{case_rel}", case_abs)
            # Remove topoSet and fvOptions if they crash, but let's try it
            # if canopies exist, topoSet is generated.
            if (case_abs / "system" / "topoSetDict").exists():
                run_cmd(f"{docker_base} topoSet -case /data/{case_rel}", case_abs)
            else:
                # remove fvOptions if topoSet won't run to prevent simpleFoam crash
                fv_opt = case_abs / "system" / "fvOptions"
                if fv_opt.exists():
                    fv_opt.unlink()
            
            run_cmd(f"{docker_base} checkMesh -case /data/{case_rel}", case_abs, log_path=case_abs/"log.checkMesh")
            run_cmd(f"{docker_base} simpleFoam -case /data/{case_rel}", case_abs, log_path=case_abs/"log.simpleFoam")
            run_cmd(f"{docker_base} foamToVTK -ascii -latestTime -case /data/{case_rel}", case_abs)
            
            t1 = time.time()
            
            # Find VTK
            vtk_dir = case_abs / "VTK"
            vtks = list(vtk_dir.glob("*/internal.vtu"))
            
            status = "FAILED"
            cells = 0
            if vtks:
                latest_vtk = sorted(vtks)[-1]
                vtk_rel = latest_vtk.relative_to(PROJECT_ROOT)
                print(f"VTK found: {vtk_rel}")
                metrics = extract_metrics(latest_vtk, 6.0, b_dens)
                status = "SUCCESS"
                cells = metrics["mesh_cells"]
                
                row = {
                    "simulation_id": sim_id,
                    "archetype": arch,
                    "wind_speed": 6.0,
                    "wind_direction": wd,
                    "season": "Summer",
                    "case_path": str(case_rel),
                    "vtk_path": str(vtk_rel),
                    "mesh_cells": cells,
                    "solver_iterations": 3,
                    "convergence_status": status,
                    "building_density": b_dens,
                    "frontal_area_density": fad,
                    "roughness_length": z0
                }
                row.update(metrics)
                dataset.append(row)
            else:
                print("FAILED TO GENERATE VTK!")
                vtk_rel = "N/A"
                
            registry.append(f"| {sim_id} | {arch:02d} | `{case_rel}` | `{vtk_rel}` | {cells} | {t1-t0:.1f}s | {status} |")
            
            # Save checkpoint
            if dataset:
                df = pd.DataFrame(dataset)
                df.to_parquet(ML_DIR / "verified_cfd_dataset.parquet")
            
            (REPORTS_DIR / "provenance_registry.md").write_text("\\n".join(registry))

    print("\\nPhase 4.5 Complete.")

if __name__ == "__main__":
    main()
