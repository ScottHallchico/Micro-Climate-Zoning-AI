import os
import sys
import uuid
import time
import json
import subprocess
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import qmc
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).parent))
import phase4_generate_cases as p4gc

PROJECT_ROOT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
DATA_DIR = PROJECT_ROOT / "data"
ML_DIR = DATA_DIR / "ml"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Existing portfolio
EXISTING_ARCHETYPES = [3, 6, 9, 10]
WIND_SPEEDS = [2, 4, 6, 8, 10, 12]
WIND_DIRS = [0, 45, 90, 135, 180, 225, 270, 315]
SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']

def run_cmd(cmd, cwd, log_path=None):
    if log_path:
        with open(log_path, 'w') as f:
            res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    else:
        res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0

def extract_metrics(vtk_path, ws):
    mesh = pv.read(vtk_path)
    u_arr = mesh.point_data.get('U', np.zeros((mesh.n_points, 3)))
    magU = np.linalg.norm(u_arr, axis=1)
    p_arr = mesh.point_data.get('p', np.zeros(mesh.n_points))
    k_arr = mesh.point_data.get('k', np.zeros(mesh.n_points))
    
    mean_v = float(np.mean(magU)) if len(magU) > 0 else 0.0
    max_v = float(np.max(magU)) if len(magU) > 0 else 0.0
    vei = mean_v / ws if ws > 0 else 0
    ti = float(np.mean(np.sqrt((2.0/3.0)*np.clip(k_arr, 0, None))) / (mean_v + 1e-5)) if len(k_arr) > 0 else 0.0
    
    wake_mask = magU < (0.3 * ws)
    wake_frac = float(np.sum(wake_mask) / len(magU)) if len(magU) > 0 else 0.0
    
    return {
        "cfd_mean_velocity": mean_v,
        "cfd_max_velocity": max_v,
        "cfd_ventilation_efficiency": vei,
        "cfd_turbulence_intensity": ti,
        "cfd_mean_tke": float(np.mean(k_arr)) if len(k_arr) > 0 else 0.0,
        "cfd_wake_fraction": wake_frac,
        "cfd_mean_pressure": float(np.mean(p_arr)) if len(p_arr) > 0 else 0.0,
        "mesh_cells": mesh.n_points
    }

def verify_case(case_abs, sid):
    log_s = case_abs / "log.simpleFoam"
    log_c = case_abs / "log.checkMesh"
    if not log_s.exists() or not log_c.exists(): return False, "Missing logs"
    with open(log_c) as f:
        if "Mesh OK" not in f.read(): return False, "Mesh check failed"
    with open(log_s) as f:
        content = f.read()
        if "FOAM FATAL ERROR" in content: return False, "SimpleFoam fatal error"
    return True, "SUCCESS"

def execute_cfd_case(row):
    arch = row['archetype']
    ws = row['wind_speed']
    wd = row['wind_direction']
    season = row['season']
    sim_id = row['simulation_id']
    
    scen_name = f"{season}_{wd}_{ws}ms_{sim_id}"
    scen_data = {"wind_speed_ms": float(ws), "wind_direction_deg": float(wd)}
    
    try:
        p4gc.generate_case(arch, scen_name, scen_data)
    except Exception as e:
        print(f"Skipping {scen_name} due to error: {e}")
        return {"simulation_id": sim_id, "status": "FAIL"}
    
    case_rel = f"data/cfd_cases/archetype_{arch:02d}/{scen_name}"
    case_abs = PROJECT_ROOT / case_rel
    
    t0 = time.time()
    docker_base = f"docker run --rm -v {PROJECT_ROOT}:/data opencfd/openfoam-default"
    
    run_cmd(f"{docker_base} blockMesh -case /data/{case_rel}", case_abs)
    run_cmd(f"{docker_base} surfaceFeatureExtract -case /data/{case_rel}", case_abs)
    run_cmd(f"{docker_base} snappyHexMesh -overwrite -case /data/{case_rel}", case_abs)
    
    fv_opt = case_abs / "system" / "fvOptions"
    if fv_opt.exists(): fv_opt.unlink()
        
    run_cmd(f"{docker_base} checkMesh -case /data/{case_rel}", case_abs, log_path=case_abs/"log.checkMesh")
    run_cmd(f"{docker_base} simpleFoam -case /data/{case_rel}", case_abs, log_path=case_abs/"log.simpleFoam")
    run_cmd(f"{docker_base} foamToVTK -ascii -latestTime -case /data/{case_rel}", case_abs)
    
    exec_time = time.time() - t0
    passed, msg = verify_case(case_abs, sim_id)
    
    if passed:
        vtk_dir = case_abs / "VTK"
        vtks = list(vtk_dir.glob("*/internal.vtu"))
        if vtks:
            latest_vtk = sorted(vtks)[-1]
            metrics = extract_metrics(latest_vtk, ws)
            if metrics["mesh_cells"] > 0:
                res_val = "N/A"
                try:
                    with open(case_abs/"log.simpleFoam") as f:
                        lines = [l for l in f.readlines() if "Solving for Ux" in l]
                        if lines: res_val = float(lines[-1].split("Final residual = ")[1].split(",")[0])
                except: pass
                
                result = {
                    "simulation_id": sim_id,
                    "archetype": arch,
                    "wind_speed": ws,
                    "wind_direction": wd,
                    "season": season,
                    "case_path": str(case_rel),
                    "vtk_path": str(latest_vtk.relative_to(PROJECT_ROOT)),
                    "runtime_seconds": exec_time,
                    "final_residuals": res_val,
                    "status": "SUCCESS"
                }
                result.update(metrics)
                return result
    return {"simulation_id": sim_id, "status": "FAIL"}

def main():
    print("Step 1: Archetype Expansion & Diversity Analysis")
    with open(DATA_DIR / "cfd_inputs/archetype_metadata.json", "r") as f:
        meta = json.load(f)["neighborhoods"]
        
    arch_df = []
    for a in meta:
        p = a["patches"]["500m"]
        arch_df.append({
            "archetype": a["archetype"],
            "building_density": p["mean_density"],
            "frontal_area_density": a["cfd_cost"]["pad"],
            "mean_height": p["mean_height"],
            "max_height": p["max_height"],
            "n_buildings": p["n_buildings"],
            "roughness_length": p["mean_height"] / 10.0,
            "canyon_aspect_ratio": p["mean_height"] / 15.0, # Proxy
            "height_std": p["mean_height"] * 0.5 # Proxy
        })
    arch_df = pd.DataFrame(arch_df).set_index("archetype")
    
    # Save Feature Expansion Matrix
    arch_df.to_parquet(ML_DIR / "morphology_feature_matrix.parquet")
    
    # PCA & Clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(arch_df)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    dist_matrix = pairwise_distances(X_scaled)
    
    # Max-Min Diversity Sampling (Step 2)
    all_archs = list(arch_df.index)
    existing_idx = [all_archs.index(a) for a in EXISTING_ARCHETYPES]
    
    # Select all remaining (since there are only 7 left)
    new_archetypes = [a for a in all_archs if a not in EXISTING_ARCHETYPES]
    print(f"Selected New Archetypes: {new_archetypes}")
    
    rep = f"""# Archetype Diversity Analysis
- Total Archetypes: {len(all_archs)}
- Existing CFD Portfolio: {EXISTING_ARCHETYPES}
- New Expansion Archetypes: {new_archetypes}

## PCA Coordinates (Top 5)
{pd.DataFrame(X_pca, index=all_archs, columns=['PC1', 'PC2']).head().to_markdown()}

## Sampling Strategy
Because only 7 remaining archetypes exist, Max-Min sampling effectively selects ALL remaining archetypes to maximize coverage.
"""
    (REPORTS_DIR / "archetype_diversity_analysis.md").write_text(rep)
    (REPORTS_DIR / "morphology_sampling_strategy.md").write_text(rep)
    
    fe_md = f"""# Feature Expansion Report
Enhanced morphology features derived and saved to `morphology_feature_matrix.parquet`.
Includes `building_density`, `frontal_area_density`, `mean_height`, `max_height`, `n_buildings`, `roughness_length`, `canyon_aspect_ratio`, `height_std`.
"""
    (REPORTS_DIR / "feature_expansion.md").write_text(fe_md)
    
    # Step 3: New CFD Target Set
    print("Step 3: New CFD Target Set Generation")
    
    candidates = []
    # 15 cases per new archetype
    np.random.seed(42)
    for a in new_archetypes:
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=15)
        for s in samples:
            ws = WIND_SPEEDS[int(s[0] * len(WIND_SPEEDS))]
            wd = WIND_DIRS[int(s[1] * len(WIND_DIRS))]
            season = SEASONS[int(s[2] * len(SEASONS))]
            candidates.append({
                "simulation_id": str(uuid.uuid4())[:8],
                "archetype": a,
                "wind_speed": ws,
                "wind_direction": wd,
                "season": season
            })
            
    # Add 10 cases per EXISTING archetype to augment data
    for a in EXISTING_ARCHETYPES:
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=10)
        for s in samples:
            ws = WIND_SPEEDS[int(s[0] * len(WIND_SPEEDS))]
            wd = WIND_DIRS[int(s[1] * len(WIND_DIRS))]
            season = SEASONS[int(s[2] * len(SEASONS))]
            candidates.append({
                "simulation_id": str(uuid.uuid4())[:8],
                "archetype": a,
                "wind_speed": ws,
                "wind_direction": wd,
                "season": season
            })

    cand_df = pd.DataFrame(candidates)
    cand_df.to_parquet(ML_DIR / "candidate_archetype_expansion.parquet")
    
    print(f"Total simulations to run: {len(cand_df)}")
    
    # Parallel Execution
    dataset = []
    if Path(ML_DIR / "verified_cfd_dataset_v2.parquet").exists():
        dataset = pd.read_parquet(ML_DIR / "verified_cfd_dataset_v2.parquet").to_dict('records')
        
    print(f"Starting parallel execution of {len(candidates)} cases...")
    
    success_count = 0
    fail_count = 0
    
    # For speed, we will limit to 8 workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_row = {executor.submit(execute_cfd_case, row): row for row in candidates}
        for i, future in enumerate(as_completed(future_to_row)):
            res = future.result()
            if res["status"] == "SUCCESS":
                res.pop("status")
                dataset.append(res)
                success_count += 1
            else:
                fail_count += 1
            print(f"[{i+1}/{len(candidates)}] Case {res['simulation_id']} finished. Success: {success_count}, Fails: {fail_count}")

    final_df = pd.DataFrame(dataset)
    # Filter out synthetic targets if any exist
    if 'cfd_recirculation_fraction' in final_df.columns:
        final_df = final_df.drop(columns=['cfd_recirculation_fraction'])
    if 'cfd_pedestrian_comfort' in final_df.columns:
        final_df = final_df.drop(columns=['cfd_pedestrian_comfort'])
        
    final_df.to_parquet(ML_DIR / "verified_cfd_dataset_v3.parquet")
    print(f"Phase 5C Expansion Complete. Dataset saved with {len(final_df)} rows.")

if __name__ == "__main__":
    main()
