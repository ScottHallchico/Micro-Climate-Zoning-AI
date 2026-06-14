#!/usr/bin/env python3
import sys, json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import DATA_DIR, REPORTS_DIR

CFD_CASES = DATA_DIR / "cfd_cases"
CFD_OUTPUTS = DATA_DIR / "cfd_outputs"
CFD_CONDITIONS = DATA_DIR / "cfd_conditions"
CFD_GEOMETRY = DATA_DIR / "cfd_geometry"
ML_DIR = DATA_DIR / "ml"
CFD_REPORTS = REPORTS_DIR

def main():
    ML_DIR.mkdir(parents=True, exist_ok=True)
    CFD_REPORTS.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    for arch_dir in CFD_OUTPUTS.iterdir():
        if not arch_dir.is_dir():
            continue
            
        arch_id = arch_dir.name.split("_")[1]
        
        geom_meta_path = CFD_GEOMETRY / arch_dir.name / "geometry_meta.json"
        geom_meta = {}
        if geom_meta_path.exists():
            with open(geom_meta_path) as f:
                geom_meta = json.load(f)
                
        for scen_dir in arch_dir.iterdir():
            if not scen_dir.is_dir():
                continue
                
            metrics_path = scen_dir / "metrics.json"
            if not metrics_path.exists():
                continue
                
            with open(metrics_path) as f:
                metrics = json.load(f)
                
            cond_path = CFD_CONDITIONS / arch_dir.name / f"{scen_dir.name}.json"
            cond_meta = {}
            if cond_path.exists():
                with open(cond_path) as f:
                    cond_meta = json.load(f)
                    
            row = {
                "Archetype": int(arch_id),
                "Scenario": scen_dir.name,
                "n_buildings": geom_meta.get("n_buildings", 0),
                "n_trees": geom_meta.get("n_trees", 0),
                "max_building_height": geom_meta.get("max_building_height", 0.0),
                "mean_building_height": geom_meta.get("mean_building_height", 0.0),
                "wind_speed_ms": cond_meta.get("wind_speed_ms", 0.0),
                "wind_direction_deg": cond_meta.get("wind_direction_deg", 0.0),
                "temperature_k": cond_meta.get("temperature_k", 293.15),
            }
            
            row.update(metrics)
            rows.append(row)
            
    if not rows:
        print("No outputs found. Exiting.")
        return
        
    df = pd.DataFrame(rows)
    out_file = ML_DIR / "cfd_training_dataset.parquet"
    df.to_parquet(out_file)
    print(f"Exported ML dataset to {out_file} with {len(df)} records.")
    
    # Generate CFD Dataset Summary Report
    with open(CFD_REPORTS / "cfd_dataset_summary.md", "w") as f:
        f.write("# CFD Training Dataset Summary\n\n")
        f.write(f"**Total Records**: {len(df)}\n")
        f.write(f"**Archetypes**: {df['Archetype'].nunique()}\n")
        f.write(f"**Scenarios per Archetype**: {df.groupby('Archetype')['Scenario'].count().mean():.1f}\n\n")
        
        f.write("## Target Variable Distributions\n\n")
        targets = ["Mean_Velocity_Magnitude", "Wind_Amplification_Factor", "Ventilation_Efficiency", 
                   "Mean_TKE", "Pedestrian_Wind_Comfort_Index", "Urban_Heat_Mitigation_Proxy"]
        
        f.write("| Target | Mean | Min | Max | Std |\n")
        f.write("|--------|------|-----|-----|-----|\n")
        for t in targets:
            if t in df.columns:
                f.write(f"| {t} | {df[t].mean():.2f} | {df[t].min():.2f} | {df[t].max():.2f} | {df[t].std():.2f} |\n")
                
        f.write("\n## Recommended PINN and Surrogate Modeling Strategy\n\n")
        f.write("### Recommended Input Features (Surrogate)\n")
        f.write("- **Morphology**: `n_buildings`, `max_building_height`, `mean_building_height`, `n_trees`\n")
        f.write("- **Meteorology**: `wind_speed_ms`, `wind_direction_deg`, `temperature_k`\n\n")
        f.write("### Recommended Target Variables (PINN / Surrogate)\n")
        f.write("- **Velocity**: `Mean_Velocity_Magnitude`, `Wind_Amplification_Factor`\n")
        f.write("- **Turbulence**: `Mean_TKE`\n")
        f.write("- **Urban Climate KPIs**: `Pedestrian_Wind_Comfort_Index`, `Ventilation_Efficiency`\n")

if __name__ == "__main__":
    main()
