#!/usr/bin/env python3
"""
Phase 8L — Full CFD Field Dataset Reconstruction

Removes the `sample_cases = exists[:30]` truncation from Phase 7A
and rebuilds the field dataset using ALL available CFD simulations.
"""

import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import gc
import time
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ML_DIR  = PROJECT / "data" / "ml"
REPORTS = PROJECT / "reports"

POINTS_PER_CASE = 2000


def ws1_cfd_inventory(df):
    """Programmatically scan every VTK file."""
    exists, missing = [], []
    for _, row in df.iterrows():
        p = PROJECT / row['vtk_path']
        if p.exists():
            exists.append(row)
        else:
            missing.append(row)

    edf = pd.DataFrame(exists)
    mdf = pd.DataFrame(missing) if missing else pd.DataFrame()

    md  = "# Full CFD Inventory\n\n"
    md += f"- **Total Simulations in Dataset**: {len(df)}\n"
    md += f"- **Valid VTK Files Found**: {len(exists)}\n"
    md += f"- **Missing VTK Files**: {len(missing)}\n\n"

    md += "## Per-Archetype Breakdown\n\n"
    md += "| Archetype | Valid | Missing |\n|---|---|---|\n"
    all_archs = sorted(df['archetype'].unique())
    for a in all_archs:
        valid = len(edf[edf['archetype'] == a]) if len(edf) else 0
        miss  = len(mdf[mdf['archetype'] == a]) if len(mdf) else 0
        md += f"| {a} | {valid} | {miss} |\n"

    md += f"\n**Previously Used**: 30 (archetypes 0, 1 only)\n"
    md += f"**Now Available**: {len(exists)} (archetypes {sorted(edf['archetype'].unique())})\n"
    md += f"**Previously Discarded**: {len(exists) - 30}\n"

    (REPORTS / "full_cfd_inventory.md").write_text(md)
    print(f"[WS1] Inventory: {len(exists)} valid, {len(missing)} missing")
    return exists


def ws2_full_extraction(exists_rows):
    """Extract field data from EVERY valid CFD simulation."""
    all_points = []
    errors = []

    for i, row in enumerate(exists_rows):
        p = PROJECT / row['vtk_path']
        try:
            mesh = pv.read(str(p))
            pts = mesh.points

            if len(pts) > POINTS_PER_CASE:
                idx = np.random.RandomState(42).choice(len(pts), POINTS_PER_CASE, replace=False)
            else:
                idx = np.arange(len(pts))

            pts = pts[idx]
            pd_data = mesh.point_data

            u, v, w = (pd_data['U'][idx].T if 'U' in pd_data
                       else (np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx))))
            p_val = pd_data['p'][idx] if 'p' in pd_data else np.zeros(len(idx))
            k_val = pd_data['k'][idx] if 'k' in pd_data else np.zeros(len(idx))

            case_df = pd.DataFrame({
                'simulation_id': row['simulation_id'],
                'archetype': row['archetype'],
                'x': pts[:, 0], 'y': pts[:, 1], 'z': pts[:, 2],
                'u': u, 'v': v, 'w': w,
                'p': p_val, 'k': k_val,
            })
            all_points.append(case_df)

            if (i + 1) % 20 == 0:
                print(f"  Extracted {i+1}/{len(exists_rows)} simulations ...")
                gc.collect()

        except Exception as e:
            errors.append((row['simulation_id'], str(e)))
            print(f"  WARN: sim {row['simulation_id']} failed: {e}")

    final_df = pd.concat(all_points, ignore_index=True)
    out_path = ML_DIR / "cfd_field_dataset_full.parquet"
    final_df.to_parquet(out_path)

    print(f"[WS2] Extracted {len(final_df):,} points from {len(all_points)} simulations → {out_path.name}")
    if errors:
        print(f"  {len(errors)} extraction errors")

    return final_df, errors


def ws3_diversity(final_df):
    """Validate archetype diversity."""
    arch_counts = final_df.groupby('archetype')['simulation_id'].nunique()
    total_archs = len(arch_counts)

    md  = "# Field Diversity Validation\n\n"
    md += f"- **Total Points**: {len(final_df):,}\n"
    md += f"- **Total Simulations**: {final_df['simulation_id'].nunique()}\n"
    md += f"- **Total Archetypes**: {total_archs}\n\n"
    md += "| Archetype | Simulations | Points |\n|---|---|---|\n"
    for a in sorted(final_df['archetype'].unique()):
        sub = final_df[final_df['archetype'] == a]
        md += f"| {a} | {sub['simulation_id'].nunique()} | {len(sub):,} |\n"

    if total_archs >= 8:
        md += f"\n**PASS**: {total_archs} archetypes ≥ 8 minimum\n"
    else:
        md += f"\n**FAIL**: {total_archs} archetypes < 8 minimum\n"

    (REPORTS / "field_diversity_validation.md").write_text(md)
    return total_archs


def ws4_morphology_coverage(final_df, cdf):
    """Per-archetype physical distributions."""
    merged = final_df.merge(cdf[['simulation_id', 'wind_speed']], on='simulation_id')

    md = "# Morphology Coverage\n\n"
    md += "| Archetype | Mean |u| | Wake % | Mean p | Mean k |\n|---|---|---|---|---|\n"

    for a in sorted(final_df['archetype'].unique()):
        sub = merged[merged['archetype'] == a]
        speed = np.sqrt(sub['u']**2 + sub['v']**2 + sub['w']**2)
        wake_pct = (speed < 0.3 * sub['wind_speed']).mean() * 100
        md += f"| {a} | {speed.mean():.2f} | {wake_pct:.1f}% | {sub['p'].mean():.2f} | {sub['k'].mean():.4f} |\n"

    (REPORTS / "morphology_coverage.md").write_text(md)


def ws5_loao_feasibility(total_archs):
    """Determine LOAO viability."""
    if total_archs >= 8:
        grade, label = "A", "Scientifically Valid"
    elif total_archs >= 5:
        grade, label = "B", "Marginally Valid"
    else:
        grade, label = "C", "Insufficient"

    md  = "# LOAO Feasibility\n\n"
    md += f"- **Archetypes Available**: {total_archs}\n"
    md += f"- **Minimum Training Set (N-1)**: {total_archs - 1}\n\n"
    md += f"**LOAO Viability: {grade} ({label})**\n"
    (REPORTS / "loao_feasibility.md").write_text(md)
    return grade


def main():
    print("Phase 8L — Full CFD Field Dataset Reconstruction")
    t0 = time.time()

    cdf = pd.read_parquet(ML_DIR / "verified_cfd_dataset_v3.parquet")

    # WS1
    exists_rows = ws1_cfd_inventory(cdf)

    # WS2
    final_df, errors = ws2_full_extraction(exists_rows)

    # WS3
    total_archs = ws3_diversity(final_df)

    # WS4
    ws4_morphology_coverage(final_df, cdf)

    # WS5
    loao_grade = ws5_loao_feasibility(total_archs)

    # Final Certification
    md  = "# Phase 8L Dataset Reconstruction\n\n"
    md += f"## Answers\n\n"
    md += f"1. **How many CFD simulations exist?** {len(exists_rows)}\n"
    md += f"2. **How many were previously discarded?** {len(exists_rows) - 30}\n"
    md += f"3. **How many archetypes now exist?** {total_archs}\n"
    md += f"4. **Is LOAO scientifically valid?** Grade {loao_grade}\n\n"
    md += f"## Dataset\n\n"
    md += f"- Output: `data/ml/cfd_field_dataset_full.parquet`\n"
    md += f"- Points: {len(final_df):,}\n"
    md += f"- Simulations: {final_df['simulation_id'].nunique()}\n"
    md += f"- Archetypes: {total_archs}\n"
    md += f"- Extraction Time: {time.time() - t0:.1f}s\n"
    if errors:
        md += f"- Extraction Errors: {len(errors)}\n"
    (REPORTS / "phase8l_dataset_reconstruction.md").write_text(md)

    print(f"\nPhase 8L Complete. {total_archs} archetypes, {len(final_df):,} points.")


if __name__ == "__main__":
    main()
