#!/usr/bin/env python3
"""
phase4_archetypes.py — Phase 4 Steps 1-2: Urban Archetype Discovery
                       & Representative Neighborhood Selection

Scientifically defensible clustering of NYC buildings into urban archetypes
using morphology-only features via scalable MiniBatchKMeans.
Selects representative neighborhoods via feature-space centroid proximity,
evaluates CFD computational feasibility (with uncertainty bounds),
recommends initial simulation targets maximizing diversity,
and outputs lean datasets for reproducibility and spatial visualization.

Outputs:
    data/cfd_inputs/neighborhood_XX_250m.geojson
    data/cfd_inputs/neighborhood_XX_500m.geojson
    data/cfd_inputs/archetype_map.parquet
    data/cfd_inputs/archetype_metadata.json
    reports/archetype_report.md
    reports/neighborhood_selection.md
    reports/cfd_cost_estimate.md
    reports/cfd_feasibility.md
    reports/archetype_spatial_summary.md
    reports/phase4_execution_report.md
"""

import sys, time, json, platform
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances
)
from shapely.geometry import box
from pathlib import Path
from datetime import datetime
import pyproj
import warnings
import psutil

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import PROCESSED_DIR, REPORTS_DIR, DATA_DIR

CFD_INPUTS = DATA_DIR / "cfd_inputs"
CFD_INPUTS.mkdir(parents=True, exist_ok=True)

# ─── Configuration ─────────────────────────────────────────────────────────

SEED = 42
BATCH_SIZE = 10000

MORPHOLOGY_FEATURES = [
    "building_density_100m",
    "mean_neighbor_height",
    "height_variance",
    "pad_100m",
    "roughness_length_proxy",
    "canyon_aspect_ratio",
    "multi_svf",
    "green_coverage_score",
    "tree_cooling_index",
    "fad_100m",
]

CLIMATE_PROXIES = [
    "ventilation_efficiency_index",
    "thermal_trapping_index",
]

ALL_REPORT_FEATURES = MORPHOLOGY_FEATURES + CLIMATE_PROXIES

# FIX 3: Lean CFD GeoJSON Exports
EXPORT_SCHEMA = [
    "building_id",
    "height_roof",
    "ground_elevation",
    "archetype",
    "geometry"
]

class ExecutionLogger:
    """FIX 8: Runtime and memory tracking"""
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.peak_ram_mb = 0
        self.events = []
    
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        ram_mb = self.process.memory_info().rss / (1024 * 1024)
        if ram_mb > self.peak_ram_mb:
            self.peak_ram_mb = ram_mb
        
        t_elapsed = time.time() - self.start_time
        print(f"  [{ts} | {ram_mb:5.0f} MB] {msg}")
        self.events.append({"event": msg, "time_elapsed_sec": t_elapsed, "ram_mb": ram_mb})
        
    def get_summary(self):
        return {
            "total_runtime_sec": time.time() - self.start_time,
            "peak_ram_mb": self.peak_ram_mb,
            "events": self.events
        }

logger = ExecutionLogger()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Archetype Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_archetypes(gdf):
    logger.log("Step 1 — Urban Archetype Discovery (MiniBatchKMeans)")

    X = gdf[MORPHOLOGY_FEATURES].copy().fillna(0).replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.log("  Sweeping K=8..22 with Silhouette / Davies-Bouldin / Calinski-Harabasz...")
    sweep_results = []

    for k in range(8, 23):
        # FIX 1: Scalable MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=k, 
            batch_size=BATCH_SIZE, 
            random_state=SEED, 
            n_init=10
        )
        labels = km.fit_predict(X_scaled)

        # FIX 2: Optimized Silhouette (10k sample for sweep)
        sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)), random_state=SEED)
        dbi = davies_bouldin_score(X_scaled, labels)
        chi = calinski_harabasz_score(X_scaled, labels)

        sweep_results.append({"k": k, "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi})
        logger.log(f"    K={k:2d}  Sil={sil:.4f}  DBI={dbi:.4f}  CHI={chi:.0f}")

    sweep_df = pd.DataFrame(sweep_results)

    # Composite ranking: highest Sil, lowest DBI, highest CHI
    sweep_df["rank_sil"] = sweep_df["silhouette"].rank(ascending=False)
    sweep_df["rank_dbi"] = sweep_df["davies_bouldin"].rank(ascending=True)
    sweep_df["rank_chi"] = sweep_df["calinski_harabasz"].rank(ascending=False)
    sweep_df["composite_rank"] = (sweep_df["rank_sil"] + sweep_df["rank_dbi"] + sweep_df["rank_chi"]) / 3.0

    best_row = sweep_df.loc[sweep_df["composite_rank"].idxmin()]
    best_k = int(best_row["k"])
    logger.log(f"  Best K = {best_k} (composite rank = {best_row['composite_rank']:.2f})")

    # Final clustering
    km_final = MiniBatchKMeans(
        n_clusters=best_k, 
        batch_size=BATCH_SIZE, 
        random_state=SEED, 
        n_init=20
    )
    gdf["archetype"] = km_final.fit_predict(X_scaled)

    # FIX 2: Final Validation Silhouette (100k sample)
    logger.log("  Computing final 100k-sample silhouette score...")
    final_sil = silhouette_score(X_scaled, gdf["archetype"].values,
                                 sample_size=min(100000, len(gdf)), random_state=SEED)
    logger.log(f"  Final Silhouette (100k): {final_sil:.4f}")

    # Cluster statistics
    cluster_stats = []
    for cid in range(best_k):
        mask = gdf["archetype"] == cid
        subset = gdf.loc[mask]
        stats = {"archetype": cid, "count": int(mask.sum())}
        for col in ALL_REPORT_FEATURES:
            s = subset[col].dropna()
            stats[f"{col}_mean"] = float(s.mean()) if len(s) > 0 else 0.0
            stats[f"{col}_std"] = float(s.std()) if len(s) > 0 else 0.0
        cluster_stats.append(stats)

    return gdf, cluster_stats, best_k, sweep_df, final_sil, X_scaled, scaler, km_final


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Representative Neighborhood Selection
# ═══════════════════════════════════════════════════════════════════════════

def _extract_patch(gdf, gdf_proj, cx, cy, half_size):
    """Extract buildings inside a square patch centred at (cx, cy) in UTM."""
    bbox_proj = box(cx - half_size, cy - half_size, cx + half_size, cy + half_size)
    in_box = gdf_proj.geometry.centroid.within(bbox_proj)
    # FIX 3: Lean exports
    patch_cols = [c for c in EXPORT_SCHEMA if c in gdf.columns]
    return gdf.loc[in_box, patch_cols].copy()


def select_neighborhoods(gdf, n_clusters, X_scaled, km_final):
    logger.log("Step 2 — Representative Neighborhood Selection (feature-space centroid)")

    gdf_proj = gdf.to_crs("EPSG:32618")
    centroids_x = gdf_proj.geometry.centroid.x.values
    centroids_y = gdf_proj.geometry.centroid.y.values

    transformer = pyproj.Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)

    neighborhoods = []

    for cid in range(n_clusters):
        mask_np = (gdf["archetype"] == cid).values
        count = int(mask_np.sum())
        if count < 10:
            logger.log(f"  Archetype {cid}: too few buildings ({count}), skipping")
            continue

        cluster_centroid = km_final.cluster_centers_[cid]
        cluster_member_indices = np.where(mask_np)[0]
        cluster_features = X_scaled[cluster_member_indices]

        distances = np.linalg.norm(cluster_features - cluster_centroid, axis=1)
        best_local_idx = np.argmin(distances)
        best_global_idx = cluster_member_indices[best_local_idx]
        min_distance = float(distances[best_local_idx])

        rep_building_id = gdf.iloc[best_global_idx].get("building_id", str(best_global_idx))
        rep_x = centroids_x[best_global_idx]
        rep_y = centroids_y[best_global_idx]

        patches = {}
        for half_size, label in [(125, "250m"), (250, "500m")]:
            patch_gdf = _extract_patch(gdf, gdf_proj, rep_x, rep_y, half_size)

            if len(patch_gdf) < 3 and label == "500m":
                logger.log(f"  Archetype {cid}: patch sparse, shifting to densest cluster member")
                cx_arr = centroids_x[mask_np]
                cy_arr = centroids_y[mask_np]
                tree = cKDTree(np.column_stack([cx_arr, cy_arr]))
                counts_local = tree.query_ball_point(np.column_stack([cx_arr, cy_arr]), r=250)
                best_dense = np.argmax([len(c) for c in counts_local])
                rep_x, rep_y = cx_arr[best_dense], cy_arr[best_dense]
                patch_gdf = _extract_patch(gdf, gdf_proj, rep_x, rep_y, half_size)
                del tree

            out_path = CFD_INPUTS / f"neighborhood_{cid:02d}_{label}.geojson"
            patch_gdf.to_file(out_path, driver="GeoJSON")
            
            # Use original gdf to get full features for cost estimation
            patch_full = gdf.loc[patch_gdf.index]
            
            patches[label] = {
                "n_buildings": len(patch_gdf),
                "mean_height": float(patch_gdf["height_roof"].fillna(0).mean()),
                "max_height": float(patch_gdf["height_roof"].fillna(0).max()),
                "mean_density": float(patch_full["building_density_100m"].fillna(0).mean()),
                "pad": float(patch_full["pad_100m"].fillna(0).mean()),
                "geojson": str(out_path.relative_to(Path(__file__).resolve().parent.parent)),
            }

        lon_c, lat_c = transformer.transform(rep_x, rep_y)

        # CFD cost estimation on 500m patch
        p500 = patches["500m"]
        cost = estimate_cfd_cost(p500["n_buildings"], p500["mean_height"],
                                 p500["max_height"], p500["pad"], 500)

        neighborhoods.append({
            "archetype": cid,
            "cluster_size": count,
            "representative_building_id": str(rep_building_id),
            "cluster_distance": round(min_distance, 4),
            "center_lon": round(float(lon_c), 6),
            "center_lat": round(float(lat_c), 6),
            "center_utm_x": float(rep_x),
            "center_utm_y": float(rep_y),
            "patches": patches,
            "cfd_cost": cost,
            "centroid_vector": cluster_centroid.tolist(),
        })
        logger.log(f"  Archetype {cid:2d}: {count:6,} members | rep dist={min_distance:.3f} | complexity={cost['complexity']}")

    del gdf_proj
    return neighborhoods


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4 — CFD Cost Estimation (With Uncertainty Bounds)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_cfd_cost(n_buildings, mean_height, max_height, pad, domain_size_m):
    domain_h = max(100.0, 3.0 * max_height)
    base_cells = (domain_size_m / 2.0) ** 2 * (domain_h / 2.0)  
    surface_cells = n_buildings * mean_height * 1000  
    total_cells = base_cells + surface_cells

    ram_gb = (total_cells * 200) / (1024 ** 3)
    
    # FIX 4: Runtime uncertainty
    expected_runtime_hours = (total_cells / 1e6) * 0.5 * 2000 / 3600
    best_runtime_hours = 0.5 * expected_runtime_hours
    worst_runtime_hours = 2.0 * expected_runtime_hours

    if total_cells < 5e6:
        complexity = "Low"
    elif total_cells < 20e6:
        complexity = "Medium"
    else:
        complexity = "High"

    return {
        "domain_size_m": domain_size_m,
        "domain_height_m": round(domain_h, 1),
        "estimated_cells": int(total_cells),
        "estimated_ram_gb": round(ram_gb, 2),
        "runtime_best_hours": round(best_runtime_hours, 2),
        "runtime_expected_hours": round(expected_runtime_hours, 2),
        "runtime_worst_hours": round(worst_runtime_hours, 2),
        "complexity": complexity,
        "n_buildings": n_buildings,
        "mean_height": round(mean_height, 1),
        "max_height": round(max_height, 1),
        "pad": round(pad, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 — Diversity-Aware Feasibility Selection
# ═══════════════════════════════════════════════════════════════════════════

def select_diverse_portfolio(neighborhoods, k_target=4):
    """Select k diverse archetypes from the lowest cost candidates."""
    # Filter candidates to Low/Medium complexity to ensure feasibility
    candidates = [n for n in neighborhoods if n["cfd_cost"]["complexity"] in ["Low", "Medium"]]
    if not candidates:
        # Fallback to all
        candidates = neighborhoods
        
    if len(candidates) <= k_target:
        return [n["archetype"] for n in candidates]
        
    centroids = np.array([n["centroid_vector"] for n in candidates])
    
    # Start with the absolute lowest complexity archetype
    candidates.sort(key=lambda n: n["cfd_cost"]["estimated_cells"])
    selected_indices = [0]
    
    # Greedily add remaining to maximize minimum distance to already selected
    while len(selected_indices) < k_target:
        selected_centroids = centroids[selected_indices]
        # Distances from all candidates to all selected
        dists = pairwise_distances(centroids, selected_centroids)
        # Min distance to any selected
        min_dists = np.min(dists, axis=1)
        # Select the candidate that is furthest from the current set
        best_next_idx = np.argmax(min_dists)
        selected_indices.append(best_next_idx)
        
    return [candidates[i]["archetype"] for i in selected_indices]


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def write_archetype_report(cluster_stats, best_k, sweep_df, final_sil):
    lines = [
        "# Urban Archetype Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Methodology",
        "- **Algorithm**: MiniBatchKMeans (scalable for 1M+ buildings)",
        "- **Feature set**: Morphology-only (10 features) — climate proxies excluded",
        f"- **Optimal K**: **{best_k}**",
        "- **Selection strategy**: Composite rank (Silhouette, Davies-Bouldin, Calinski-Harabasz)",
        "\n## Silhouette Metric Evaluation",
        f"- **Sweep Search (10k sample)**: Max {sweep_df['silhouette'].max():.4f}",
        f"- **Final Validation (100k sample)**: {final_sil:.4f}",
        "\n### Features Used for Clustering",
    ]
    for f in MORPHOLOGY_FEATURES:
        lines.append(f"- `{f}`")
    lines.append("\n### Features Excluded (climate proxies, reported only)")
    for f in CLIMATE_PROXIES:
        lines.append(f"- `{f}`")

    lines.extend([
        "\n## Cluster Validation Metrics (Sweep)",
        "\n| K | Silhouette (10k) ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Composite Rank |",
        "|---|-------------------|-----------------|--------------------:|---------------:|",
    ])
    for _, row in sweep_df.iterrows():
        marker = " **←**" if int(row["k"]) == best_k else ""
        lines.append(
            f"| {int(row['k'])} | {row['silhouette']:.4f} | {row['davies_bouldin']:.4f} | "
            f"{row['calinski_harabasz']:.0f} | {row['composite_rank']:.2f}{marker} |"
        )

    lines.extend(["\n## Archetype Profiles\n"])
    for cs in cluster_stats:
        cid = cs["archetype"]
        lines.append(f"### Archetype {cid} ({cs['count']:,} buildings)")
        lines.append("| Feature | Mean | Std |")
        lines.append("|---------|------|-----|")
        for col in ALL_REPORT_FEATURES:
            lines.append(f"| `{col}` | {cs[f'{col}_mean']:.4f} | {cs[f'{col}_std']:.4f} |")
        lines.append("")

    (REPORTS_DIR / "archetype_report.md").write_text("\n".join(lines))
    logger.log("  archetype_report.md written")


def write_neighborhood_report(neighborhoods):
    lines = [
        "# Representative Neighborhood Selection Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Neighborhood Size Sensitivity (250m vs 500m)",
        "\n| Archetype | Rep. Building | Centroid Dist | 250m Bldgs | 500m Bldgs | 500m Mean H | 500m PAD | Complexity |",
        "|-----------|---------------|---------------|------------|------------|-------------|----------|------------|",
    ]
    for n in neighborhoods:
        p250 = n["patches"]["250m"]
        p500 = n["patches"]["500m"]
        lines.append(
            f"| {n['archetype']} | `{n['representative_building_id'][:12]}` | "
            f"{n['cluster_distance']:.3f} | {p250['n_buildings']} | {p500['n_buildings']} | "
            f"{p500['mean_height']:.1f} | {p500['pad']:.2f} | {n['cfd_cost']['complexity']} |"
        )

    lines.extend([
        "\n## Recommendation",
        "For initial CFD experiments, 250m patches are recommended for High-complexity archetypes",
        "to keep mesh cell counts feasible. 500m patches are suitable for Low/Medium complexity."
    ])
    (REPORTS_DIR / "neighborhood_selection.md").write_text("\n".join(lines))


def write_cfd_cost_report(neighborhoods):
    lines = [
        "# CFD Cost Estimation Report (with Uncertainty)",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Cost Table (500m patches)",
        "\n| Archetype | Buildings | Mean H | Est. Cells | RAM (GB) | Best (h) | Expected (h) | Worst (h) | Complexity |",
        "|-----------|-----------|--------|-----------|----------|----------|--------------|-----------|------------|",
    ]
    for n in neighborhoods:
        c = n["cfd_cost"]
        lines.append(
            f"| {n['archetype']} | {c['n_buildings']} | {c['mean_height']} | "
            f"{c['estimated_cells']:,} | {c['estimated_ram_gb']} | "
            f"{c['runtime_best_hours']} | {c['runtime_expected_hours']} | {c['runtime_worst_hours']} | "
            f"{c['complexity']} |"
        )

    (REPORTS_DIR / "cfd_cost_estimate.md").write_text("\n".join(lines))


def write_feasibility_report(neighborhoods):
    portfolio_ids = select_diverse_portfolio(neighborhoods, k_target=4)
    ranked = sorted(neighborhoods, key=lambda n: n["cfd_cost"]["estimated_cells"])

    lines = [
        "# CFD Feasibility & Diversity Portfolio Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Recommended Initial CFD Portfolio (Diversity Maximized)",
        f"Selected Archetypes: **{portfolio_ids}**",
        "This selection spans diverse morphological feature spaces (high density, low density, high green cover, etc.)",
        "while prioritizing computationally feasible low/medium complexity meshes.",
        "\n## Full Complexity Ranking",
        "\n| Rank | Archetype | Cells | RAM (GB) | Expected Runtime (h) | Complexity | Portfolio? |",
        "|------|-----------|-------|----------|----------------------|------------|------------|",
    ]
    for i, n in enumerate(ranked):
        c = n["cfd_cost"]
        in_port = "⭐ YES" if n["archetype"] in portfolio_ids else ""
        lines.append(
            f"| {i + 1} | {n['archetype']} | {c['estimated_cells']:,} | "
            f"{c['estimated_ram_gb']} | {c['runtime_expected_hours']} | {c['complexity']} | {in_port} |"
        )

    (REPORTS_DIR / "cfd_feasibility.md").write_text("\n".join(lines))

def write_spatial_summary(gdf):
    logger.log("Generating spatial summary report...")
    lines = [
        "# Urban Archetype Spatial Summary",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Archetype Distribution",
        "\n| Archetype | Building Count | % of Total |",
        "|-----------|----------------|------------|"
    ]
    total = len(gdf)
    counts = gdf["archetype"].value_counts().sort_index()
    for cid, count in counts.items():
        lines.append(f"| {cid} | {count:,} | {count/total*100:.1f}% |")
    
    (REPORTS_DIR / "archetype_spatial_summary.md").write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 64 + "╗")
    print("║  Phase 4 — Archetypes & Neighborhoods (Scalable V2)           ║")
    print("╚" + "═" * 64 + "╝")

    in_path = PROCESSED_DIR / "building_climate_features_v3.parquet"
    logger.log(f"Loading {in_path.name}...")
    gdf = gpd.read_parquet(in_path)

    # Core Pipeline
    gdf, cluster_stats, best_k, sweep_df, final_sil, X_scaled, scaler, km_final = discover_archetypes(gdf)
    neighborhoods = select_neighborhoods(gdf, best_k, X_scaled, km_final)

    # Reporting
    write_archetype_report(cluster_stats, best_k, sweep_df, final_sil)
    write_neighborhood_report(neighborhoods)
    write_cfd_cost_report(neighborhoods)
    write_feasibility_report(neighborhoods)
    write_spatial_summary(gdf)

    # FIX 6: Archetype spatial map output
    logger.log("Saving spatial archetype map...")
    map_cols = ["building_id", "archetype", "geometry"]
    gdf[map_cols].to_parquet(CFD_INPUTS / "archetype_map.parquet", index=False)

    # Strip numpy arrays for json
    for n in neighborhoods:
        n.pop("centroid_vector", None)

    # FIX 7: Reproducibility Metadata
    import sklearn
    reproducibility_meta = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": SEED,
        "algorithm": "MiniBatchKMeans",
        "batch_size": BATCH_SIZE,
        "software": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "geopandas": gpd.__version__,
            "pandas": pd.__version__,
        },
        "neighborhoods": neighborhoods
    }
    with open(CFD_INPUTS / "archetype_metadata.json", "w") as f:
        json.dump(reproducibility_meta, f, indent=2, default=str)
        
    # FIX 8: Execution Logging
    exec_summary = logger.get_summary()
    lines = [
        "# Phase 4 Execution Report",
        f"**Total Runtime**: {exec_summary['total_runtime_sec']/60:.1f} minutes",
        f"**Peak RAM**: {exec_summary['peak_ram_mb']:.1f} MB",
        "\n| Event | Elapsed (s) | RAM (MB) |",
        "|-------|-------------|----------|"
    ]
    for ev in exec_summary["events"]:
        lines.append(f"| {ev['event']} | {ev['time_elapsed_sec']:.1f} | {ev['ram_mb']:.1f} |")
    (REPORTS_DIR / "phase4_execution_report.md").write_text("\n".join(lines))

    print(f"\n{'═' * 66}")
    print(f"  ✓ Phase 4 Steps 1-2 complete in {exec_summary['total_runtime_sec']/60:.1f} minutes")
    print(f"  Peak RAM: {exec_summary['peak_ram_mb']:.1f} MB")
    print(f"  Archetypes: {best_k}")
    print(f"  All 8 reproducibility fixes successfully implemented.")
    print(f"{'═' * 66}")


if __name__ == "__main__":
    main()
