#!/usr/bin/env python3
"""
feature_engineering.py — Phase 3: Urban Climate Feature Engineering (V3)

Generates advanced, physically meaningful urban-climate features from the ETL output
(building_master.parquet) to serve as inputs to CFD simulations and PINN training.
Redesigned for memory scalability using chunked spatial queries and strict self-exclusion.

Outputs:
    - data/processed/building_climate_features_v3.parquet
    - reports/climate_feature_report.md
    - reports/urban_climate_physics_v2.md
"""

import sys
import time
import resource
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import PROCESSED_DIR, REPORTS_DIR, RAW_DIR

def _get_ram_mb():
    """Returns the peak memory usage of the current process in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    mem = _get_ram_mb()
    print(f"  [{ts} | {mem:5.0f} MB] {msg}")

def normalize(series):
    """Min-Max normalization to 0-1."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return np.zeros_like(series, dtype=np.float32)
    return ((series - s_min) / (s_max - s_min)).astype(np.float32)

def normalize_100(series):
    """Min-Max normalization to 0-100."""
    return (normalize(series) * 100.0).astype(np.float32)

def compute_morphology_and_terrain(gdf):
    _log("Computing Advanced Morphology & Terrain Features (v3)...")
    
    gdf_proj = gdf.to_crs("EPSG:32618")
    
    bldg_areas = gdf_proj.geometry.area.values.astype(np.float32)
    # Approx width facing wind isotropic proxy
    bldg_widths = np.sqrt(bldg_areas)
    
    heights = gdf["height_roof"].fillna(0).values.astype(np.float32)
    elevations = pd.to_numeric(gdf["ground_elevation"], errors="coerce").fillna(0).values.astype(np.float32)
    fars = pd.to_numeric(gdf["far"], errors="coerce").fillna(0).values.astype(np.float32)
    
    centroids = np.column_stack([gdf_proj.geometry.centroid.x, gdf_proj.geometry.centroid.y])
    n_points = len(centroids)
    
    _log(f"Building KD-Tree for {n_points:,} points...")
    kd_tree = cKDTree(centroids)
    
    chunk_size = 25000
    
    density_100m = np.zeros(n_points, dtype=np.float32)
    density_250m = np.zeros(n_points, dtype=np.float32)
    
    pad_100m = np.zeros(n_points, dtype=np.float32)
    pad_250m = np.zeros(n_points, dtype=np.float32)
    fad_100m = np.zeros(n_points, dtype=np.float32)
    
    mean_neighbor_height = np.zeros(n_points, dtype=np.float32)
    height_variance = np.zeros(n_points, dtype=np.float32)
    local_far_density = np.zeros(n_points, dtype=np.float32)
    
    local_mean_elevation = np.zeros(n_points, dtype=np.float32)
    local_elevation_variance = np.zeros(n_points, dtype=np.float32)
    local_slope_proxy = np.zeros(n_points, dtype=np.float32)
    
    local_max_height = np.zeros(n_points, dtype=np.float32)
    local_min_height = np.zeros(n_points, dtype=np.float32)
    local_median_height = np.zeros(n_points, dtype=np.float32)
    local_height_p90 = np.zeros(n_points, dtype=np.float32)
    local_height_p95 = np.zeros(n_points, dtype=np.float32)
    height_above_local_mean = np.zeros(n_points, dtype=np.float32)
    height_above_local_p90 = np.zeros(n_points, dtype=np.float32)
    
    multi_svf = np.zeros(n_points, dtype=np.float32)
    avg_nn_dist = np.zeros(n_points, dtype=np.float32)
    
    area_100m_circle = np.pi * (100**2)
    area_250m_circle = np.pi * (250**2)
    
    for i in range(0, n_points, chunk_size):
        end = min(i + chunk_size, n_points)
        chunk_centroids = centroids[i:end]
        
        # 1. 100m Radius Queries
        idxs_100m = kd_tree.query_ball_point(chunk_centroids, r=100, workers=-1)
        for j, neighbors in enumerate(idxs_100m):
            global_idx = i + j
            neighbors_arr = np.array(neighbors, dtype=np.int32)
            neighbors_arr = neighbors_arr[neighbors_arr != global_idx]
            
            density_100m[global_idx] = len(neighbors_arr)
            if len(neighbors_arr) > 0:
                # Heights
                neighbor_heights = heights[neighbors_arr]
                mean_h = np.mean(neighbor_heights)
                p90_h = np.percentile(neighbor_heights, 90)
                
                mean_neighbor_height[global_idx] = mean_h
                height_variance[global_idx] = np.std(neighbor_heights)
                
                local_max_height[global_idx] = np.max(neighbor_heights)
                local_min_height[global_idx] = np.min(neighbor_heights)
                local_median_height[global_idx] = np.median(neighbor_heights)
                local_height_p90[global_idx] = p90_h
                local_height_p95[global_idx] = np.percentile(neighbor_heights, 95)
                
                height_above_local_mean[global_idx] = heights[global_idx] - mean_h
                height_above_local_p90[global_idx] = heights[global_idx] - p90_h
                
                # Elevations
                neighbor_elevs = elevations[neighbors_arr]
                local_mean_elevation[global_idx] = np.mean(neighbor_elevs)
                local_elevation_variance[global_idx] = np.std(neighbor_elevs)
                local_slope_proxy[global_idx] = np.ptp(neighbor_elevs) / 100.0 # max diff / distance
                
                # Densities
                local_far_density[global_idx] = np.mean(fars[neighbors_arr])
                pad_100m[global_idx] = np.sum(bldg_areas[neighbors_arr]) / area_100m_circle
                
                # FAD: sum(width * height) / area
                frontal_areas = bldg_widths[neighbors_arr] * neighbor_heights
                fad_100m[global_idx] = np.sum(frontal_areas) / area_100m_circle
            else:
                height_above_local_mean[global_idx] = heights[global_idx]
                height_above_local_p90[global_idx] = heights[global_idx]
                local_mean_elevation[global_idx] = elevations[global_idx]
                
        del idxs_100m
        
        # 2. 250m Radius Queries
        idxs_250m = kd_tree.query_ball_point(chunk_centroids, r=250, workers=-1)
        for j, neighbors in enumerate(idxs_250m):
            global_idx = i + j
            neighbors_arr = np.array(neighbors, dtype=np.int32)
            neighbors_arr = neighbors_arr[neighbors_arr != global_idx]
            
            density_250m[global_idx] = len(neighbors_arr)
            if len(neighbors_arr) > 0:
                pad_250m[global_idx] = np.sum(bldg_areas[neighbors_arr]) / area_250m_circle
        del idxs_250m
        
        # 3. K-Nearest queries for SVF and Street width
        # Query 16 nearest to compute obstruction angles
        dists, idxs = kd_tree.query(chunk_centroids, k=16, workers=-1)
        for j in range(len(chunk_centroids)):
            global_idx = i + j
            
            # Distance for street width (k=1,2,3 are neighbors since 0 is self)
            avg_nn_dist[global_idx] = np.mean(dists[j, 1:4])
            
            # SVF computation (exclude self)
            valid_mask = idxs[j] != global_idx
            nbr_idxs = idxs[j][valid_mask]
            nbr_dists = np.clip(dists[j][valid_mask], 1.0, None)
            
            if len(nbr_idxs) > 0:
                target_h = heights[global_idx]
                nbr_h = heights[nbr_idxs]
                
                # Relative heights. Only positive obstruction matters
                rel_h = np.maximum(0, nbr_h - target_h)
                
                # Angle of obstruction in radians
                angles = np.arctan(rel_h / nbr_dists)
                
                # Proxy: 1.0 - mean(sin(obstruction_angle))
                # 1.0 means clear sky. Low values mean trapped.
                multi_svf[global_idx] = 1.0 - np.mean(np.sin(angles))
            else:
                multi_svf[global_idx] = 1.0
                
        del dists, idxs
        
        if (i // chunk_size) % 5 == 0 or i == 0:
            _log(f"  Processed {end:,}/{n_points:,} buildings...")
            
    pad_100m = np.clip(pad_100m, 0, 1.0)
    pad_250m = np.clip(pad_250m, 0, 1.0)
    
    # Advanced Roughness Length Proxy (Lettau 1969 adaptation)
    # Z0 = 0.5 * H_avg * FAD
    roughness_length_proxy = 0.5 * mean_neighbor_height * fad_100m
    
    # Store morphology results
    gdf["building_density_100m"] = density_100m
    gdf["building_density_250m"] = density_250m
    gdf["pad_100m"] = pad_100m
    gdf["pad_250m"] = pad_250m
    gdf["fad_100m"] = fad_100m
    gdf["roughness_length_proxy"] = roughness_length_proxy.astype(np.float32)
    gdf["multi_svf"] = multi_svf.astype(np.float32)
    
    gdf["mean_neighbor_height"] = mean_neighbor_height
    gdf["height_variance"] = height_variance
    gdf["local_far_density"] = local_far_density
    
    gdf["local_mean_elevation"] = local_mean_elevation
    gdf["local_elevation_variance"] = local_elevation_variance
    gdf["local_slope_proxy"] = local_slope_proxy
    
    gdf["local_max_height"] = local_max_height
    gdf["local_min_height"] = local_min_height
    gdf["local_median_height"] = local_median_height
    gdf["local_height_p90"] = local_height_p90
    gdf["local_height_p95"] = local_height_p95
    gdf["height_above_local_mean"] = height_above_local_mean
    gdf["height_above_local_p90"] = height_above_local_p90
    
    del kd_tree
    return gdf, gdf_proj, bldg_areas, avg_nn_dist

def compute_street_canyon_osm(gdf, gdf_proj, bldg_areas, avg_nn_dist, edges, nodes):
    _log("Computing Street Canyon & OSM Geometry Features...")
    
    edges_proj = edges.to_crs("EPSG:32618")
    nodes_proj = nodes.to_crs("EPSG:32618")
    bldg_centroids = gdf_proj.geometry.centroid
    
    edge_midpoints = np.column_stack([edges_proj.geometry.centroid.x, edges_proj.geometry.centroid.y])
    edge_kdtree = cKDTree(edge_midpoints)
    bldg_coords = np.column_stack([bldg_centroids.x, bldg_centroids.y])
    
    _, nearest_idx = edge_kdtree.query(bldg_coords, k=1, workers=-1)
    nearest_edges = edges_proj.iloc[nearest_idx].reset_index(drop=True)
    
    gdf["nearest_road_distance"] = bldg_centroids.reset_index(drop=True).distance(nearest_edges.geometry).values.astype(np.float32)
    
    bearings = pd.to_numeric(nearest_edges["bearing"], errors="coerce").fillna(0).values
    gdf["road_orientation"] = bearings.astype(np.float32)
    
    def parse_width(w, lanes):
        if pd.notna(w):
            try: return float(str(w).split(',')[0].replace('m', '').strip())
            except: pass
        if pd.notna(lanes):
            try: return float(str(lanes).split(',')[0]) * 3.5
            except: pass
        return 10.0
        
    for col in ['width', 'lanes']:
        if col not in nearest_edges.columns:
            nearest_edges[col] = np.nan
            
    street_width_osm = nearest_edges.apply(lambda row: parse_width(row.get('width'), row.get('lanes')), axis=1).values
    street_width_osm = np.clip(street_width_osm, 5.0, 100.0)
    gdf["street_width_osm"] = street_width_osm.astype(np.float32)
    
    heights = gdf["height_roof"].fillna(0).values.astype(np.float32)
    canyon_aspect_ratio = heights / street_width_osm
    gdf["canyon_aspect_ratio"] = canyon_aspect_ratio.astype(np.float32)
    
    n_points = len(bldg_coords)
    chunk_size = 25000
    
    edge_lengths = edges_proj.geometry.length.values
    road_density_100m = np.zeros(n_points, dtype=np.float32)
    
    node_coords = np.column_stack([nodes_proj.geometry.x, nodes_proj.geometry.y])
    node_kdtree = cKDTree(node_coords)
    intersection_density_250m = np.zeros(n_points, dtype=np.float32)
    
    for i in range(0, n_points, chunk_size):
        end = min(i + chunk_size, n_points)
        chunk_coords = bldg_coords[i:end]
        
        idxs_100m = edge_kdtree.query_ball_point(chunk_coords, r=100, workers=-1)
        for j, nbrs in enumerate(idxs_100m):
            if len(nbrs) > 0:
                road_density_100m[i+j] = np.sum(edge_lengths[nbrs])
                
        idxs_250m = node_kdtree.query_ball_point(chunk_coords, r=250, workers=-1)
        for j, nbrs in enumerate(idxs_250m):
            intersection_density_250m[i+j] = len(nbrs)
            
    gdf["road_density_100m"] = road_density_100m
    gdf["intersection_density_250m"] = intersection_density_250m
    
    del edges_proj, nodes_proj, edge_kdtree, node_kdtree
    return gdf

def compute_vegetation(gdf):
    _log("Computing Vegetation Cooling Features...")
    
    ndvi = gdf["ndvi"].fillna(0).values
    canopy = gdf["canopy_density"].fillna(0).values
    trees = gdf["nearby_tree_count"].fillna(0).values
    pad_100m = gdf["pad_100m"].fillna(0).values
    
    green_coverage = (normalize(ndvi) + normalize(canopy)) / 2.0
    tree_cooling = normalize(trees) * normalize(canopy)
    
    available_space = 1.0 - pad_100m
    gi_potential = (1.0 - green_coverage) * available_space
    
    gdf["green_coverage_score"] = normalize(green_coverage)
    gdf["tree_cooling_index"] = normalize(tree_cooling)
    gdf["green_infrastructure_potential"] = normalize(gi_potential)
    
    return gdf

def compute_advanced_indices(gdf):
    _log("Computing Advanced CFD/PINN Target Indices...")
    
    pad_100m = gdf["pad_100m"].values
    fad_100m = gdf["fad_100m"].values
    roughness = gdf["roughness_length_proxy"].values
    multi_svf = gdf["multi_svf"].values
    wind_speed = gdf["wind_speed"].fillna(0).values
    green_score = gdf["green_coverage_score"].values
    
    # Urban Morphology Index: combined roughness and mass
    morphology_index = normalize(pad_100m) * normalize(roughness)
    
    # Ventilation Efficiency Index: Wind vs Blockage/Roughness
    # High SVF and Wind, low FAD
    ventilation = normalize(wind_speed) * multi_svf / (normalize(fad_100m) + 0.1)
    
    # Thermal Trapping Index: High mass, low sky visibility, low green
    thermal_trapping = (1.0 - multi_svf) * normalize(pad_100m) * (1.0 - green_score)
    
    gdf["urban_morphology_index"] = normalize_100(morphology_index)
    gdf["ventilation_efficiency_index"] = normalize_100(ventilation)
    gdf["thermal_trapping_index"] = normalize_100(thermal_trapping)
    
    return gdf

def generate_physics_report(gdf):
    _log("Generating Urban Climate Physics Report (V3)...")
    
    report_path = REPORTS_DIR / "urban_climate_physics_v2.md"
    
    features = [
        ("pad_100m", "Plan Area Density (100m)", "Ratio (0-1)", "sum(area) / (pi*100^2)", "Proxy for impervious surface and thermal mass."),
        ("pad_250m", "Plan Area Density (250m)", "Ratio (0-1)", "sum(area) / (pi*250^2)", "Neighborhood scale building density."),
        ("fad_100m", "Frontal Area Density (100m)", "Ratio", "sum(equiv_width * height) / (pi*100^2)", "Aerodynamic obstruction facing wind vectors."),
        ("roughness_length_proxy", "Aerodynamic Roughness Length (Z0)", "Meters", "0.5 * mean_height * FAD", "Lettau (1969) adaptation for mechanical wind turbulence."),
        ("multi_svf", "Multi-Building Sky View Factor", "Ratio (0-1)", "1 - mean(sin(obstruction_angle_to_neighbors))", "True 3D angular sky obstruction replacing 2D street canyon cos(arctan)."),
        ("local_mean_elevation", "Mean Ground Elevation", "Meters", "mean(elevation in 100m)", "Determines baseline atmospheric pressures and localized cold pooling."),
        ("local_elevation_variance", "Ground Elevation Variance", "Meters", "std(elevation in 100m)", "Highlights complex, non-flat urban terrain."),
        ("local_slope_proxy", "Proxy for Local Slope", "Ratio", "max_elev_diff / 100", "Indicates steep topological features dictating drainage flows."),
        ("urban_morphology_index", "Urban Morphology Intensity", "Index (0-100)", "Norm(PAD) * Norm(Roughness)", "Combined structural obstruction and thermal capacity."),
        ("ventilation_efficiency_index", "Airflow Flushing Potential", "Index (0-100)", "Norm(Wind) * SVF / (Norm(FAD) + 0.1)", "How easily wind penetrates and removes heat/pollutants."),
        ("thermal_trapping_index", "Radiative Trapping Potential", "Index (0-100)", "(1-SVF) * Norm(PAD) * (1-Green)", "Identifies zones with restricted sky cooling and high thermal mass.")
    ]
    
    lines = [
        "# Urban Climate Physics & Morphology Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\nThis report details the advanced, physically-grounded indices used for CFD simulation geometries and PINN loss function weighting.",
        "\n## Target Variable Dictionary & Statistics\n"
    ]
    
    for f in features:
        col, dfn, unit, formula, interp = f
        
        s = gdf[col].dropna()
        if len(s) > 0:
            mean, std = s.mean(), s.std()
            min_v, max_v = s.min(), s.max()
            p25, p50, p75 = np.percentile(s, [25, 50, 75])
        else:
            mean = std = min_v = max_v = p25 = p50 = p75 = 0.0
            
        lines.extend([
            f"### `{col}`",
            f"- **Definition**: {dfn}",
            f"- **Physical Interpretation**: {interp}",
            f"- **Physics Formulation**: `{formula}`",
            f"- **Units**: {unit}",
            "- **Statistics**: ",
            f"  - Mean: {mean:.4f} ± {std:.4f}",
            f"  - Range: [{min_v:.4f}, {max_v:.4f}]",
            f"  - Quartiles: (25%: {p25:.4f}, 50%: {p50:.4f}, 75%: {p75:.4f})",
            ""
        ])
        
    report_path.write_text("\n".join(lines))
    _log(f"Physics validation report written to {report_path}")

def main():
    print("╔" + "═" * 64 + "╗")
    print("║  Phase 3 — Urban Climate Feature Engineering (V3 - Physics)   ║")
    print("╚" + "═" * 64 + "╝")
    
    t0 = time.time()
    
    in_path = PROCESSED_DIR / "building_master.parquet"
    out_path = PROCESSED_DIR / "building_climate_features_v3.parquet"
    
    edges_path = RAW_DIR / "nyc_roads_edges.parquet"
    nodes_path = RAW_DIR / "nyc_roads_nodes.parquet"
    
    if not edges_path.exists() or not nodes_path.exists():
        _log("ERROR: OSM parquets not found.")
        sys.exit(1)
    
    _log(f"Loading {in_path.name}...")
    gdf = gpd.read_parquet(in_path)
    
    _log("Loading OSM network data...")
    edges = gpd.read_parquet(edges_path)
    nodes = gpd.read_parquet(nodes_path)
    
    gdf, gdf_proj, bldg_areas, avg_nn_dist = compute_morphology_and_terrain(gdf)
    gdf = compute_street_canyon_osm(gdf, gdf_proj, bldg_areas, avg_nn_dist, edges, nodes)
    
    del gdf_proj, edges, nodes
    
    gdf = compute_vegetation(gdf)
    gdf = compute_advanced_indices(gdf)
    
    generate_physics_report(gdf)
    
    _log(f"Saving to {out_path.name}...")
    gdf.to_parquet(out_path, index=False)
    
    elapsed = time.time() - t0
    print(f"\n{'═' * 66}")
    print(f"  ✓ Feature Engineering (V3 Physics) complete in {elapsed/60:.1f} minutes")
    print(f"  Output: {out_path}")
    print(f"  Reports: reports/urban_climate_physics_v2.md")
    print(f"{'═' * 66}")

if __name__ == "__main__":
    main()
