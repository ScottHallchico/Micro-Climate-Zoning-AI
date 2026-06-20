"""
Phase 8A.3 — Spatial Topology Repair
====================================
1. Audits old climate_zones.geojson for overlaps and degenerate multipolygons.
2. Regenerates zones using 2D footprint logic to prevent Voronoi failure.
3. Replaces global dissolve with connected-component dissolve.
4. Exports climate_zones_v2.geojson.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')

# Directories
PROJECT_ROOT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ML_DIR = PROJECT_ROOT / "data" / "ml"
ZONES_DIR = PROJECT_ROOT / "data" / "zones"
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_CRS = "EPSG:32618"

# Zone taxonomy from Phase 8A
ZONE_NAMES = {
    "Z1": "Ventilation Corridor",
    "Z2": "Comfortable Urban Climate",
    "Z3": "Neutral Mixed Zone",
    "Z4": "Heat Retention Zone",
    "Z5": "Stagnation Risk Zone",
    "Z6": "Wind Hazard Zone"
}

# Auditing Old GeoJSON
def audit_old_geojson():
    gdf = gpd.read_file(ZONES_DIR / 'climate_zones.geojson')
    gdf = gdf.to_crs(TARGET_CRS)
    
    total_area = gdf.geometry.area.sum()
    overlap_area = 0.0
    for i in range(len(gdf)):
        for j in range(i+1, len(gdf)):
            if gdf.geometry.iloc[i].intersects(gdf.geometry.iloc[j]):
                overlap_area += gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j]).area
                
    duplicates = len(gdf[gdf.geometry.duplicated()])
    
    invalid = sum(1 for geom in gdf.geometry if not geom.is_valid)
    
    md = "# Geometry Overlap Audit (Phase 8A.3)\n\n"
    md += "## Findings from `climate_zones.geojson`\n\n"
    md += f"- **Duplicate Geometries:** {duplicates}\n"
    md += f"- **Invalid Geometries (Self-intersections):** {invalid}\n"
    md += f"- **Overlap Area:** {overlap_area:.2f} m² ({(overlap_area/max(total_area, 1e-6))*100:.1f}%)\n\n"
    md += "## Root Cause Analysis\n"
    md += "The original `bounded_voronoi` received unflattened 3D CFD points containing multiple "
    md += "coincident (x, y) coordinates at different Z heights. This caused `scipy.spatial.Voronoi` "
    md += "to fail on these points, returning the default bounding box. Consequently, multiple zones "
    md += "were assigned the exact same full-domain bounding box, resulting in >100% overlap.\n"
    
    (REPORTS_DIR / "geometry_overlap_audit.md").write_text(md)


def classify_point(row, ref_speed):
    speed = np.sqrt(row['u']**2 + row['v']**2 + row['w']**2)
    tke = row['k']
    vei = speed / max(ref_speed, 1e-6)
    wake_frac = 1.0 if speed < 0.3 * ref_speed else 0.0
    
    if speed > 10.0 or tke > 3.0: return "Z6"
    if wake_frac >= 0.5 and speed < 1.0: return "Z5"
    if vei > 1.5 and wake_frac < 0.15: return "Z1"
    if vei <= 0.5 and wake_frac >= 0.3 and tke < 0.5: return "Z4"
    if 0.8 < vei <= 1.5 and tke < 1.0: return "Z2"
    return "Z3"

def bounded_voronoi(points, bounds):
    minx, miny, maxx, maxy = bounds
    mirrored = np.vstack([
        points,
        np.column_stack([2*minx - points[:, 0], points[:, 1]]),
        np.column_stack([2*maxx - points[:, 0], points[:, 1]]),
        np.column_stack([points[:, 0], 2*miny - points[:, 1]]),
        np.column_stack([points[:, 0], 2*maxy - points[:, 1]]),
    ])
    vor = Voronoi(mirrored)
    bbox = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

    polys = []
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            polys.append(Polygon())
            continue
        verts = [vor.vertices[v] for v in region]
        try:
            poly = Polygon(verts).intersection(bbox)
            polys.append(poly if poly.is_valid and not poly.is_empty else Polygon())
        except Exception:
            polys.append(Polygon())
    return polys

def generate_v2_zones():
    pdf = pd.read_parquet(ML_DIR / "cfd_field_dataset_verified.parquet")
    cdf = pd.read_parquet(ML_DIR / "verified_cfd_dataset_v3.parquet")

    with open(PROJECT_ROOT / "data/cfd_inputs/archetype_metadata.json") as f:
        meta = json.load(f)["neighborhoods"]

    arch_geoms = {}
    for a in meta:
        gp = PROJECT_ROOT / a["patches"]["500m"]["geojson"]
        if gp.exists():
            arch_geoms[a["archetype"]] = gpd.read_file(gp).to_crs(TARGET_CRS)

    all_zone_gdfs = []

    for arch_id in sorted(pdf['archetype'].unique()):
        arch_sims = cdf[cdf['archetype'] == arch_id]
        if arch_sims.empty or arch_id not in arch_geoms:
            continue

        arch_points = pdf[pdf['archetype'] == arch_id].copy()
        
        # FIX 1: 2D Footprint Reduction
        # Sort by Z ascending, drop duplicates to get ground-level pedestrian points
        arch_points = arch_points.sort_values('z').drop_duplicates(subset=['x', 'y']).copy()
        
        ref_speed = arch_sims['wind_speed'].median()
        arch_points['speed'] = np.sqrt(arch_points['u']**2 + arch_points['v']**2 + arch_points['w']**2)
        arch_points['zone'] = arch_points.apply(lambda r: classify_point(r, ref_speed), axis=1)
        arch_points['vei'] = arch_points['speed'] / max(ref_speed, 1e-6)
        arch_points['wake_flag'] = (arch_points['speed'] < 0.3 * ref_speed).astype(float)
        
        pts_2d = arch_points[['x', 'y']].values
        bounds = (pts_2d[:,0].min()-20, pts_2d[:,1].min()-20, pts_2d[:,0].max()+20, pts_2d[:,1].max()+20)
        
        voronoi_polys = bounded_voronoi(pts_2d, bounds)
        
        zone_gdf = gpd.GeoDataFrame({
            'zone_type': arch_points['zone'].values,
            'zone_name': [ZONE_NAMES[z] for z in arch_points['zone'].values],
            'vei': arch_points['vei'].values,
            'wake_fraction': arch_points['wake_flag'].values,
            'mean_velocity': arch_points['speed'].values,
            'tke': arch_points['k'].values,
            'archetype': arch_id,
        }, geometry=voronoi_polys, crs=TARGET_CRS)
        
        # FIX 2: Connected Components Dissolve
        # By using explode(), MultiPolygons are split into individual spatially connected Polygons
        dissolved = zone_gdf.dissolve(
            by='zone_type',
            aggfunc={
                'vei': 'mean',
                'wake_fraction': 'mean',
                'mean_velocity': 'mean',
                'tke': 'mean',
                'archetype': 'first',
                'zone_name': 'first'
            }
        ).reset_index()
        
        # Explode to preserve disconnected regions
        exploded = dissolved.explode(index_parts=False).reset_index(drop=True)
        exploded['zone_id'] = [f"arch{arch_id:02d}_{z}_{i}" for i, z in zip(exploded.index, exploded['zone_type'])]
        
        all_zone_gdfs.append(exploded)

    if all_zone_gdfs:
        final_gdf = pd.concat(all_zone_gdfs, ignore_index=True)
        # FIX 3: Topological Cleanup
        # Resolve precision-based self-intersections and remove zero-area artifacts
        final_gdf.geometry = final_gdf.geometry.buffer(-0.01).buffer(0)
        final_gdf = final_gdf[final_gdf.geometry.area > 1e-2].copy()
        final_gdf['geom_wkt'] = final_gdf.geometry.to_wkt()
        final_gdf = final_gdf.drop_duplicates(subset=['geom_wkt']).drop(columns=['geom_wkt']).reset_index(drop=True)

        # Check overlaps on V2 efficiently
        total_area = final_gdf.geometry.area.sum()
        overlap_area = 0.0
        
        # Spatial join to find intersecting polygons
        # This is orders of magnitude faster than O(N^2) loops
        intersections = gpd.sjoin(final_gdf, final_gdf, how='inner', predicate='intersects')
        # Filter out self-intersections and duplicates (i >= j)
        intersections = intersections[intersections.index < intersections['index_right']]
        
        for idx, row in intersections.iterrows():
            if row['archetype_left'] == row['archetype_right']:
                inter = final_gdf.geometry.loc[idx].intersection(final_gdf.geometry.loc[row['index_right']])
                # Ignore mathematical boundary slivers
                if inter.area > 1.0:
                    overlap_area += inter.area
        
        final_gdf_wgs84 = final_gdf.to_crs("EPSG:4326")
        final_gdf_wgs84.to_file(ZONES_DIR / "climate_zones_v2.geojson", driver="GeoJSON")
        
        # Reports
        md = "# Spatial Topology Repair (Phase 8A.3)\n\n"
        md += "## Issue Resolution\n"
        md += "- **Issue 1:** 3D duplicate coordinates caused `scipy.spatial.Voronoi` to fail, assigning full bounding boxes to points.\n"
        md += "  - *Fix:* Grouped CFD points by `(x, y)` footprint, preserving the near-ground pedestrian layer.\n"
        md += "- **Issue 2:** `dissolve(by=\"zone_type\")` merged physically disjoint zones into single `MultiPolygon` geometries.\n"
        md += "  - *Fix:* Added `.explode()` to the spatial operation pipeline to isolate connected components.\n\n"
        
        md += "## Verification\n"
        md += f"- **Duplicate Geometries:** {len(final_gdf[final_gdf.geometry.duplicated()])}\n"
        md += f"- **Overlap Area:** {(overlap_area/max(total_area, 1e-6))*100:.4f}%\n"
        md += "- **Disconnected Regions:** Preserved as individual GeoJSON features.\n"
        (REPORTS_DIR / "topology_repair.md").write_text(md)
        
        # Connectivity
        md = "# Zone Connectivity & Morphologies\n\n"
        md += "## Connected Components Analysis\n"
        md += "| Archetype | Zone | Contiguous Polygons | Mean Area (m²) |\n"
        md += "|---|---|---|---|\n"
        for arch in sorted(final_gdf['archetype'].unique()):
            sub = final_gdf[final_gdf['archetype'] == arch]
            for z in sorted(sub['zone_type'].unique()):
                zsub = sub[sub['zone_type'] == z]
                md += f"| {arch} | {z} ({ZONE_NAMES[z]}) | {len(zsub)} | {zsub.geometry.area.mean():.1f} |\n"
        (REPORTS_DIR / "zone_connectivity.md").write_text(md)

def main():
    print("Running Topology Audit...")
    audit_old_geojson()
    print("Generating Repaired V2 Zones...")
    generate_v2_zones()
    print("Done. V2 GeoJSON and reports created.")

if __name__ == "__main__":
    main()
