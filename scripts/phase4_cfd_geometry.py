#!/usr/bin/env python3
"""
phase4_cfd_geometry.py — Phase 4B-A: CFD Geometry & ERA5 Extraction

- Fixed Issue 1 & 2: Constrained Domain Scaling Logic. H_ref is now capped 
  to prevent single-building outliers from causing domain explosions.
"""

import sys, json, time, platform
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import scipy
from scipy.interpolate import griddata
import trimesh
from shapely.geometry import Polygon, box
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import DATA_DIR, RAW_DIR, REPORTS_DIR

CFD_INPUTS = DATA_DIR / "cfd_inputs"
CFD_GEOMETRY = DATA_DIR / "cfd_geometry"
CFD_CONDITIONS = DATA_DIR / "cfd_conditions"
CFD_GEOMETRY.mkdir(parents=True, exist_ok=True)
CFD_CONDITIONS.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ARCHETYPES = [0, 1, 2, 4, 5, 7, 8]

def _log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

def rotate_point(x, y, angle_rad):
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a

def repair_mesh(mesh):
    """Rigorous mesh repair sequence to guarantee watertight, manifold OpenFOAM topology."""
    if mesh is None or getattr(mesh, 'is_empty', True) or len(getattr(mesh, 'vertices', [])) < 3:
        return mesh
        
    mesh.process() # Base trimesh processing (merges vertices)
    try:
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
        
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh

def generate_building_mesh(poly, height, ground_z):
    try:
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        mesh = trimesh.creation.extrude_polygon(poly, height, engine='earcut')
        mesh.apply_translation([0, 0, ground_z])
        mesh = repair_mesh(mesh)
        return mesh
    except Exception as e:
        print(f"Exception extruding: {e}")
        return None

def generate_tree_meshes(x, y, ground_z, dbh_inch):
    dbh_m = dbh_inch * 0.0254
    height_m = max(2.0, dbh_m * 30.0)
    trunk_radius = max(0.05, dbh_m / 2.0)
    canopy_radius = max(1.0, height_m * 0.3)
    trunk_h = height_m * 0.3
    
    try:
        trunk = trimesh.creation.cylinder(radius=trunk_radius, height=trunk_h, sections=8)
        trunk.apply_translation([0, 0, trunk_h / 2.0])
        trunk.apply_translation([x, y, ground_z])
        trunk = repair_mesh(trunk)
        
        canopy = trimesh.creation.icosphere(subdivisions=2, radius=canopy_radius)
        canopy.apply_translation([0, 0, trunk_h + (height_m - trunk_h)/2.0])
        canopy.apply_translation([x, y, ground_z])
        canopy = repair_mesh(canopy)
        
        meta = {
            "canopy_volume": float((4/3) * np.pi * (canopy_radius**3)),
            "canopy_radius": float(canopy_radius),
            "canopy_height": float(height_m - trunk_h),
            "projected_canopy_area": float(np.pi * canopy_radius**2),
            "drag_coefficient": 0.2,
            "LAD_estimate": 1.5,
            "center_x": float(x),
            "center_y": float(y),
            "ground_z": float(ground_z)
        }
        
        return trunk, canopy, meta
    except Exception:
        return None, None, None

def load_trees():
    tree_path = RAW_DIR / "tree_census_2015.csv"
    if not tree_path.exists():
        _log("  Warning: tree_census_2015.csv not found, trees will be skipped.")
        return None
    _log("  Loading tree census data...")
    df = pd.read_csv(tree_path, usecols=["tree_dbh", "latitude", "longitude", "status"])
    df = df[df["status"] == "Alive"].dropna(subset=["latitude", "longitude", "tree_dbh"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:32618")
    return gdf

def extract_era5_bcs(cid, center_lon, center_lat, out_dir):
    ds_path = RAW_DIR / "era5_nyc_hourly.nc"
    if not ds_path.exists():
        _log("  Warning: ERA5 data not found, skipping BCs.")
        return 270.0 # fallback west
        
    ds = xr.open_dataset(ds_path)
    
    lat_name = "latitude" if "latitude" in ds.coords else "lat" if "lat" in ds.coords else None
    lon_name = "longitude" if "longitude" in ds.coords else "lon" if "lon" in ds.coords else None
    
    if not lat_name or not lon_name:
        raise ValueError("Neither latitude/longitude nor lat/lon found in ERA5 coordinates.")
        
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    
    lat_idx = np.abs(lats - center_lat).argmin()
    lon_idx = np.abs(lons - center_lon).argmin()
    
    sel_lat = float(lats[lat_idx])
    sel_lon = float(lons[lon_idx])
    
    u10 = ds["u10"].isel({lat_name: lat_idx, lon_name: lon_idx}).values
    v10 = ds["v10"].isel({lat_name: lat_idx, lon_name: lon_idx}).values
    t2m = ds["t2m"].isel({lat_name: lat_idx, lon_name: lon_idx}).values
    
    u_mean = np.nanmean(u10)
    v_mean = np.nanmean(v10)
    prevailing_speed = np.sqrt(u_mean**2 + v_mean**2)
    prevailing_dir = (np.degrees(np.arctan2(-u_mean, -v_mean)) + 360) % 360
    
    mean_temp = np.nanmean(t2m)
    
    speed = np.sqrt(u10**2 + v10**2)
    direction = (np.degrees(np.arctan2(-u10, -v10)) + 360) % 360
    
    p95_val = np.nanpercentile(speed, 95)
    candidates_mask = speed >= p95_val
    max_idx = np.nanargmax(np.where(candidates_mask, speed, -1))
    
    high_speed = float(speed[max_idx])
    high_dir = float(direction[max_idx])
    
    time_name = "time" if "time" in ds.coords else "valid_time" if "valid_time" in ds.coords else None
    high_time_str = str(ds[time_name].values[max_idx]) if time_name else "Unknown"

    scenarios = {
        "A_prevailing": {
            "description": "Prevailing wind conditions (circular mean)",
            "wind_speed_ms": round(float(prevailing_speed), 2),
            "wind_direction_deg": round(float(prevailing_dir), 1),
            "temperature_K": round(float(mean_temp), 1),
        },
        "B_plus45": {
            "description": "Prevailing wind rotated +45°",
            "wind_speed_ms": round(float(prevailing_speed), 2),
            "wind_direction_deg": round(float((prevailing_dir + 45) % 360), 1),
            "temperature_K": round(float(mean_temp), 1),
        },
        "C_minus45": {
            "description": "Prevailing wind rotated −45°",
            "wind_speed_ms": round(float(prevailing_speed), 2),
            "wind_direction_deg": round(float((prevailing_dir - 45) % 360), 1),
            "temperature_K": round(float(mean_temp), 1),
        },
        "D_high_wind": {
            "description": "High-wind event (maximum above 95th percentile)",
            "wind_speed_ms": round(float(high_speed), 2),
            "wind_direction_deg": round(float(high_dir), 1),
            "temperature_K": round(float(mean_temp), 1),
            "p95_threshold": round(float(p95_val), 2),
            "timestamp": high_time_str
        },
    }

    cond_dir = CFD_CONDITIONS / f"archetype_{cid:02d}"
    cond_dir.mkdir(parents=True, exist_ok=True)
    for name, sc in scenarios.items():
        with open(cond_dir / f"{name}.json", "w") as f:
            json.dump(sc, f, indent=2)
            
    bc_meta = {
        "p95_threshold_ms": round(float(p95_val), 2),
        "selected_speed_ms": round(float(high_speed), 2),
        "selected_direction_deg": round(float(high_dir), 1),
        "selected_timestamp": high_time_str,
    }
    with open(cond_dir / "boundary_condition_metadata.json", "w") as f:
        json.dump(bc_meta, f, indent=2)
        
    era5_meta = {
        "lat_name": lat_name,
        "lon_name": lon_name,
        "grid_dimensions": list(ds.dims.keys()),
        "selected_grid_cell": {
            "latitude": sel_lat,
            "longitude": sel_lon
        }
    }
    with open(cond_dir / "era5_extraction.json", "w") as f:
        json.dump(era5_meta, f, indent=2)

    ds.close()
    return prevailing_dir

def generate_terrain_mesh(pts, z_vals, dom_x_min, dom_x_max, dom_y_min, dom_y_max):
    res = 10.0
    grid_x, grid_y = np.mgrid[dom_x_min:dom_x_max:res, dom_y_min:dom_y_max:res]
    
    if len(pts) > 0:
        pts_np = np.array(pts)
        z_np = np.array(z_vals)
        grid_z = griddata(pts_np, z_np, (grid_x, grid_y), method='linear', fill_value=np.mean(z_np))
    else:
        grid_z = np.zeros_like(grid_x)
        
    v_x = grid_x.flatten()
    v_y = grid_y.flatten()
    v_z = grid_z.flatten()
    
    verts = np.column_stack([v_x, v_y, v_z])
    
    rows, cols = grid_x.shape
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            faces.append([idx, idx + cols, idx + 1])
            faces.append([idx + 1, idx + cols, idx + cols + 1])
            
    terrain = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    terrain = repair_mesh(terrain)
    return terrain

def process_archetype(cid, trees_gdf):
    _log(f"\nProcessing Archetype {cid}...")
    
    in_file = CFD_INPUTS / f"neighborhood_{cid:02d}_250m.geojson"
    if not in_file.exists():
        _log(f"  ERROR: {in_file.name} not found.")
        return

    gdf = gpd.read_file(in_file)
    gdf_proj = gdf.to_crs("EPSG:32618")
    
    out_dir = CFD_GEOMETRY / f"archetype_{cid:02d}"
    domain_dir = out_dir / "domain"
    domain_dir.mkdir(parents=True, exist_ok=True)

    bounds = gdf_proj.total_bounds
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0
    
    meta_path = CFD_INPUTS / "archetype_metadata.json"
    prevailing_dir = 270.0
    if meta_path.exists():
        with open(meta_path) as f:
            all_meta_dict = json.load(f)
            all_meta = all_meta_dict.get("neighborhoods", [])
        nbr_meta = next((m for m in all_meta if m["archetype"] == cid), None)
        if nbr_meta:
            prevailing_dir = extract_era5_bcs(cid, nbr_meta["center_lon"], nbr_meta["center_lat"], out_dir)
            _log(f"  Extracted ERA5 BCs. Prevailing wind: {prevailing_dir:.1f}°")

    rotation_deg = 270.0 - prevailing_dir
    rotation_rad = np.radians(rotation_deg)
    
    valid_heights = []
    b_meshes = []
    
    for _, row in gdf_proj.iterrows():
        geom = row.geometry
        h = float(row.get("height_roof", 10) or 10)
        gz = float(row.get("ground_elevation", 0) or 0)
        
        valid_heights.append(h)
        
        if geom is None or geom.is_empty:
            continue
            
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            ext_coords = [(x - cx, y - cy) for x, y in poly.exterior.coords]
            rot_coords = [rotate_point(x, y, rotation_rad) for x, y in ext_coords]
            local_poly = Polygon(rot_coords)
            
            m = generate_building_mesh(local_poly, h, gz)
            if m is not None:
                b_meshes.append(m)
                
    if b_meshes:
        combined_b = trimesh.util.concatenate(b_meshes)
        combined_b = repair_mesh(combined_b)
        combined_b.export(str(out_dir / "buildings.stl"))
        combined_b.export(str(out_dir / "buildings.obj"))
        _log(f"  Exported buildings: {len(b_meshes)} geometries.")
    else:
        _log("  WARNING: No buildings generated.")
        
    t_trunks = []
    t_canopies = []
    veg_meta = []
    if trees_gdf is not None:
        tree_box = box(cx - 150, cy - 150, cx + 150, cy + 150)
        local_trees = trees_gdf[trees_gdf.geometry.within(tree_box)]
        
        pts_2d = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf_proj.geometry])
        elevs = gdf_proj["ground_elevation"].fillna(0).values
        
        for idx, row in local_trees.iterrows():
            tx = row.geometry.x - cx
            ty = row.geometry.y - cy
            rx, ry = rotate_point(tx, ty, rotation_rad)
            
            dist = np.sqrt((pts_2d[:, 0] - row.geometry.x)**2 + (pts_2d[:, 1] - row.geometry.y)**2)
            gz = elevs[np.argmin(dist)] if len(dist) > 0 else 0.0
            
            dbh = float(row["tree_dbh"])
            trunk, canopy, meta = generate_tree_meshes(rx, ry, gz, dbh)
            if trunk is not None and canopy is not None:
                t_trunks.append(trunk)
                t_canopies.append(canopy)
                meta["tree_id"] = str(idx)
                veg_meta.append(meta)
                
    if t_trunks:
        c_trunks = trimesh.util.concatenate(t_trunks)
        c_trunks = repair_mesh(c_trunks)
        c_trunks.export(str(out_dir / "trees.stl"))
        
        c_canopies = trimesh.util.concatenate(t_canopies)
        c_canopies = repair_mesh(c_canopies)
        c_canopies.export(str(out_dir / "canopies.stl"))
        
        with open(out_dir / "vegetation_zones.json", "w") as f:
            json.dump(veg_meta, f, indent=2)
        _log(f"  Exported trees: {len(t_trunks)} trunks and canopies.")
    else:
        _log("  No trees exported.")
        
    pts = []
    z_vals = []
    
    for _, row in gdf_proj.iterrows():
        cx_b = row.geometry.centroid.x - cx
        cy_b = row.geometry.centroid.y - cy
        rx, ry = rotate_point(cx_b, cy_b, rotation_rad)
        gz = float(row.get("ground_elevation", 0) or 0)
        pts.append((rx, ry))
        z_vals.append(gz)
        
    mean_elev = np.mean(z_vals) if z_vals else 0.0
    
    # FIX 1: Scientific Constrained H_ref to prevent domain explosion
    h_max_raw = max(valid_heights) if valid_heights else 10.0
    h_mean = np.mean(valid_heights) if valid_heights else 10.0
    h_p95 = np.percentile(valid_heights, 95) if valid_heights else 10.0
    
    # H_ref should not be dictated by a single skyscraper in a neighborhood.
    # We cap H_ref scientifically using 95th percentile and 3 * mean height logic.
    h_ref = min(h_max_raw, h_p95, 3.0 * h_mean)
    if h_max_raw > h_ref:
        _log(f"  WARNING: Outlier height {h_max_raw:.1f}m. Capping H_ref scientifically at {h_ref:.1f}m.")
    
    up_dist = 5.0 * h_ref
    down_dist = 15.0 * h_ref
    side_dist = 5.0 * h_ref
    top_dist = 6.0 * h_ref
    
    nx_min = -150.0
    nx_max =  150.0
    ny_min = -150.0
    ny_max =  150.0
    
    dom_x_min = nx_min - up_dist
    dom_x_max = nx_max + down_dist
    dom_y_min = ny_min - side_dist
    dom_y_max = ny_max + side_dist
    dom_z_max = mean_elev + h_max_raw + top_dist  # Still clear the actual tallest building
    
    boundary_pts = [
        (dom_x_min, dom_y_min), (dom_x_max, dom_y_min),
        (dom_x_max, dom_y_max), (dom_x_min, dom_y_max)
    ]
    for bx, by in boundary_pts:
        pts.append((bx, by))
        z_vals.append(mean_elev)
        
    terrain = generate_terrain_mesh(pts, z_vals, dom_x_min, dom_x_max, dom_y_min, dom_y_max)
    terrain.export(str(out_dir / "terrain.stl"))
    _log(f"  Exported terrain: {len(terrain.vertices)} vertices, {len(terrain.faces)} faces.")
    
    domain_mesh = trimesh.creation.box(bounds=[
        [dom_x_min, dom_y_min, terrain.vertices[:, 2].min()],
        [dom_x_max, dom_y_max, dom_z_max]
    ])
    domain_mesh = repair_mesh(domain_mesh)
    if len(getattr(domain_mesh, 'vertices', [])) > 2:
        domain_mesh.export(str(domain_dir / "domain.stl"))
    
    z_min_terrain = float(terrain.vertices[:, 2].min())
    z_max_terrain = float(terrain.vertices[:, 2].max())
    
    meta_out = {
        "archetype": cid,
        "n_buildings": len(b_meshes),
        "n_trees": len(t_trunks),
        "max_building_height": float(h_max_raw),
        "mean_building_height": float(h_mean),
        "h_ref_used": float(h_ref),
        "terrain_elevation_min": z_min_terrain,
        "terrain_elevation_max": z_max_terrain,
        "terrain_vertices": len(terrain.vertices),
        "terrain_faces": len(terrain.faces),
        "domain_bounds": {
            "x_min": float(dom_x_min),
            "x_max": float(dom_x_max),
            "y_min": float(dom_y_min),
            "y_max": float(dom_y_max),
            "z_min": z_min_terrain,
            "z_max": float(dom_z_max)
        },
        "padding": {
            "upstream": float(up_dist),
            "downstream": float(down_dist),
            "side": float(side_dist),
            "top": float(top_dist)
        },
        "prevailing_wind_direction": float(prevailing_dir)
    }
    with open(out_dir / "geometry_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)
        
    wind_align_meta = {
        "prevailing_direction_deg": float(prevailing_dir),
        "rotation_applied_deg": float(rotation_deg),
        "transformed_bounds": meta_out["domain_bounds"]
    }
    with open(out_dir / "wind_aligned_domain.json", "w") as f:
        json.dump(wind_align_meta, f, indent=2)

def main():
    print("╔" + "═" * 64 + "╗")
    print("║  Phase 4B-A — Geometry Generation (V6 Anomaly Fixed)          ║")
    print("╚" + "═" * 64 + "╝")
    t0 = time.time()
    
    trees_gdf = load_trees()
    
    for cid in TARGET_ARCHETYPES:
        process_archetype(cid, trees_gdf)
        
    elapsed = time.time() - t0
    print(f"\n{'═' * 66}")
    print(f"  ✓ Geometry generation complete in {elapsed/60:.1f} minutes.")
    print(f"{'═' * 66}")

if __name__ == "__main__":
    main()
