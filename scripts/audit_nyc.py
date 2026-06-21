import geopandas as gpd
import pandas as pd
import numpy as np
import json
import shapely

file_path = 'data/raw/BUILDING_20260602.geojson'
print("Loading geojson...")
gdf = gpd.read_file(file_path)
print("GeoJSON loaded.")

metrics = {}

# WORKSTREAM 1 - Health
metrics['total'] = len(gdf)
metrics['geom_types'] = {str(k): int(v) for k, v in gdf.geometry.geom_type.value_counts().to_dict().items()}
metrics['invalid'] = sum(~gdf.geometry.is_valid)
metrics['empty'] = sum(gdf.geometry.is_empty)
metrics['duplicates'] = sum(gdf.geometry.duplicated())

# WORKSTREAM 2 - Heights
h = pd.to_numeric(gdf['height_roof'], errors='coerce')
metrics['h_total'] = len(h)
metrics['h_missing'] = h.isna().sum()
metrics['h_pct_missing'] = metrics['h_missing'] / metrics['total'] * 100
metrics['h_pct_present'] = 100 - metrics['h_pct_missing']

h_valid = h.dropna()
metrics['h_min'] = h_valid.min() if len(h_valid) else 0
metrics['h_max'] = h_valid.max() if len(h_valid) else 0
metrics['h_mean'] = h_valid.mean() if len(h_valid) else 0
metrics['h_median'] = h_valid.median() if len(h_valid) else 0
metrics['h_zero'] = sum(h_valid == 0)
metrics['h_outliers'] = sum(h_valid > 500) # Extreme outliers (e.g. > 500m)

# WORKSTREAM 3 - Complexity
def count_vertices(geom):
    if not geom or geom.is_empty:
        return 0
    if geom.geom_type == 'Polygon':
        return len(geom.exterior.coords) + sum(len(i.coords) for i in geom.interiors)
    elif geom.geom_type == 'MultiPolygon':
        return sum(count_vertices(p) for p in geom.geoms)
    return 0

print("Calculating vertices...")
vertex_counts = gdf.geometry.apply(count_vertices)
metrics['v_avg'] = vertex_counts.mean()
metrics['v_max'] = vertex_counts.max()

for k, v in metrics.items():
    if isinstance(v, (np.int64, np.int32)):
        metrics[k] = int(v)
    elif isinstance(v, (np.float64, np.float32)):
        metrics[k] = float(v)

with open('scratch_audit_results.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Audit complete.")
