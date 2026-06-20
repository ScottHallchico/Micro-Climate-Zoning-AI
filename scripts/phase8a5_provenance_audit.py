import geopandas as gpd
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

PROJECT_ROOT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ZONES_DIR = PROJECT_ROOT / "data" / "zones"
REPORTS_DIR = PROJECT_ROOT / "reports"

def main():
    print("Running Provenance Integrity Audit...")
    
    # Load GeoJSON and convert to a metric CRS to calculate actual area in square meters
    # EPSG:32618 is the local UTM zone for NYC
    gdf = gpd.read_file(ZONES_DIR / "climate_zones_provenance.geojson")
    gdf_metric = gdf.to_crs("EPSG:32618")
    
    # Calculate areas
    areas = gdf_metric.geometry.area
    gdf['area_sqm'] = areas
    
    # 2. Extract Statistics
    num_unique_classes = gdf['zone_type'].nunique()
    polygons_per_class = gdf['zone_type'].value_counts().to_dict()
    polygons_per_archetype = gdf['archetype'].value_counts().to_dict()
    
    mean_area = areas.mean()
    median_area = areas.median()
    largest_polygon = areas.max()
    smallest_polygon = areas.min()
    
    # 3. & 4. Audit Confidence Score & Compute Pearson Correlation
    # Exact formula used in phase8a4: min(1.0, pt_count / 100.0)
    
    point_counts = gdf['source_cfd_points_count'].astype(float)
    confidence_scores = gdf['confidence_score'].astype(float)
    
    pearson_corr, _ = pearsonr(point_counts, confidence_scores)
    
    # 5. Flag logic
    flagged = pearson_corr > 0.95
    
    # 6. Generate Report
    md = "# Provenance Integrity Audit\n\n"
    
    md += "## 1. Geometric Statistics\n"
    md += f"- **Unique Zone Classes:** {num_unique_classes}\n"
    
    md += "- **Polygons per Class:**\n"
    for k, v in polygons_per_class.items():
        md += f"  - `{k}`: {v}\n"
        
    md += "- **Polygons per Archetype:**\n"
    for k, v in polygons_per_archetype.items():
        md += f"  - `Archetype {k}`: {v}\n"
        
    md += f"- **Mean Polygon Area:** {mean_area:.2f} m²\n"
    md += f"- **Median Polygon Area:** {median_area:.2f} m²\n"
    md += f"- **Largest Polygon:** {largest_polygon:.2f} m²\n"
    md += f"- **Smallest Polygon:** {smallest_polygon:.2f} m²\n\n"
    
    md += "## 2. Confidence Score Audit\n"
    md += "- **Exact Formula Used:** `confidence_score = min(1.0, source_cfd_points_count / 100.0)`\n"
    md += "- **Feature Importance:** By definition, 100% of the variance in the unsaturated regime is explained exclusively by CFD point density.\n"
    md += f"- **Pearson Correlation (Confidence vs Point Count):** {pearson_corr:.4f}\n"
    
    if flagged:
        md += "\n> **⚠️ FLAG TRIGGERED:** Correlation between Confidence Score and CFD Point Count exceeds 0.95.\n"
        md += "> This indicates that the current confidence metric is mathematically entangled with grid resolution and does not capture true epistemic or predictive uncertainty.\n\n"
        
    md += "## 3. Recommendation\n"
    md += "**Action Required: RECALIBRATE CONFIDENCE METRIC**\n\n"
    md += "The current metric merely acts as a proxy for physical voxel density. A true epistemic confidence score must be decoupled from the grid structure. It should instead incorporate:\n"
    md += "1. **PINN Predictive Variance:** Derivable from the Monte Carlo Dropout variance (calculated in Phase 8A.3).\n"
    md += "2. **Local Velocity Gradient Variance:** Reflecting actual physical flow instability.\n"
    md += "3. **OOD Score:** The Isolation Forest prediction confidence.\n"
    
    (REPORTS_DIR / "provenance_integrity_audit.md").write_text(md)
    print("Done. Report generated at reports/provenance_integrity_audit.md")

if __name__ == "__main__":
    main()
