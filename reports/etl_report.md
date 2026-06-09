# ETL Pipeline Report

Generated: 2026-06-09 14:46:31

## Output: `data/processed/building_master.parquet`
- **Rows:** 1,082,945
- **Columns:** 18
- **CRS:** EPSG:4326

## Processing Steps

| Step | Duration |
|------|----------|
| Load Building Footprints | 84.3s |
| Load MapPLUTO | 3.7s |
| Load Tree Census | 3.3s |
| Join Buildings ↔ MapPLUTO | 0.8s |
| Compute Tree Features (spatial) | 16.9s |
| Compute Raster Features | 97.5s |
| Compute Weather Features (ERA5) | 1.8s |
| Finalize & Save | 3.0s |
| **Total** | **211.3s** |

## Join Statistics

- Buildings ↔ MapPLUTO: 920,139/1,082,945 (85.0% match)
- Buildings with ≥1 tree within 100m: 1,062,858/1,082,945 (98.1%)