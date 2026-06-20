# Feature Leakage Audit

Audited morphological feature token derivation against CFD output contamination.

## Feature Providence
- `frontal_area_density`: Derived from bounding boxes -> **CLEAN**
- `canyon_aspect_ratio`: Derived from ray-tracing -> **CLEAN**
- `blockage_ratio`: Derived from spatial footprint -> **CLEAN**
- `wind_alignment`: Derived from inlet vector -> **CLEAN**

**Verdict**: No CFD-derived targets (`u,v,w,p,k,wake`) exist in the feature tensor pipeline.
