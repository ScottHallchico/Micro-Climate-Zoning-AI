# Provenance Integrity Audit

## 1. Geometric Statistics
- **Unique Zone Classes:** 5
- **Polygons per Class:**
  - `Z3`: 267
  - `Z6`: 228
  - `Z2`: 169
  - `Z4`: 151
  - `Z5`: 13
- **Polygons per Archetype:**
  - `Archetype 0`: 480
  - `Archetype 1`: 348
- **Mean Polygon Area:** 1691.95 m²
- **Median Polygon Area:** 41.15 m²
- **Largest Polygon:** 422318.86 m²
- **Smallest Polygon:** 9.90 m²

## 2. Confidence Score Audit
- **Exact Formula Used:** `confidence_score = min(1.0, source_cfd_points_count / 100.0)`
- **Feature Importance:** By definition, 100% of the variance in the unsaturated regime is explained exclusively by CFD point density.
- **Pearson Correlation (Confidence vs Point Count):** 0.2392
## 3. Recommendation
**Action Required: RECALIBRATE CONFIDENCE METRIC**

The current metric merely acts as a proxy for physical voxel density. A true epistemic confidence score must be decoupled from the grid structure. It should instead incorporate:
1. **PINN Predictive Variance:** Derivable from the Monte Carlo Dropout variance (calculated in Phase 8A.3).
2. **Local Velocity Gradient Variance:** Reflecting actual physical flow instability.
3. **OOD Score:** The Isolation Forest prediction confidence.
