# Tree Allometry Diagnostics & Assumptions

Generated: 2026-06-14 01:36:26

## Biological Assumptions
The following empirical heuristics are utilized to approximate tree mesh volumes from the NYC Tree Census.
They translate simple structural variables (DBH - Diameter at Breast Height) into CFD-ready obstructions.

## Equations & Units
- **DBH** is converted from inches to meters: `DBH_m = DBH_in * 0.0254`
- **Height**: `H = max(2.0, DBH_m * 30.0)` meters
- **Trunk Radius**: `R_t = max(0.05, DBH_m / 2.0)` meters
- **Trunk Height**: `H_t = H * 0.3` meters
- **Canopy Radius**: `R_c = max(1.0, H * 0.3)` meters

## Intended CFD Interpretation (Vegetation Model)
The trunks act as solid slip walls. The canopy is explicitly separated and exported as `canopies.stl`. Canopies are modeled as aerodynamic momentum sinks (porous zones using fvOptions).
The following defaults are utilized for vegetation metadata based on generic deciduous urban trees:
- **Drag Coefficient (Cd)**: 0.2 (Kenjereš & ter Kuile, 2013)
- **Leaf Area Density (LAD)**: 1.5 m²/m³

## Limitations
- Exact crown structures, species variations, and seasonal foliage density are excluded.
- Root geometry and ground-level structural interference are ignored.