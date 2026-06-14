# Tree Allometry Diagnostics & Assumptions

Generated: 2026-06-14 01:24:47

## Biological Assumptions
The following empirical heuristics are utilized to approximate tree mesh volumes from the NYC Tree Census.
They translate simple structural variables (DBH - Diameter at Breast Height) into CFD-ready obstructions.

## Equations & Units
- **DBH** is converted from inches to meters: `DBH_m = DBH_in * 0.0254`
- **Height**: `H = max(2.0, DBH_m * 30.0)` meters
- **Trunk Radius**: `R_t = max(0.05, DBH_m / 2.0)` meters
- **Trunk Height**: `H_t = H * 0.3` meters
- **Canopy Radius**: `R_c = max(1.0, H * 0.3)` meters

## Intended CFD Interpretation
The trunks act as solid slip walls. The canopy is currently represented as a solid watertight shell (icosphere) but is intended to be parameterized as a porous medium region in advanced buoyantFoam models. The approximations ensure volume presence without violating mesh resolution minimums.

## Limitations
- Exact crown structures, species variations, and seasonal foliage density are excluded.
- Root geometry and ground-level structural interference are ignored.