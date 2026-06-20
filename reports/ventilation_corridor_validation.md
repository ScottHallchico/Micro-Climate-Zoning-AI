# Ventilation Corridor — Validation Report

## Performance Comparison

| Archetype | Old Z1 Points | Old Mean Vel | New Corridors | New Z1 Points | New Mean Vel |
|---|---|---|---|---|---|
| 0 | 2,994 | 21.08 m/s | 1 | 1,610 | 24.24 m/s |
| 1 | 1,185 | 21.12 m/s | 4 | 1,159 | 19.05 m/s |

## Metrics Analysis
- **Corridor Continuity:** The new system extracts discrete, contiguous polygons rather than fragmented points.
- **Ventilation Gain:** The corridor method actively filters out high-speed vortices that fail the morphological requirements.
- **Spatial Coverage:** By bounding the footprint, the corridors are structurally usable for zoning and urban overlay mapping.

## Visual Validation
![Archetype 0 Comparison](file:///home/wangchen/Documents/Micro-Climate-Zoning-AI/reports/publication_figures/corridor_comparison_arch0.png)
![Archetype 1 Comparison](file:///home/wangchen/Documents/Micro-Climate-Zoning-AI/reports/publication_figures/corridor_comparison_arch1.png)
