# Parametric CFD Expansion Plan

## Objective
To horizontally expand the dataset from 16 to 768 samples by systematically perturbing meteorological variables across the 4 validated archetypes.

## Experimental Matrix
- **Archetypes**: 03, 06, 09, 10
- **Wind Speeds**: 2, 4, 6, 8, 10, 12 m/s
- **Wind Directions**: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- **Seasons**: Winter, Spring, Summer, Autumn

### Season Assumptions
*   **Winter**: T=273.15K, RH=40%, Neutral-Stable Atmosphere, TI_ref=0.12
*   **Spring**: T=288.15K, RH=55%, Neutral Atmosphere, TI_ref=0.10
*   **Summer**: T=303.15K, RH=65%, Unstable Atmosphere, TI_ref=0.08
*   **Autumn**: T=288.15K, RH=60%, Neutral Atmosphere, TI_ref=0.10

## Dataset Footprint
4 × 6 × 8 × 4 = **768** unique CFD configurations.
