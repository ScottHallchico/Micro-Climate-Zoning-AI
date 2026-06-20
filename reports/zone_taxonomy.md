# Climate Zone Taxonomy

## Overview
The Urban Climate Zoning Engine converts raw aerodynamic predictions (velocity, wake fraction, TKE, pressure, VEI) into interpretable planning classes that urban designers can act on directly.

## Zone Definitions

| Zone ID | Name | Description | Physical Criteria |
|---------|------|-------------|-------------------|
| **Z1** | Ventilation Corridor | Open-air channels with high through-flow. Ideal for pedestrian comfort and heat dissipation. | VEI > 1.5 **and** Wake Fraction < 0.15 |
| **Z2** | Comfortable Urban Climate | Well-ventilated areas with moderate turbulence. Suitable for outdoor seating, parks, plazas. | 0.8 < VEI ≤ 1.5 **and** TKE < 1.0 m²/s² |
| **Z3** | Neutral Mixed Zone | Neither strongly ventilated nor stagnant. Typical streetscape conditions. | 0.5 < VEI ≤ 0.8 **and** Wake Fraction < 0.4 |
| **Z4** | Heat Retention Zone | Reduced airflow traps urban heat. Risk of elevated temperatures during summer. | VEI ≤ 0.5 **and** Wake Fraction ≥ 0.3 **and** TKE < 0.5 m²/s² |
| **Z5** | Stagnation Risk Zone | Near-zero velocity recirculation. Pollutant accumulation and thermal stress risk. | Wake Fraction ≥ 0.5 **and** Mean Velocity < 1.0 m/s |
| **Z6** | Wind Hazard Zone | Dangerous wind acceleration (Venturi effect, corner flows). Pedestrian safety risk. | Mean Velocity > 10 m/s **or** TKE > 3.0 m²/s² |

## Hierarchy
Zones are evaluated in priority order: **Z6 → Z5 → Z1 → Z4 → Z2 → Z3** (default fallback).

Higher-priority zones override lower-priority ones when multiple criteria are satisfied simultaneously. This ensures safety-critical classifications (Wind Hazard, Stagnation) always dominate.

## Color Scheme
| Zone | Hex Color | Rationale |
|------|-----------|-----------|
| Z1 | `#22D3EE` (Cyan) | Fresh air, openness |
| Z2 | `#10B981` (Emerald) | Comfort, green spaces |
| Z3 | `#A78BFA` (Violet) | Neutral, transitional |
| Z4 | `#F59E0B` (Amber) | Heat warning |
| Z5 | `#EF4444` (Red) | Danger, stagnation |
| Z6 | `#F472B6` (Pink) | Hazard, extreme wind |
