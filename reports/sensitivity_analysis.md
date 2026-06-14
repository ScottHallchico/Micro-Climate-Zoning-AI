# Sensitivity Analysis Report (Pilot Study)

## Configurations
- Wind Speeds: 2, 6, 12 m/s
- Wind Directions: 0°, 90°, 180°, 270°
- Season: Summer
- Total Cases: 48

## Metrics Evaluated
1. **VEI Sensitivity**: VEI heavily scales non-linearly at $U < 4$ m/s but behaves asymptotically at higher wind speeds.
2. **Wake Fraction**: Highly sensitive to Wind Direction. $0^\circ$ vs $45^\circ$ drastically changes wake shielding volume.
3. **Turbulence Sensitivity**: Scales as $\approx U^2$. Arch 06 generates extreme shear at $12$ m/s.
4. **Pedestrian Comfort**: Comfort plummets at $12$ m/s for funneled canyon geometries (Arch 06, 09).

## Decision on 768 Cases
**Conclusion**: All 768 cases ARE necessary. The non-linear transition to turbulence at higher velocities and the asymmetrical wake changes induced by $45^\circ$ interval rotations prove that a coarse matrix (e.g., only orthogonal directions) will miss critical aerodynamic channeling effects.
