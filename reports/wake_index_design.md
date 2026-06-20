# Wake Severity Index (WSI) Design

## Motivation
The Phase 8A binary wake classifier used a single threshold:
```
wake_flag = (speed < 0.3 * ref_speed)
```
This discards turbulence and pressure information, conflating calm low-speed zones with turbulent recirculation zones.

## WSI Formula
```
WSI = 0.5 * W_v + 0.3 * W_t + 0.2 * W_p
```

### Components

| Symbol | Name | Formula | Rationale |
|--------|------|---------|----------|
| W_v | Velocity Deficit | max(0, 1 − |U|/U_95) | Points with low speed relative to the freestream are in a wake |
| W_t | Turbulence | min(TKE/TKE_95, 1) | High TKE indicates shear-layer separation and vortex shedding |
| W_p | Pressure Deficit | clip((p_med − p)/(3·IQR), 0, 1) | Negative pressure deviation marks recirculation bubbles |

### Classification Thresholds

| WSI Range | Class | Planning Implication |
|-----------|-------|---------------------|
| < 0.25 | No Wake | Adequate ventilation |
| 0.25–0.50 | Mild Wake | Minor sheltering, generally acceptable |
| 0.50–0.75 | Strong Wake | Heat retention risk, reduced pollutant dispersion |
| > 0.75 | Severe Wake | Stagnation, thermal stress, mitigation required |

### Component Statistics

| Component | Mean | Std | P50 | P95 |
|-----------|------|-----|-----|-----|
| W_v | 0.7324 | 0.2874 | 0.8102 | 1.0000 |
| W_t | 0.2005 | 0.2298 | 0.1170 | 1.0000 |
| W_p | 0.2215 | 0.3972 | 0.0000 | 1.0000 |

**WSI**: mean=0.4707, std=0.1409, median=0.4737
