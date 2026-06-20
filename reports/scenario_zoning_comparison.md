# Scenario Zoning Comparison

## Implementation
The `/compare` route allows users to define two urban density scenarios (A and B) and instantly compare their aerodynamic and zoning impacts.

## Metrics Displayed
- **Ventilation Gain**: Δ velocity (m/s) between layouts
- **Wake Reduction**: Δ wake fraction
- **VEI Change**: Δ ventilation efficiency index
- **Zone Migration**: Visual indicator when the predicted climate zone class changes between scenarios (e.g., Z5 → Z3)

## Use Case
Urban planners can evaluate the impact of reducing building density from λp = 0.55 to λp = 0.30, immediately seeing whether the intervention transitions a neighborhood from a Stagnation Risk Zone to a Comfortable Climate Zone.
