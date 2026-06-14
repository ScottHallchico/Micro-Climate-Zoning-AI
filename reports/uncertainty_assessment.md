# Uncertainty Assessment

## Sources of Uncertainty
1. **Inlet Boundary Condition**: The standard logarithmic ABL profile assumes neutral atmospheric stability. Thermal stratifications (buoyancy) are neglected in this steady-state isothermal run.
2. **Turbulence Modeling**: While $k-\omega$ SST is state-of-the-art for RANS, it fundamentally overpredicts stagnation turbulence kinetic energy on windward facades compared to LES.
3. **Vegetation Porosity**: Darcy-Forchheimer coefficients were uniformly applied based on estimated LAD. True foliage distributions are heterogeneous.
4. **Mesh Discretization**: The primary mesh resolution is 10m with iterative snapping to 1.25m near walls. Some micro-scale aerodynamic features <1m are lost.

## Recommendation
The uncertainties are bounded and acceptable for neighborhood-scale pedestrian wind comfort and ventilation studies.
