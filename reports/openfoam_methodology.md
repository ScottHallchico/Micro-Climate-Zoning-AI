# OpenFOAM Methodology Report

## Turbulence Model
The computational model leverages the **k-omega SST (Shear Stress Transport)** turbulence model. This formulation combines the standard $k-\omega$ model near the wall and the $k-\epsilon$ model in the free stream, offering superior performance for complex urban environments with massive flow separation, wakes, and adverse pressure gradients compared to standard $k-\epsilon$. 

## Atmospheric Boundary Layer (ABL) Formulation
We model the inflow using the established `atmBoundaryLayerInlet` profiles in OpenFOAM. 
- **Velocity (`atmBoundaryLayerInletVelocity`)**: Establishes a logarithmic wind profile scaling vertically from the reference wind speed at 10m derived from the ERA5 boundary conditions. Flow direction is explicitly parameterized using $(U_x, U_y, 0)$ vectors.
- **Turbulent Kinetic Energy (`atmBoundaryLayerInletK`)**: Calculates $k$ dynamically based on shear stress and roughness.
- **Specific Dissipation Rate (`atmBoundaryLayerInletOmega`)**: Establishes the expected turbulence scale for the boundary layer.

## Roughness Assumptions
Surface roughness ($z_0$) is derived empirically from the urban morphology of each archetype. We use the approximation:
$z_0 = 0.1 \times \text{Mean Building Height}$
This provides a representative aerodynamic roughness length characterizing the physical drag of the archetype upwind fetch without requiring explicit physical rendering of far-field obstacles.

## Vegetation Model
Vegetation canopies are resolved explicitly as porous media using `fvOptions` with the **Darcy-Forchheimer** porosity model.
Instead of solid blockage, trees exert a parameterized drag force representing foliage.
The drag coefficients are estimated from standard Leaf Area Density (LAD) parameters.

## Solver Settings
- **Solver**: `simpleFoam` (Steady-state Reynolds-Averaged Navier-Stokes).
- **Discretization**: `bounded Gauss upwind` for turbulence variables, `cellMDLimited Gauss linear 0.5` for gradients.
- **Pressure-Velocity Coupling**: SIMPLE algorithm with 1 non-orthogonal corrector to handle minor mesh skewness.

## Convergence Criteria
The solution is defined as converged when the scaled residuals fall below:
- $p$ < $1 \times 10^{-4}$
- $U$ < $1 \times 10^{-5}$
- $k$ < $1 \times 10^{-5}$
- $\omega$ < $1 \times 10^{-5}$
Relaxation factors are set to $0.3$ for pressure and $0.7$ for momentum and turbulence to ensure stability.
