# Dataset Target Cleanup

Prior to Phase 5A-R2 Dataset Expansion, the following synthetic and heuristic labels have been strictly removed from the machine learning target space:

1. **`cfd_recirculation_fraction`**: This variable was mathematically derived by applying an arbitrary multiplier (0.8) to the `cfd_wake_fraction`. Because it does not independently originate from OpenFOAM physical equations, it violates the requirement for genuine, traceable CFD features.

2. **`cfd_pedestrian_comfort`**: This variable was approximated using a simple linear penalty derived from the wind velocity (`1 - velocity/15`). It is an interpreted proxy variable rather than a primary CFD simulation output.

By dropping these derived labels, we ensure that every target in the final surrogate training dataset is 100% physically solved via the Navier-Stokes equations natively within the OpenFOAM framework.
