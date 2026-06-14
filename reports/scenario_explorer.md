# Scenario Explorer

The Scenario Explorer is the core interactive component of the frontend Dashboard. 

## Features
- **Real-time Parametric Design**: Users manipulate sliders for building density, building height, vegetation fraction, wind speed, and wind direction.
- **Instant Flow Prediction**: Changes trigger an immediate `POST /predict` API call. Due to the 180x speedup of the PINN model over CFD, the UI updates almost instantaneously.
- **Delta Analysis**: Allows users to compare Scenario A vs Scenario B side-by-side, displaying velocity reduction and wake fraction increases as spatial heatmaps.
