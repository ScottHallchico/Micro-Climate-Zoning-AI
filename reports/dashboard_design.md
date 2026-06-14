# Dashboard Design

## Framework
React + TypeScript using Vite. Styled with Tailwind CSS for rapid prototyping and professional aesthetics.

## Key Views
1. **Home / Overview**: High-level summary of the Urban Climate AI system.
2. **Neighborhood Analysis**: Allows selection of base archetypes and inputs for weather conditions.
3. **Flow Visualization**: Utilizes interactive web maps (e.g. Deck.gl / Mapbox / Plotly 3D) to render the 3D volume outputs of the PINN model natively in the browser. Users can slice the volume at specific Z-heights.
4. **Uncertainty Viewer**: Visualizes the predictive variance from Monte Carlo Dropout, highlighting regions where the model has low confidence (OOD geometry).
5. **Model Diagnostics**: Displays the `/health` endpoint response, proving which model checkpoint is currently active in the backend.
