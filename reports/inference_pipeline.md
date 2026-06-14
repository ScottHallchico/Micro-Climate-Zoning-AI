# Inference Pipeline

The deployment inference pipeline is fully decoupled from the training routines, designed for low-latency requests via FastAPI.

## Execution Flow
1. **Geometry Ingestion**: API receives building polygons (or archetype ID) and weather context (Wind Speed, Direction, Density).
2. **Graph Builder**: Instantly re-projects geometry to `EPSG:32618` to prevent edge explosion. Constructs distance edges < 50m and aerodynamic upstream/downstream flow masks.
3. **GAT Encoder**: Passes the PyG `Data` object through 2 layers of `GATv2Conv` to generate a 64-dimensional topological embedding.
4. **PINN Decoder**: Concatenates the topological embedding with the dense $X, Y, Z$ grid and evaluates the dense network to generate a 5-channel output field ($U, V, W, P, K$).
5. **Postprocessor**: Extracts the 10th percentile velocity thresholds to classify regions into binary Wake/Non-Wake zones, converting raw tensors into Base64 Plotly/Deck.gl compatible JSON payloads.
