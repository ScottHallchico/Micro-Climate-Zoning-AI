# GNN Dataset Design

## Graph Construction Topology
The transition from tabular Machine Learning to Graph Neural Networks required converting the verified CFD spatial outputs into relational graphs.

### Node Representation
* **Entities**: Individual building footprints extracted from `neighborhood_XX_500m.geojson`.
* **Features**:
  * Height (m)
  * Footprint Area (m²)
  * Perimeter (m)
  * Compactness Ratio
* **Node Count**: Averaged ~150 nodes per graph.

### Edge Connectivity Strategy
* **Primary**: Distance-based threshold (Radius = 50.0m). This captures standard cross-canyon aerodynamic interactions.
* **Secondary**: Fallback to k-Nearest Neighbors (k=4) for heavily dispersed suburban forms.
* **Edge Features**: Euclidean separation distance to weight the message-passing convolutions.

### Global Context Injection
To condition the graph on the specific CFD boundary conditions, the meteorological variables (`wind_speed`, `wind_dir_sin`, `wind_dir_cos`, `season`) were explicitly concatenated onto every node feature vector before passing through the GNN layers. This enforces global awareness of the wind vector relative to the neighborhood orientation.
