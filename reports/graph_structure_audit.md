# Graph Structure Audit

- 20m Radius Graph Edges: 472 (Density: 0.9)
- 50m Radius Graph Edges: 5206 (Density: 10.4)
- Wind Influence Pruning: Cuts ~45% of edges (enforcing causality but creating disconnected local components at low subsample rates).

**Conclusion**: At 500 nodes, 20m radius graphs are too fragmented. Multi-scale (50m+) is absolutely required to maintain connected momentum flow.