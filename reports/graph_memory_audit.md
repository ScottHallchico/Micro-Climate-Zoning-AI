# Graph Memory Audit

## Root Finding: Coordinate Reference System Mismatch

**All GeoJSON files use EPSG:4326 (WGS84 lat/lon degrees).**

The entire neighborhood for Archetype 06 (851 buildings) spans:
- X: -73.850358 to -73.844389 (**0.006 degrees** = ~530 meters)
- Y: 40.871388 to 40.875882 (**0.004 degrees** = ~500 meters)

**Maximum pairwise distance between any two buildings: 0.007 degrees**

The graph construction code applies distance thresholds of **30 meters** and **50 meters** to coordinates measured in **degrees**. Since all pairwise distances are < 0.01 degrees, and the threshold is 30.0, **every single building is connected to every other building**.

This creates **fully connected graphs** — the worst possible graph density.

## Graph Statistics

| Archetype | Nodes | Edges | Edges/Node | % Connected | edge_attr (MB) |
|-----------|-------|-------|------------|-------------|----------------|
| 00 | 623 | 387,506 | 622 | 100.0% | 10.3 |
| 01 | 593 | 351,056 | 592 | 100.0% | 9.4 |
| 02 | 606 | 366,630 | 605 | 100.0% | 9.8 |
| 03 | 237 | 55,932 | 236 | 100.0% | 1.5 |
| 04 | 692 | 478,172 | 691 | 100.0% | 12.8 |
| 05 | 702 | 492,102 | 701 | 100.0% | 13.1 |
| **06** | **851** | **723,350** | **850** | **100.0%** | **19.3** |
| 07 | 242 | 58,322 | 241 | 100.0% | 1.6 |
| 08 | 258 | 66,306 | 257 | 100.0% | 1.8 |
| 09 | 357 | 127,092 | 356 | 100.0% | 3.4 |
| 10 | 467 | 217,622 | 466 | 100.0% | 5.8 |

## Per-Graph Tensor Shapes

### Minimum Graph (Archetype 03)
- `graph.x.shape`: [237, 12]
- `graph.edge_index.shape`: [2, 55,932]
- `graph.edge_attr.shape`: [55,932, 7]
- Memory Footprint: ~1.7 MB

### Median Graph (Archetype 09)
- `graph.x.shape`: [357, 12]
- `graph.edge_index.shape`: [2, 127,092]
- `graph.edge_attr.shape`: [127,092, 7]
- Memory Footprint: ~3.8 MB

### Maximum Graph (Archetype 06)
- `graph.x.shape`: [851, 12]
- `graph.edge_index.shape`: [2, 723,350]
- `graph.edge_attr.shape`: [723,350, 7]
- Memory Footprint: ~20.9 MB

## Batch-Level Memory (batch_size=32)

When PyG collates multiple graphs into a single batch, all edges are concatenated:

| Scenario | Batch Edges | edge_attr Tensor (GB) | GATv2 Intermediates (GB) | **Total Peak (GB)** |
|----------|-------------|----------------------|--------------------------|---------------------|
| Worst (29 × arch06) | 20,977,150 | 0.5 | 33.3 | **33.8** |
| Typical mixed | ~10,000,000 | 0.3 | 15.9 | **16.2** |
| Best (29 × arch03) | 1,622,028 | 0.04 | 2.6 | **2.6** |

**Available system RAM: ~11 GB. Even the "typical" case exceeds this by 50%.**
