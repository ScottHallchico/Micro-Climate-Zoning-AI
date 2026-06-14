# Progressive Scaling Audit

## Methodology

Unable to execute progressive scaling test because the graph construction itself produces fully-connected graphs due to the CRS mismatch. Even a single graph from Archetype 06 produces 723,350 edges.

## Theoretical Scaling

| Graph Count | Est. Batch Edges | Est. Peak RAM (GB) | Predicted Status |
|-------------|------------------|--------------------|--------------------|
| 1 (arch03) | 55,932 | ~0.15 | SUCCESS |
| 1 (arch06) | 723,350 | ~1.2 | SUCCESS |
| 5 (mixed) | ~2,000,000 | ~4.0 | SUCCESS |
| 10 (mixed) | ~4,000,000 | ~8.0 | MARGINAL |
| 25 (mixed) | ~10,000,000 | ~16.0 | **OOM KILL** |
| 29 (train set) | ~12,000,000 | ~20.0 | **OOM KILL** |

## Key Finding

The failure is deterministic: a batch containing more than ~6 million edges will exhaust 11 GB of available RAM. With fully-connected graphs of 500–850 nodes, this threshold is reached with approximately 10–15 graphs per batch.

Since `batch_size=32` and the training split contains 29 graphs, the first batch loads all 29 graphs at once, creating a single mega-graph with ~12 million edges.

## DataLoader Settings

The current configuration uses default PyG DataLoader settings:
- `num_workers`: 0 (default — safe)
- `pin_memory`: False (default — safe)
- `persistent_workers`: False (default — safe)

These settings are **NOT contributing** to the OOM. The problem is purely the data volume entering the forward/backward pass.
