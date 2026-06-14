# GAT Architecture Audit

## EdgeEnhancedGAT Specification

| Parameter | Value |
|-----------|-------|
| Number of Layers | 2 (GATv2Conv) |
| Hidden Dimensions | 32 |
| Attention Heads | 2 |
| Input Node Dim | 12 |
| Input Edge Dim | 7 |
| Output Dim | 3 (multi-target) or 1 (wake specialist) |

## Parameter Count

- Conv1: GATv2Conv(12 → 32, heads=2, edge_dim=7)
  - W_src: 12 × 32 × 2 = 768
  - W_dst: 12 × 32 × 2 = 768
  - W_edge: 7 × 32 × 2 = 448
  - att: 32 × 2 = 64
  - bias: 32
  - Subtotal: ~2,080

- Conv2: GATv2Conv(32 → 32, heads=2, edge_dim=7)
  - Similar structure: ~2,240

- FC: Linear(32, 3) = 99

- **Total Parameters: ~4,419**

## Verdict

The model is **extremely small** (< 5,000 parameters). The architecture itself is NOT the memory problem.

The problem is the **number of edges flowing through the attention mechanism**. GATv2Conv computes per-edge attention scores. With 723,350 edges per graph and 29 graphs per batch, the attention computation processes **~21 million edges**, requiring intermediate tensors of ~33 GB.
