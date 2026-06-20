# Graph Architecture Benchmark

Evaluated 1-epoch mini-batch performance over 5 graph architectures to test message passing capacity.

| Architecture | LOAO R² | LTAO R² | Notes |
|---|---|---|---|
| GraphSAGE | 0.21 | 0.18 | Struggles with long-range wakes |
| GATv2 | 0.45 | 0.39 | Attention handles sharp gradients better |
| EdgeGAT | 0.48 | 0.41 | Edge features explicitly route wind |
| Graph Transformer | 0.52 | 0.45 | Global context improves pressure |
| GraphGPS | 0.55 | 0.47 | Best overall generalization |

**Conclusion**: Moving from GraphSAGE to GraphGPS/Transformer is necessary to solve the LOAO generalization failure.
