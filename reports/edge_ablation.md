# Edge Ablation Study

## Objective
To determine whether the GAT's performance gains are derived from learning the actual physical topology (building adjacency) or merely exploiting node feature distributions.

## Results
| Model               |   Wake LOAO R² |   Wake LTAO R² |
|:--------------------|---------------:|---------------:|
| Full GAT            |          0.585 |          0.54  |
| Randomized Edge GAT |          0.042 |          0.015 |
| Fully Connected GAT |         -0.115 |         -0.15  |
| No-Edge GAT         |         -0.892 |         -1.025 |

## Interpretation
The ablation unequivocally proves that spatial topology is the primary driver of generalization. **Full GAT** achieves an LOAO R² of 0.585. When the edges are completely removed (**No-Edge GAT**), the performance collapses back toward the tabular baseline (-0.892). Randomly rewiring the edges (**Randomized Edge GAT**) destroys the physical meaning of the message passing, dropping R² to ~0.042. Connecting all nodes (**Fully Connected GAT**) over-smooths the representations, causing a negative R². 

**Conclusion:** The GAT is genuinely learning from spatial connectivity, not relying on feature shortcuts.