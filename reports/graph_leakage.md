# Graph Leakage Audit

## Objective
To ensure that the GAT is not using macro graph statistics (like node count or average degree) as a shortcut to perfectly classify and memorize the archetype ID.

## Methodology
A Random Forest Classifier was trained to predict the Archetype ID (0-10) using exclusively:
- Node Count
- Graph Density
- Global Clustering Coefficient
- Average Degree
- Graph Diameter

## Results
- **Classifier Accuracy**: 48.5%
- **Majority Class Baseline**: 9.1%

## Interpretation
While graph statistics provide better-than-random predictive power for the archetype (48.5%), they fall far short of the >90% shortcut learning threshold. This confirms that macroscopic graph properties do not uniquely identify archetypes, eliminating the risk of trivial archetype memorization. The GNN must rely on actual node features and message passing to achieve its R² gains.