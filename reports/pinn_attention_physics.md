# Physics-Informed Attention Analysis

By enforcing Navier-Stokes residuals during backpropagation, the GAT layers inside the Hybrid model learn a fundamentally different attention distribution compared to the Pure GAT.

- **Pure GAT**: Attends heavily to the nearest upstream buildings.
- **Hybrid GAT-PINN**: Attends to upstream buildings, but also establishes strong attention links along the 'canyon axes' where flow channeling (Venturi effect) is constrained by continuity. The physics loss literally taught the graph to look for pressure-driven channeling corridors.