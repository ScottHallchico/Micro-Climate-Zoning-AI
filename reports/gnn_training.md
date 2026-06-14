# GNN Training Configuration

## Architecture Details
Three benchmark spatial architectures were trained using PyTorch Geometric:
1. **GraphSAGE**: Aggregates neighborhood features using max/mean pooling.
2. **GAT (Graph Attention Network)**: Dynamically weights neighborhood features using multi-headed attention.
3. **GIN (Graph Isomorphism Network)**: Employs MLP-based aggregations to maximize graph discrimination.

## Hyperparameters
* Hidden Dimensions: 64
* Convolutional Layers: 2
* Readout Layer: Global Mean Pooling
* Optimizer: Adam (lr=0.01)
* Loss Function: MSE Loss
* Batch Size: 16
* Evaluation: K-Fold (5-splits), LOAO (11-splits)

## Outcome
All models successfully converged, but GAT demonstrated the highest generalization stability, leveraging attention mechanisms to prioritize upstream aerodynamic obstacles over downstream ones.
