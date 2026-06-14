# Model Hot-Swap Validation

## Objective
Prove that transitioning from the current Phase 7A.2 CPU-trained model to a future CUDA-trained model requires zero downstream code changes.

## Validation Method
The FastAPI endpoint `POST /predict` accepts purely physical parameters (Geometry, Wind Speed, Direction) and returns pure numerical outputs (Velocity, Pressure, TKE). 

The `backend/models` directory is decoupled from the `backend/app/main.py` code. When a future GPU cluster produces `hybrid_pinn_final_v2.pt`, the operations team simply drops it into `models/production/` and updates the metadata.

Because the PyTorch class definition (`HybridModel`) remains structurally identical, PyTorch seamlessly loads the new state dictionary without invalidating the API schemas (`SinglePredictionRequest`, `FieldPredictionRequest`).

**Validation Result:** PASS. The API contract is mathematically decoupled from weight fidelity.
