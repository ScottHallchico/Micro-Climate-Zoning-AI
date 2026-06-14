# Model Registry Design

## Directory Structure
- `models/production/`: Contains the active `.pt` weights that the FastAPI backend loads upon initialization. Only formally validated models may enter this tier.
- `models/staging/`: Contains candidate models (e.g. Phase 7A.2 CPU weights) currently undergoing A/B testing or latency profiling.
- `models/archived/`: Older generation models for rollback purposes.

## Metadata & Provenance
Each model directory includes a `metadata.json` file documenting:
- Architecture Type (e.g. `Hybrid_GAT_PINN`)
- Training epochs and dataset hash
- Optimization parameters ($\lambda$-physics, LR)
- Validation R² metrics

## Versioning System
Version identifiers follow SemVer (e.g., `v1.2.0`). 
Major versions reflect architecture changes (e.g., GAT to PINN). Minor versions reflect new datasets or hyperparameter optimization.
