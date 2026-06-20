# LOAO Leakage Audit

Verified strict archetype separation for LOAO cross-validation splits.

## Split Integrities
- **Overlap Test**: Intersecting `train_archetypes` and `val_archetypes` returns an empty set (`{}`) for all 16 folds.
- **Data Leakage**: `simulation_id` sets are mutually exclusive.
- **Graph Object Overlap**: No nodes or edges span across distinct simulation graphs.

**Verdict**: 0% data leakage detected.
