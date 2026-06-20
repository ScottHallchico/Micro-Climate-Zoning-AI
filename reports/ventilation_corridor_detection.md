# Ventilation Corridor Detection (Phase 8A.2)

## Design Motivation
The threshold-based `Z1` class (`VEI > 1.5`) often creates fragmented patches that lack spatial continuity. True ventilation corridors are continuous flow structures that channel wind through urban canyons over significant distances.

## Detection Algorithm
1. **Pedestrian Layer Filtering:** Restricted analysis to the first 10m above ground level.
2. **Velocity Threshold:** Extracted the top 15% velocity magnitude points per archetype.
3. **Streamline-Connected Components:** Clustered points using DBSCAN (`eps=15m`, `min_samples=5`).
4. **Morphological Filtering:** Corridors must satisfy:
   - Length $\ge 50m$
   - Width $\ge 5m$
5. **Polygon Extraction:** Generated continuous footprints via spatial buffering.
