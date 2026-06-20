# Dashboard Integration (Phase 8A.2 Step 7)

## Frontend Updates
The `App.tsx` has been updated to ingest the new `ventilation_corridors.geojson`.
Corridors are rendered with categorical coloring:
- **Permanent Corridor:** High opacity blue.
- **Seasonal Corridor:** Medium opacity teal.
- **Conditional Corridor:** Low opacity green.
