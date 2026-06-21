# Workstream 4: Ventilation Corridor Integration

## Objective
Visualize the actual airflow corridor geometries derived from the topology engine.

## Dataset Audit
- **Source:** `data/ventilation_corridors_v2.geojson`
- **Schema:** `corridor_id`, `corridor_length`, `corridor_width`, `mean_velocity`, `persistence_annual`

## Integration Validation
- Integrated via `deck.GeoJsonLayer` mapping thick cyan lines (`[56, 189, 248, 150]`) utilizing the exact geometric traces of the urban street grid airflow.
- Corridors are interactively pickable, exposing exact velocity and dimensions inside the HUD.
