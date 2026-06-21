# Workstream 1: Building Tile Validation

## Objective
Investigate why the NYC building extrusions were not visible and restore the layer.

## Audit Findings
- **MVTLayer Loading:** The `deck.MVTLayer` was loaded, but the endpoint it requested (`/tiles/buildings/{z}/{x}/{y}`) returned HTTP 404.
- **MBTiles existence:** The `buildings.mbtiles` file was successfully generated inside `data/tiles/` containing over 1 million features.
- **Root Cause:** The `api_server.py` script was missing the `/tiles/buildings/{z}/{x}/{y}` FastAPI route definition, preventing the SQLite payload from being served.
- **Attribute Schema:** The protobuf payload successfully decodes the `height_roof` property per feature.

## Resolution
The backend tile streaming route was fully restored, decoding the standard XYZ request into inverted TMS format and directly querying the SQLite database. Building features are now fully visibly rendering and extruded based on the `getElevation: f => f.properties.height_roof` function.
