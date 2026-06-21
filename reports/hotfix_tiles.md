# Hotfix Report: Tile Server Failure

## Issue
The FastAPI tile server returned continuous `404 Not Found` errors when the browser requested tiles from `/tiles/buildings/{z}/{x}/{y}`.

## Cause
Two compounding defects caused the 404 stream:
1. The `api_server.py` file missed the actual `@app.get("/tiles/buildings/{z}/{x}/{y}")` route definition during the previous rewrite.
2. The `tippecanoe` tile generator was executed incorrectly (without the `.venv` path), resulting in an empty or missing `buildings.mbtiles` SQLite database.

## Resolution
- The FastAPI router in `src/services/api_server.py` was fully re-implemented to correctly decode XYZ coordinates to TMS-Y and stream the gzip-compressed Protobuf chunks from SQLite.
- The `tippecanoe` process was restarted via `.venv/bin/tippecanoe` to generate the true, populated `buildings.mbtiles` from `BUILDING_CLEAN.geojson`. 
- The endpoint now correctly resolves tile data or returns a valid `204 No Content` for empty regions.
