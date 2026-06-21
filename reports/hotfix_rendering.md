# Hotfix Report: Building Rendering Validation

## Issue
Because the MBTiles endpoint returned 404s, `deck.MVTLayer` was unable to decode any building geometries, resulting in a completely empty city layout.

## Cause
A direct dependency on the missing `buildings.mbtiles` archive and the malfunctioning FastAPI route.

## Resolution
- With the `/tiles/buildings/{z}/{x}/{y}` route restored and generating `application/x-protobuf` encoded payloads, the `MVTLayer` now correctly ingests the vector tiles.
- Extrusions are automatically driven by the `height_roof` property decoded from the tiles.
- NYC's full 1-Million building footprint is now natively streamed and hardware-accelerated directly inside the workstation.
