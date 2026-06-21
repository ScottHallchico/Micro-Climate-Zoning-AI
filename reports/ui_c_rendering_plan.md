# Phase UI-C: Production Rendering Plan

## Architecture Blueprint

This document serves as the implementation blueprint for scaling the Climate Zoning AI frontend to handle the complete 1,082,945 building dataset of New York City.

### 1. Recommended Renderer
**Deck.gl (via MapLibre GL JS integration)**
- Pure Three.js struggles with frustum culling, hierarchical LOD, and spatial indexing out-of-the-box for 1M+ extruded polygons.
- Deck.gl's `MVTLayer` or `PolygonLayer` is heavily optimized for rendering millions of vertices using instanced draw calls and Web Workers for binary parsing.

### 2. Recommended Tile Architecture
**Mapbox Vector Tiles (MVT) via Tippecanoe**
- The 962 MB `BUILDING_20260602.geojson` must be processed via `tippecanoe` into an `.mbtiles` package.
- The tiles will be served statically via a lightweight tile server (like `mbtiles-server` or Martin) hosted alongside the FastAPI backend.
- Max zoom level: 16 (for block-level precision). Min zoom level: 11 (for zoomed-out city overview).

### 3. Estimated Memory Footprint
- **Current Raw Approach (GeoJSON):** ~2 GB RAM parsing overhead, browser crash likely.
- **Proposed MVT Approach:** ~100 MB to 200 MB VRAM dynamically managed by Deck.gl. Only features within the active camera frustum and current zoom level are loaded and held in memory.

### 4. Estimated Visible Buildings
- **Worst-case Camera Angle (Horizon view of entire NYC):** Automatically generalized by MVT LOD, rendering ~10,000 to 20,000 merged polygon representations instead of 1,000,000 distinct block extrusions.
- **Normal Operating View (Planners looking at Neighborhoods):** 5,000 to 15,000 distinct buildings loaded dynamically in real-time.

### 5. Expected Browser Performance
- **Target FPS:** Steady 60 FPS on standard modern laptops.
- **Initial Load Time:** < 1 second for the base tiles (vs. several minutes for the raw 962 MB file).

### Conclusion
By adopting Deck.gl + MVT, the Climate Zoning AI workstation will successfully render the entire New York City dataset interactively, leaving plenty of compute headroom to run the CFD/Surrogate particle effects and interactive UI panels smoothly.
