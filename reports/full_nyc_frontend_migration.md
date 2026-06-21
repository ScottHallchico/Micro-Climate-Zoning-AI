# NYC Dataset Frontend Migration Plan

## 1. Recommended Source File
**Source Dataset:** `data/raw/BUILDING_20260602.geojson`

## 2. Expected Building Count
**Total Features:** 1,082,945 buildings

## 3. Preprocessing Requirements
Rendering a dataset of this magnitude in a web browser requires significant preprocessing. It cannot be directly loaded as a single JSON file.
*   **Vector Tiling:** The GeoJSON must be converted into vector tiles (e.g., using `tippecanoe` to create `.mbtiles`) to stream data progressively.
*   **Spatial Indexing:** Implementation of robust spatial clustering (e.g., QuadTree, R-tree) for fast querying and occlusion culling.
*   **Level of Detail (LOD):** Simplification of complex building geometries when viewed from high altitudes.
*   **Format Conversion:** Transformation to optimized binary formats such as Deck.gl's 3D Tiles, Arrow, or binary FlatGeobuf.

## 4. Estimated Browser Performance Impact
Attempting to parse and render a 962 MB GeoJSON file with 1M+ polygons entirely in the main browser thread will result in:
*   **OOM Crashes:** Exceeding maximum browser heap memory allocations.
*   **Thread Blocking:** Complete unresponsiveness of the UI while parsing the JSON string.
*   **Frame Drops:** Extreme lag (sub-1 FPS) if the WebGL buffer processes uninstanced/unoptimized vertices for 1 million distinct 3D extrusions simultaneously.

A migration to a streaming, tiled architecture is mandatory for Production Authorization of the full dataset.
