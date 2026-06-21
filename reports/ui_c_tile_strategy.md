# Phase UI-C: Tile Strategy Analysis

## Option A: 500m x 500m Gridded Tiles
- **Estimated Tiles for NYC:** ~3,120 tiles
- **Buildings per Tile (Avg):** ~347 (Peaks up to 2,500 in Midtown Manhattan)
- **Estimated Memory per Tile:** 1 MB – 5 MB
- **Expected FPS:** 60 FPS (Loading chunks causes minor stutter, moderate latency)
- **Verdict:** Acceptable, but rigid. Doesn't handle variable LOD natively.

## Option B: 250m x 250m Gridded Tiles
- **Estimated Tiles for NYC:** ~12,480 tiles
- **Buildings per Tile (Avg):** ~86 (Peaks up to 600)
- **Estimated Memory per Tile:** 250 KB – 1 MB
- **Expected FPS:** 60 FPS
- **Verdict:** Too fragmented. Leads to an excessive number of HTTP requests and high WebGL draw call overhead unless heavily batched.

## Option C: Mapbox Vector Tiles (MVT) / 3D Tiles
- **Architecture:** Quad-tree based adaptive zoom levels.
- **Buildings per Tile:** Dynamically scales. Zoom 15 provides roughly block-level granularity, while Zoom 10 provides city-level generalized polygons.
- **Estimated Memory per Tile:** Highly compressed binary protobuf format (usually < 200 KB per tile).
- **Expected FPS:** 60+ FPS (Zero stuttering due to multithreaded web-worker decoding).
- **Verdict:** The industry standard for handling 1M+ polygons. Supports automatic Level of Detail (LOD) and frustum culling.

## Recommendation
**Option C (MVT / 3D Tiles)** is the only production-ready strategy. Using rigid 500m/250m custom JSON grids is an anti-pattern for datasets exceeding 100,000 features.
