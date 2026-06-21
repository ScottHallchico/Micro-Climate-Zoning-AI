# Phase UI-C: Geometry Complexity Audit

## Overview
This audit examines the vertex density of the building footprints to estimate GPU rendering impact.

## Metrics
- **Total Polygons/MultiPolygons:** 1,082,945
- **Average Vertices per Building:** 8.96
- **Maximum Vertices per Building:** 2,580

## Hardware Estimation
- **Total Vertex Count (Approximate):** ~9.7 million vertices.
- **Rendering Complexity:** The vast majority of buildings are highly simplified (mean ~9 vertices, typical of rectangular footprints). However, the worst-case maximum of 2,580 vertices indicates the presence of a highly complex footprint (likely large campus structures, stadiums, or irregular mega-blocks).
- **GPU Impact:** 
  - Standard WebGL buffers limit draw calls to 65k vertices (for 16-bit indexing) or require OES_element_index_uint.
  - Pushing 9.7M vertices directly into a single Three.js geometry buffer will exceed GPU VRAM constraints on many standard machines and cause sub-10 FPS performance.
  - **Conclusion:** Earcut triangulation and streaming architectures are mandatory to handle the localized dense geometry pockets.
