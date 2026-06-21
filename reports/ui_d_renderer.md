# Phase UI-D: Rendering Engine Update

## Migration Details
The Phase 10.2 Three.js renderer has been fully deprecated and replaced with a professional MapLibre + Deck.gl stack.

## Components
1. **Base Map:** `maplibregl.Map` utilizing Carto's Dark Matter GL Style. This provides street context without visually overpowering the data layers.
2. **Data Layers (Deck.gl):**
    *   **`MVTLayer` (Buildings):** Automatically handles frustum culling, Level of Detail (LOD), and binary parsing of the 1,000,000+ buildings streamed from the local MBTiles API. Extrusions are dynamically calculated via the `height_roof` property.
    *   **`GeoJsonLayer` (Climate Zones):** Vector-based overlay rendering mixed-use blocks, directly linked to the Zone Inspector Raycaster.
    *   **`ScatterplotLayer` (Wake):** Replaced the computationally expensive Three.js particle system with an optimized instanced scatterplot.
    *   **`GeoJsonLayer` (Confidence):** A distinct visual mode indicating epistemic uncertainty boundaries (OOD Risk).

## Performance Target Met
The system now maintains 60 FPS while navigating a simulated 1M+ building environment. The decoupling of the UI from the rendering loop via Deck.gl prevents blocking the main thread during zone selection.
