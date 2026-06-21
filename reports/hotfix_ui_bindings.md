# Hotfix Report: Dead UI Controls

## Issue
The "Ventilation Corridors" toggle and the "Run L1 Surrogate" button were visually present in the CSS/HTML but unresponsive to user clicks.

## Cause
The JavaScript event listeners mapping these DOM elements to application state were accidentally omitted during the rewrite of `app.js` in Phase UI-D.

## Resolution
- Added `DOM.lCorr.addEventListener` in `app.js` to toggle `layers.corridors` state and invoke `renderLayers()`.
- Implemented the `deck.LineLayer` inside the `renderLayers()` loop to visualize the corridors when enabled.
- Added `DOM.btnRun.addEventListener` to simulate the Surrogate Model inference delay and visually update the button state from "Inferring Field..." back to "Run L1 Surrogate".
