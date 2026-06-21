# Hotfix Report: Canvas Initialization Failure

## Issue
The Phase 10.3 acceptance test revealed a fatal crash on startup (`ReferenceError`) because Deck.gl could not find the target canvas element in the DOM.

## Cause
During the migration from Three.js to MapLibre+Deck.gl in `index.html`, the explicit `<canvas id="deck-canvas"></canvas>` element was omitted. Deck.gl requires this element to mount its WebGL context when overlaying on MapLibre.

## Resolution
- Injected `<canvas id="deck-canvas"></canvas>` inside the `#map-container` div in `src/frontend/index.html`.
- Verified Deck.gl mounts successfully without runtime crashes.
