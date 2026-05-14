# Implementation Plan: Mumbai 3D District Map

## Overview

Replace the synthetic 4×4 grid of city blocks with geographically accurate Mumbai district polygons. The implementation proceeds in layers: static data first, then backend extensions, then the two new frontend shared modules, then updates to the existing dashboard and 3D map, and finally tests. Every step integrates immediately so there is no orphaned code.

## Tasks

- [x] 1. Create GeoJSON data file and update CSV
  - [x] 1.1 Create `src/dashboard/data/mumbai_districts.geojson` with ≥24 Mumbai district polygons
    - Add top-level `metadata` object with `source`, `license`, and `retrieved_date` fields
    - Each Feature must have `id` (district slug), `properties.name`, `properties.district`, `properties.area_km2`, and a `geometry` of type `"Polygon"` or `"MultiPolygon"`
    - Coordinates in WGS 84 decimal degrees with ≥4 decimal places of precision
    - Polygons must be non-overlapping and cover the Mumbai Metropolitan Region within the bounding box 18.89°N–19.27°N, 72.77°E–73.00°E
    - _Requirements: 1.1, 6.1, 6.2, 6.3_

  - [x] 1.2 Update `data/actual_blocks.csv` with district rows
    - Replace the 20 `BLK-XX` rows with ≥24 district rows
    - Add columns: `district_name`, `centroid_lat`, `centroid_lon`, `area_km2`
    - Update `block_id` values to district slugs matching GeoJSON Feature `id` values exactly (case-sensitive)
    - Retain all existing columns (`col`, `row`, `osm_building_count`, `osm_road_count`, etc.) for backward compatibility; assign `col` ∈ [0, 4] and `row` ∈ [0, 3] to each district
    - _Requirements: 4.1, 4.3_

- [x] 2. Extend backend `api.py` with GeoJSON loading and district fields
  - [x] 2.1 Add GeoJSON startup loading and `_district_geometry_cache` to `api.py`
    - Add module-level `_district_geometry_cache: dict[str, dict]` and `GEOJSON_PATH` constant pointing to `src/dashboard/data/mumbai_districts.geojson`
    - At module load time, read the GeoJSON file, populate `_district_geometry_cache` keyed by Feature `id`, and log `metadata.source` + `metadata.retrieved_date` at INFO level
    - If the file is absent or `metadata` fields are missing/empty, log an ERROR identifying the missing field(s) and continue with an empty cache
    - _Requirements: 6.4, 6.5_

  - [x] 2.2 Extend `_build_actual_blocks()` to include district geographic fields
    - Add `district_name`, `centroid_lat`, `centroid_lon`, `area_km2`, and `geometry` (looked up from `_district_geometry_cache` by `block_id`) to each block dict returned by `_build_actual_blocks()`
    - Use `_as_float` for `centroid_lat`, `centroid_lon`, `area_km2`; use `row.get("district_name", "")` for the name
    - Ensure `_classify_actual_zone()` continues to work correctly with the new district rows (no changes to classification logic required, only verify it handles the new CSV columns without error)
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x] 2.3 Write Python unit tests for backend district changes (`tests/test_district_map.py`)
    - Test zone classifier exhaustiveness: verify `_classify_actual_zone` returns one of the four valid zone class strings for all boundary combinations of `osm_road_count`, `green_cover_pct`, `lambda_p`, `max_height_m`
    - Test default fallback values: verify a district row with all-empty morphology fields produces `BASELINE_UNCHANGED` with the specified default metric values per Requirement 4.5
    - Test GeoJSON metadata logging: verify API startup logs `metadata.source` and `metadata.retrieved_date` at INFO level when the file is present
    - Test missing GeoJSON at startup: verify a missing GeoJSON file logs an ERROR and the `/v1/health` endpoint still returns HTTP 200
    - Test `District_Block` serialisation: verify `_build_actual_blocks()` output is JSON-serialisable and that `json.loads(json.dumps(block)) == block` for all fields
    - _Requirements: 4.2, 4.4, 4.5, 6.4, 6.5_

- [ ] 3. Checkpoint — backend data layer complete
  - Ensure all Python tests pass. Verify `GET /v1/data/actual` returns ≥24 blocks with `district_name`, `centroid_lat`, `centroid_lon`, `area_km2`, and `geometry` fields. Ask the user if questions arise.

- [x] 4. Create `src/dashboard/geojson-loader.js` shared module
  - [x] 4.1 Implement `loadMumbaiGeoJSON(url)` in `geojson-loader.js`
    - Fetch the GeoJSON file; on HTTP 4xx/5xx or network error emit `console.error` with path + failure description and return `null`
    - On JSON parse failure emit `console.error` with path + "JSON syntax error" and return `null`
    - On success, call `validateGeoJSONFeature` for each Feature, collect valid features, call `checkOverlaps`, and return `{ features, metadata }`
    - If all features are invalid after validation, return `null`
    - _Requirements: 1.2, 1.3_

  - [x] 4.2 Implement `validateGeoJSONFeature(feature, index)` in `geojson-loader.js`
    - Return `true` if `feature.geometry.type` is `"Polygon"` or `"MultiPolygon"` AND `feature.properties` has a non-empty `name` or `district` string
    - Return `false` and emit `console.warn` with feature index + missing field name for any failing check
    - _Requirements: 1.2_

  - [x] 4.3 Implement `checkOverlaps(features)` in `geojson-loader.js`
    - Perform pairwise overlap check; emit `console.warn` with both district names and overlap percentage for any pair where shared area exceeds 0.01% of the smaller polygon's area
    - _Requirements: 1.4_

  - [x] 4.4 Write JavaScript unit tests for `geojson-loader.js` (`src/dashboard/tests/geojson-loader.test.js`)
    - Test `validateGeoJSONFeature` with: valid feature, feature missing geometry, feature with wrong geometry type (`"Point"`), feature missing both `name` and `district`, feature with empty string `name`
    - Test `loadMumbaiGeoJSON` with: successful load returning correct feature count, HTTP 404 returning `null`, malformed JSON returning `null`, all-invalid features returning `null`
    - _Requirements: 1.2, 1.3_

- [ ] 5. Create `src/dashboard/district-projector.js` shared module
  - [x] 5.1 Implement `projectToSVG(lat, lon, svgWidth, svgHeight)` in `district-projector.js`
    - Use equirectangular projection anchored to `MUMBAI_BBOX = { minLat: 18.89, maxLat: 19.27, minLon: 72.77, maxLon: 73.00 }`
    - Formula: `x = ((lon - minLon) / (maxLon - minLon)) * svgWidth`, `y = ((maxLat - lat) / (maxLat - minLat)) * svgHeight`
    - Export `MUMBAI_BBOX` constant alongside the function
    - _Requirements: 2.1, 2.3_

  - [x] 5.2 Implement `projectToScene(lat, lon)` in `district-projector.js`
    - Normalise Mumbai extent to [−38, +38] on X and Z axes
    - Formula: `x = ((lon - minLon) / (maxLon - minLon) - 0.5) * 2 * 38`, `z = ((maxLat - lat) / (maxLat - minLat) - 0.5) * 2 * 38`
    - _Requirements: 3.1_

  - [x] 5.3 Implement `resolveCollisions(blocks, minSeparation = 3)` in `district-projector.js`
    - Iterate all pairs; if two projected centroids are closer than `minSeparation` scene units, translate the block with the smaller `area_km2` by the minimum vector to achieve separation
    - Log `console.log` with both district names, original separation, and applied translation vector
    - Mutate the blocks array in-place
    - _Requirements: 3.4_

  - [x] 5.4 Implement `computeFootprint(areaKm2, maxAreaKm2)` in `district-projector.js`
    - Formula: `Math.max(4, Math.min(14, Math.sqrt(areaKm2 / maxAreaKm2) * 14))`
    - Guard against `maxAreaKm2 <= 0` by returning minimum footprint of 4
    - _Requirements: 3.3_

  - [ ] 5.5 Write JavaScript unit tests for `district-projector.js` (`src/dashboard/tests/district-projector.test.js`)
    - Test `projectToSVG`: all four corners of Mumbai bounding box, a centroid coordinate, exact bounding box edge values
    - Test `projectToScene`: corners produce ±38, interior coordinate produces value within [−38, +38]
    - Test `computeFootprint`: `area = 0` returns 4, `area = maxArea` returns 14, `area = maxArea/4` returns 7, `area > maxArea` clamps to 14, `maxAreaKm2 = 0` returns 4
    - Test `resolveCollisions`: no-collision case leaves positions unchanged, single collision achieves ≥3 unit separation, multiple collisions all resolved
    - _Requirements: 2.1, 3.1, 3.3, 3.4_

  - [ ] 5.6 Write property-based tests for projector functions (`src/dashboard/tests/properties.test.js`)
    - **Property 4: SVG projection stays within viewport** — for any (lat, lon) within Mumbai bounding box and any (svgWidth, svgHeight) in [100, 2000], `projectToSVG` returns x ∈ [0, svgWidth] and y ∈ [0, svgHeight]
    - **Property 7: Footprint formula invariant** — for any `areaKm2 ≥ 0` and `maxAreaKm2 > 0`, `computeFootprint` returns a value in [4, 14]
    - Each property runs a minimum of 100 iterations using fast-check
    - **Validates: Requirements 2.1, 2.3, 3.3**

- [ ] 6. Update `dashboard.js` to render district polygons
  - [ ] 6.1 Add GeoJSON loading to `dashboard.js` initialisation
    - Import `loadMumbaiGeoJSON` from `geojson-loader.js` and `projectToSVG` from `district-projector.js`
    - In `initializeBlocks()` (or `DOMContentLoaded` handler), call `loadMumbaiGeoJSON('./data/mumbai_districts.geojson')` and store result in `state.geoJSONFeatures`
    - If result is `null`, set `state.geoJSONFeatures` to `[]`
    - _Requirements: 1.1, 1.3_

  - [ ] 6.2 Implement `renderDistrictMap()` and update `renderMap()` branching in `dashboard.js`
    - Update `renderMap()` to call `renderDistrictMap()` when `state.geoJSONFeatures.length > 0`, otherwise call the existing rect-based rendering (renamed to `renderSyntheticGrid()`)
    - In `renderDistrictMap()`: for each block, find the matching GeoJSON feature by `block.id`; project all polygon ring coordinates using `projectToSVG`; render `<polygon>` or `<path>` SVG element with `data-district-name` attribute, zone colour fill, and `opacity="0.85"`
    - Selected district gets a 2.5px stroke in `var(--accent-primary)` and the `block-pulse` CSS animation class
    - Render centroid label at projected centroid using `.block-label` CSS class, truncated to 12 characters
    - If a polygon has fewer than 3 projected coordinate pairs, skip it and emit `console.warn` with district name and actual count
    - If all features are invalid, show an error banner inside `.city-map` and fall back to `renderSyntheticGrid()`
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ] 6.3 Extend `normalizeDashboardBlock()` in `dashboard.js` to preserve geographic fields
    - Add `geometry`, `centroid_lat`, `centroid_lon`, `area_km2`, `district_name` to the spread in `normalizeDashboardBlock()` so these fields survive the normalisation pass
    - Verify `saveBlocksToSharedState()` already serialises the full block object (no change needed if it does `JSON.stringify(state.blocks)`)
    - _Requirements: 4.1, 5.1, 5.3_

  - [ ] 6.4 Write JavaScript unit tests for `dashboard.js` district rendering (`src/dashboard/tests/dashboard-district.test.js`)
    - Test `renderDistrictMap()` SVG output contains `<polygon>` or `<path>` elements with correct `data-district-name` and fill attributes
    - Test label truncation: names of length 0, 12, 13, and 50 characters
    - Test fallback to `renderSyntheticGrid()` when `state.geoJSONFeatures` is empty
    - Test error banner appears when all features are invalid
    - _Requirements: 2.2, 2.6, 2.8_

  - [ ] 6.5 Write property-based tests for GeoJSON round-trip and District_Block serialisation (`src/dashboard/tests/properties.test.js`)
    - **Property 2: GeoJSON round-trip fidelity** — for any valid GeoJSON FeatureCollection, `JSON.parse(JSON.stringify(fc))` produces the same feature count and coordinates within 1×10⁻⁹ degrees
    - **Property 10: District_Block JSON round-trip fidelity** — for any District_Block with arbitrary field values including a `geometry` polygon, `JSON.parse(JSON.stringify(block))` equals the original with coordinate precision ≤1×10⁻⁹ degrees
    - Each property runs a minimum of 100 iterations using fast-check
    - **Validates: Requirements 1.5, 4.4**

- [ ] 7. Checkpoint — 2D dashboard district rendering complete
  - Ensure all JavaScript tests pass. Open `index.html` and verify district polygons render in the `.city-map` container with correct zone colours and labels. Ask the user if questions arise.

- [ ] 8. Update `map3d.js` to use geographic positioning
  - [ ] 8.1 Add `getBlockPosition(block)` and import projector functions in `map3d.js`
    - Import `projectToScene` from `district-projector.js`
    - Implement `getBlockPosition(block)`: if `Number.isFinite(block.centroid_lat) && Number.isFinite(block.centroid_lon)`, return `projectToScene(block.centroid_lat, block.centroid_lon)`; otherwise return legacy `col`/`row` grid position (`{ x: xOffset + block.col * spacing, z: zOffset + block.row * spacing }`)
    - _Requirements: 3.1, 3.2, 5.2, 5.4_

  - [ ] 8.2 Update `createBlocks()` in `map3d.js` to use geographic positioning and footprint sizing
    - Call `resolveCollisions(blocks)` (imported from `district-projector.js`) before positioning meshes
    - Use `getBlockPosition(block)` instead of the fixed grid offset for each mesh position
    - Compute `maxAreaKm2 = Math.max(...blocks.map(b => b.area_km2 || 0))` and use `computeFootprint(block.area_km2, maxAreaKm2)` for the ground footprint geometry dimensions
    - Fall back to footprint of 7 (current default) if `area_km2` is absent or zero
    - _Requirements: 3.2, 3.3, 3.4_

  - [ ] 8.3 Extend `normalizeSharedBlock()` in `map3d.js` to preserve geographic fields
    - Add `geometry`, `centroid_lat`, `centroid_lon`, `area_km2`, `district_name` to the spread in `normalizeSharedBlock()` so these fields survive the normalisation pass
    - In `loadBlocksFromSharedState()`: after normalisation, if `block.geometry` is a non-null object with `type === "Polygon"`, ensure `centroid_lat`/`centroid_lon` are preserved; otherwise the legacy `col`/`row` fallback in `getBlockPosition` handles positioning
    - _Requirements: 5.2, 5.3, 5.4_

  - [ ] 8.4 Add `storage` event listener and loading indicator to `map3d.js`
    - Add `window.addEventListener('storage', ...)` handler: when `event.key === SHARED_BLOCK_STATE_KEY`, reload blocks from localStorage, call `resolveCollisions`, and schedule a mesh re-render within one `requestAnimationFrame`
    - Show `id="map3d-loading"` element (create it if absent) before the API fetch in `initScene()`; remove it after `createBlocks()` completes
    - Record `performance.now()` at `DOMContentLoaded` and at the first `requestAnimationFrame` callback after scene init; log a warning if elapsed time exceeds 3000 ms
    - _Requirements: 5.5, 7.2, 7.3, 7.4_

  - [ ] 8.5 Add `<script nomodule>` browser compatibility message to `3d-map.html`
    - Add `<script nomodule>` tag that injects a visible message "Please upgrade your browser to use this application" into the `#map3d-canvas` container
    - _Requirements: 7.5_

  - [ ] 8.6 Write JavaScript unit tests for `map3d.js` geographic changes (`src/dashboard/tests/map3d-district.test.js`)
    - Test `getBlockPosition` returns scene coordinates when `centroid_lat`/`centroid_lon` are present and finite
    - Test `getBlockPosition` falls back to `col`/`row` grid when centroid fields are missing or non-finite
    - Test `normalizeSharedBlock` preserves `geometry`, `centroid_lat`, `centroid_lon`, `area_km2`, `district_name`
    - Test `loadBlocksFromSharedState` uses geographic positioning when `geometry.type === "Polygon"` is present
    - _Requirements: 3.2, 5.2, 5.4_

  - [ ] 8.7 Write property-based test for shared state geometry preservation (`src/dashboard/tests/properties.test.js`)
    - **Property 11: Shared state preserves geometry through serialisation** — for any array of District_Block records with arbitrary `geometry.coordinates`, calling `saveBlocksToSharedState` then `loadBlocksFromSharedState` returns blocks where every coordinate differs by no more than 1×10⁻⁹ degrees from the original
    - Runs a minimum of 100 iterations using fast-check
    - **Validates: Requirements 5.1, 5.3**

- [ ] 9. Final checkpoint — full integration complete
  - Ensure all Python and JavaScript tests pass. Verify the 2D dashboard renders district polygons, the 3D map positions meshes geographically, and overrides in the 2D dashboard propagate to the 3D map via localStorage. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- The design uses JavaScript (frontend) and Python (backend) — no language selection needed
- Property tests use **fast-check** (MIT licence); install with `npm install --save-dev fast-check` in `src/dashboard/`
- Python tests use **pytest** (already in `pyproject.toml`); add `tests/test_district_map.py`
- Checkpoints ensure incremental validation at the data layer, 2D rendering layer, and full integration
- Properties 2, 4, 7, 10, 11 from the design document are covered by PBT sub-tasks 5.6, 6.5, and 8.7
- The `col`/`row` legacy fields must be retained on every District_Block for backward compatibility with cached localStorage state

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2"] },
    { "id": 3, "tasks": ["2.3", "4.1", "4.2", "4.3"] },
    { "id": 4, "tasks": ["4.4", "5.1", "5.2", "5.3", "5.4"] },
    { "id": 5, "tasks": ["5.5", "5.6", "6.1"] },
    { "id": 6, "tasks": ["6.2", "6.3"] },
    { "id": 7, "tasks": ["6.4", "6.5", "8.1"] },
    { "id": 8, "tasks": ["8.2", "8.3"] },
    { "id": 9, "tasks": ["8.4", "8.5"] },
    { "id": 10, "tasks": ["8.6", "8.7"] }
  ]
}
```
