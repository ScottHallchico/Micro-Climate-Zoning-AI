# Requirements Document

## Introduction

This feature replaces the current synthetic 4×4 grid of city blocks in the Micro-Climate Zoning AI dashboard with a geographically accurate 3D map of Mumbai's real administrative districts. Zones displayed in both the 2D scenario dashboard (`index.html`) and the 3D heat map (`3d-map.html`) must correspond to actual Mumbai districts sourced from GeoJSON boundary data with real GPS coordinates. District polygons must be non-overlapping, cover the full Mumbai Metropolitan Region, and carry the same micro-climate zone classifications (WIND_CORRIDOR_CRITICAL, THERMAL_REMEDIATION, DENSITY_ADAPTIVE, BASELINE_UNCHANGED) that the existing pipeline already produces. The feature integrates with the existing governance API, PINN surrogate, and CFD simulation endpoints without breaking any current functionality.

## Glossary

- **District**: An official Mumbai administrative or ward-level geographic unit with a unique name and boundary polygon.
- **GeoJSON**: An open standard format (RFC 7946) for encoding geographic data structures using JSON, used here to store district boundary polygons.
- **GPS Coordinate**: A latitude/longitude pair in the WGS 84 datum representing a real-world geographic position.
- **District_Boundary**: A closed polygon defined by an ordered sequence of GPS coordinates that encloses exactly one Mumbai district without overlapping any other district boundary.
- **District_Map**: The frontend component that renders Mumbai district polygons on a 2D SVG canvas or a 3D Three.js scene, replacing the current synthetic grid.
- **GeoJSON_Loader**: The frontend module responsible for fetching, parsing, and validating the Mumbai GeoJSON file at `src/dashboard/data/mumbai_districts.geojson`.
- **District_Projector**: The module that converts WGS 84 latitude/longitude coordinates into 2D SVG viewport coordinates or 3D scene coordinates.
- **District_Block**: A data record that combines a district's geographic boundary with its micro-climate zone classification and sensor-derived metrics (UHI intensity, surface temperature, etc.), replacing the current `BLK-XX` grid block.
- **Zone_Classifier**: The backend logic (already in `api.py`) that assigns one of the four zone classes to each district based on OSM morphology and weather data.
- **Dashboard**: The existing scenario dashboard at `src/dashboard/index.html` and `src/dashboard/dashboard.js`.
- **3D_Map**: The existing Three.js heat map at `src/dashboard/3d-map.html` and `src/dashboard/map3d.js`.
- **Shared_State**: The `localStorage` key `microclimate-dashboard-blocks-v1` used to synchronise block data between the Dashboard and the 3D_Map.
- **Governance_API**: The FastAPI backend at `src/governance/api.py`, specifically the `/v1/data/actual` endpoint that returns district-level block data.
- **Pretty_Printer**: The serialisation function that converts a District_Block record back into a GeoJSON Feature for round-trip validation.
- **Mumbai_Bounding_Box**: The geographic extent 18.89°N–19.27°N, 72.77°E–73.00°E used as the canonical projection reference for all coordinate transformations.

---

## Requirements

### Requirement 1: Mumbai GeoJSON District Data

**User Story:** As a city planner, I want the map to show real Mumbai district boundaries sourced from GPS coordinates, so that zoning decisions correspond to actual geographic areas.

#### Acceptance Criteria

1. THE GeoJSON_Loader SHALL load the GeoJSON FeatureCollection file `src/dashboard/data/mumbai_districts.geojson` containing at least 24 Mumbai district or ward-level boundary polygons.
2. WHEN the GeoJSON file is loaded, THE GeoJSON_Loader SHALL validate that every Feature contains a `geometry` of type `Polygon` or `MultiPolygon` and a `properties` object with at least a `name` or `district` string field; IF an individual Feature fails this validation, THEN THE GeoJSON_Loader SHALL skip that Feature, log a console warning identifying the Feature index and the missing field, and continue processing the remaining Features.
3. IF the GeoJSON file is missing or the top-level JSON fails to parse, THEN THE GeoJSON_Loader SHALL emit a console error message that includes the attempted file path and the nature of the failure (e.g. HTTP status code or JSON syntax error), and SHALL fall back to the existing synthetic 5×4 grid of 20 blocks so the dashboard remains functional.
4. WHEN the GeoJSON file is loaded successfully, THE GeoJSON_Loader SHALL verify the non-overlapping constraint by checking that for every pair of district polygons the shared interior area is less than 0.01% of the smaller polygon's area; IF any pair exceeds this threshold, THE GeoJSON_Loader SHALL log a console warning identifying the two district names and the overlap percentage.
5. THE GeoJSON_Loader SHALL ensure that for any valid GeoJSON FeatureCollection, parsing then serialising to JSON string then parsing again SHALL produce a FeatureCollection with the same number of Features and coordinate values that differ by no more than 1×10⁻⁹ degrees from the originals (round-trip property).

---

### Requirement 2: District Projection to 2D SVG Map

**User Story:** As a city planner, I want the 2D scenario dashboard to display Mumbai districts as correctly shaped SVG polygons instead of uniform squares, so that I can visually identify real geographic areas.

#### Acceptance Criteria

1. THE District_Projector SHALL convert each district's GPS coordinates to SVG viewport coordinates using a Mercator or equirectangular projection with the Mumbai_Bounding_Box (18.89°N–19.27°N, 72.77°E–73.00°E) as the projection extent.
2. WHEN a district polygon is rendered, THE District_Map SHALL draw it as an SVG `<polygon>` or `<path>` element with the district name as a `data-district-name` attribute and the district's zone class colour as the fill.
3. THE District_Map SHALL scale all district polygons to fit within the existing `.city-map` container without overflow, maintaining the aspect ratio of the Mumbai_Bounding_Box within a ±5% tolerance.
4. WHEN a user clicks a district polygon, THE Dashboard SHALL select that district and display its micro-climate metrics in the existing Block Details panel on the right sidebar.
5. WHILE a district is selected, THE District_Map SHALL highlight the selected polygon with a 2.5px stroke in `var(--accent-primary)` and apply the existing `block-pulse` animation.
6. THE District_Map SHALL render a district name label at the centroid of each polygon using the existing `.block-label` CSS class, truncated to 12 characters if necessary.
7. IF a district polygon has fewer than 3 coordinate pairs after projection, THEN THE District_Projector SHALL skip that polygon, log a console warning that includes the district name and the actual coordinate pair count, so that degenerate geometries do not cause rendering errors.
8. IF the GeoJSON data fails to load or all Features are skipped due to validation errors, THEN THE District_Map SHALL fall back to rendering the existing uniform-square synthetic grid and SHALL display a visible error banner inside the `.city-map` container describing the failure.

---

### Requirement 3: District Projection to 3D Scene

**User Story:** As a city planner, I want the 3D heat map to show Mumbai districts as correctly positioned 3D building clusters instead of a uniform grid, so that the spatial relationships between districts are geographically accurate.

#### Acceptance Criteria

1. THE District_Projector SHALL convert each district's GPS centroid (stored as `centroid_lat` and `centroid_lon` fields on the District_Block) to a Three.js scene coordinate using an equirectangular projection anchored to the Mumbai_Bounding_Box, normalised so that the full Mumbai extent maps to the range [−38, +38] on both the X and Z axes.
2. WHEN the 3D scene is initialised, THE 3D_Map SHALL position each district's building cluster mesh at the scene coordinate derived from the district's GPS centroid rather than a fixed grid offset.
3. THE 3D_Map SHALL scale each district's ground footprint mesh using the formula `footprint = clamp(sqrt(district_area_km2 / max_area_km2) × 14, 4, 14)` scene units per side, where `district_area_km2` and `max_area_km2` are sourced from the District_Block `area_km2` field.
4. WHEN two district centroids project to scene positions closer than 3 scene units apart, THE District_Projector SHALL translate the centroid with the smaller `area_km2` value by the minimum vector required to achieve a 3-scene-unit separation, and SHALL log a console message identifying both district names, the original separation, and the applied translation vector.
5. THE 3D_Map SHALL attach all existing visual layer objects (wind flow bands, solar beams, thermal rings, PINN halos, heat plumes) to each district mesh at the same relative offsets used in the current grid layout, so that removing any district mesh also removes its associated visual layers.

---

### Requirement 4: District-Level Block Data Model

**User Story:** As a developer, I want each Mumbai district to carry the same micro-climate data fields as the current synthetic blocks, so that all existing dashboard panels and API endpoints continue to work without modification.

#### Acceptance Criteria

1. THE District_Block SHALL include the fields: `id` (district slug, e.g. `"andheri-east"`), `district_name` (human-readable string), `zone_class` (one of the four zone class strings), `max_height_m`, `uhi_intensity`, `svf`, `lambda_p`, `hw_ratio`, `albedo`, `green_cover`, `surface_temp_c`, `air_temp_c`, `centroid_lat`, `centroid_lon`, `area_km2`, `col` (integer in [0, 4] for legacy grid compatibility), `row` (integer in [0, 3] for legacy grid compatibility), and `geometry` (a GeoJSON geometry object of type `"Polygon"`, not a GeoJSON Feature wrapper).
2. THE Zone_Classifier SHALL assign exactly one of `WIND_CORRIDOR_CRITICAL`, `THERMAL_REMEDIATION`, `DENSITY_ADAPTIVE`, or `BASELINE_UNCHANGED` to each District_Block using the existing classification logic in `_classify_actual_zone`.
3. WHEN the Governance_API `/v1/data/actual` endpoint is called, THE Governance_API SHALL return District_Block records whose `id` values are case-sensitive exact matches of the district slugs defined in `src/dashboard/data/mumbai_districts.geojson`, so that the frontend can join geographic and climate data by `id`.
4. THE District_Block SHALL be serialisable to and parseable from a JSON object such that `parse(serialise(district_block)) == district_block` for all field values (round-trip property).
5. IF a district has no matching weather or OSM morphology data, THEN THE Zone_Classifier SHALL assign `BASELINE_UNCHANGED` and set `uhi_intensity` to `0.0`, `surface_temp_c` to `28.0`, `air_temp_c` to `30.0`, `svf` to `0.5`, `lambda_p` to `0.3`, `hw_ratio` to `1.0`, `albedo` to `0.2`, and `green_cover` to `0.1`, so that the map renders without gaps.

---

### Requirement 5: Shared State Compatibility

**User Story:** As a user, I want changes I make in the 2D dashboard (block overrides, Pareto configuration selection) to be reflected in the 3D map, so that both views stay in sync.

#### Acceptance Criteria

1. WHEN a District_Block is updated via the Block Override panel, THE Dashboard SHALL serialise the updated District_Block (including its `geometry` field) to the `microclimate-dashboard-blocks-v1` localStorage key using the existing `saveBlocksToSharedState` function.
2. WHEN the 3D_Map loads, IF a Shared_State record contains a `geometry` field that is a non-null object with a `type` property equal to `"Polygon"`, THEN THE 3D_Map SHALL use that `geometry` field to position the district mesh geographically; OTHERWISE THE 3D_Map SHALL apply the legacy `col`/`row` grid positioning for that record.
3. THE Shared_State SHALL preserve the `geometry` field through at least one serialise-then-parse cycle without coordinate loss, verified by checking that every coordinate value in each polygon ring differs by no more than 1×10⁻⁹ degrees before and after the cycle.
4. IF the `geometry` field is absent, null, or not an object with a `type` property in a Shared_State record, THEN THE 3D_Map SHALL fall back to the legacy `col`/`row` grid positioning for that record, so that old cached state does not break the 3D view.
5. WHEN the `microclimate-dashboard-blocks-v1` localStorage key is updated while the 3D_Map page is open, THE 3D_Map SHALL listen for the browser `storage` event and re-render the affected district meshes within one animation frame, so that live overrides from the 2D dashboard are reflected without a page reload.

---

### Requirement 6: GeoJSON Data Sourcing and Provenance

**User Story:** As a data steward, I want the Mumbai district GeoJSON to be sourced from verifiable public geographic data, so that the boundaries are legally and factually accurate.

#### Acceptance Criteria

1. THE GeoJSON_Loader SHALL load district boundary data from the file `src/dashboard/data/mumbai_districts.geojson` committed to the repository, sourced from OpenStreetMap or the Survey of India open data portal.
2. THE GeoJSON file SHALL contain coordinate pairs in WGS 84 (EPSG:4326) decimal degrees with at least 4 decimal places of precision (approximately 11-metre accuracy).
3. THE GeoJSON file SHALL include a `metadata` top-level property containing: `source` (non-empty string), `license` (non-empty string), and `retrieved_date` (a non-empty string in ISO 8601 date format, e.g. `"2024-01-15"`) so that data provenance is auditable.
4. WHEN the Governance_API starts, THE Governance_API SHALL read `src/dashboard/data/mumbai_districts.geojson` and log the values of `metadata.source` and `metadata.retrieved_date` at INFO level so that provenance is captured in server logs.
5. IF the GeoJSON file is absent or the `metadata` object is missing or any of `source`, `license`, or `retrieved_date` fields are absent or empty at Governance_API startup, THEN THE Governance_API SHALL log an ERROR-level message identifying the missing field(s) and SHALL continue starting up using default district data, so that a missing metadata file does not prevent the API from serving requests.

---

### Requirement 7: Integration with Existing Dashboard Navigation

**User Story:** As a user, I want the "3D Map" navigation button in the dashboard to open the geographically accurate 3D district map, so that both views are consistent.

#### Acceptance Criteria

1. THE Dashboard SHALL retain the existing `<a class="nav-link-btn" href="3d-map.html">3D Map</a>` navigation link so that users can switch between the 2D and 3D views without additional steps.
2. WHEN the 3D_Map page loads, THE 3D_Map SHALL complete the first Three.js render call within 3 seconds of the `DOMContentLoaded` event firing, measured by recording `performance.now()` at `DOMContentLoaded` and at the first `requestAnimationFrame` callback that follows scene initialisation.
3. WHILE the backend data API request is in flight after the 3D_Map page loads, THE 3D_Map SHALL display a loading indicator element with `id="map3d-loading"` inside the `#map3d-canvas` container.
4. WHEN the backend data API request completes and the scene is rendered, THE 3D_Map SHALL remove the `id="map3d-loading"` element from the DOM.
5. IF the browser does not support ES module imports, THEN THE Dashboard SHALL display a static message reading "Please upgrade your browser to use this application" in place of the canvas element, so that the application does not silently fail.
