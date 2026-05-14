# Design Document: Mumbai 3D District Map

## Overview

This feature replaces the synthetic 4×4 grid of city blocks in the Micro-Climate Zoning AI dashboard with a geographically accurate representation of Mumbai's real administrative districts. Both the 2D scenario dashboard (`index.html` / `dashboard.js`) and the 3D heat map (`3d-map.html` / `map3d.js`) are updated to render district polygons sourced from a GeoJSON file containing real GPS boundary data.

The design introduces three new logical modules — `GeoJSON_Loader`, `District_Projector`, and an extended `District_Block` data model — while preserving all existing API contracts, shared state mechanics, and simulation endpoints. A graceful fallback to the existing synthetic grid ensures the dashboard remains functional if geographic data is unavailable.

### Key Design Decisions

- **Equirectangular projection** is used for both 2D SVG and 3D scene coordinate conversion. It is computationally trivial, sufficient for the ~40 km × ~42 km Mumbai extent, and produces no perceptible distortion at this scale.
- **GeoJSON committed to the repository** (not fetched at runtime from an external URL) eliminates network dependency and ensures reproducibility.
- **Legacy `col`/`row` fields retained** on every `District_Block` so that old cached `localStorage` state and any downstream code that reads grid coordinates continues to work without modification.
- **Collision resolution** is applied only in the 3D scene where overlapping meshes would be visually confusing; the 2D SVG renders true polygon shapes so no collision resolution is needed there.
- **Fallback-first architecture**: every loading path (GeoJSON fetch, API call) has an explicit fallback so the UI never shows a blank canvas.

---

## Architecture

```mermaid
graph TD
    subgraph "Static Assets"
        GJ[mumbai_districts.geojson]
    end

    subgraph "Backend (api.py)"
        CSV[actual_blocks.csv\n(district rows)]
        API[GET /v1/data/actual]
        ZC[_classify_actual_zone]
        CSV --> ZC --> API
        GJ_BE[GeoJSON metadata reader\n(startup log)]
    end

    subgraph "Frontend Shared"
        GL[GeoJSON_Loader]
        DP[District_Projector]
        SS[localStorage\nmicroclimate-dashboard-blocks-v1]
    end

    subgraph "2D Dashboard (dashboard.js)"
        LB[loadActualBlocks]
        RM[renderMap\n<polygon>/<path> SVG]
        SB[selectBlock]
        SAVE[saveBlocksToSharedState]
    end

    subgraph "3D Map (map3d.js)"
        LA[loadActualBlocksFromApi]
        CB[createBlocks\n(centroid positioning)]
        FP[footprint formula]
        CR[collision resolver]
        LOAD[loadBlocksFromSharedState]
        SE[storage event listener]
    end

    GJ --> GL
    GL --> DP
    API --> LB
    LB --> RM
    RM --> SB
    SB --> SAVE
    SAVE --> SS
    SS --> LOAD
    LOAD --> CB
    DP --> RM
    DP --> CB
    CB --> FP
    CB --> CR
    SS --> SE
    SE --> CB
    GJ_BE --> GJ
```

### Data Flow Summary

1. On page load, `GeoJSON_Loader` fetches `mumbai_districts.geojson` and validates each Feature.
2. `loadActualBlocks` (dashboard) / `loadActualBlocksFromApi` (3D map) calls `GET /v1/data/actual`, which reads the updated `actual_blocks.csv` containing district-level rows.
3. The API response is joined with GeoJSON geometry by matching `id` (district slug) to produce complete `District_Block` records.
4. `District_Projector` converts GPS coordinates to SVG viewport or Three.js scene coordinates.
5. The 2D dashboard renders `<polygon>`/`<path>` SVG elements; the 3D map positions meshes from centroid projections.
6. `saveBlocksToSharedState` persists the full `District_Block` (including `geometry`) to `localStorage`.
7. The 3D map listens for `storage` events and re-renders affected meshes within one animation frame.

---

## Components and Interfaces

### GeoJSON_Loader

A frontend module (added to `dashboard.js` and `map3d.js`, or extracted to a shared `geojson-loader.js` import) responsible for fetching and validating the GeoJSON file.

```javascript
/**
 * Loads and validates the Mumbai districts GeoJSON file.
 * @param {string} url - Path to the GeoJSON file
 * @returns {Promise<{features: GeoJSONFeature[], metadata: object} | null>}
 *   Resolves to validated features and metadata, or null on fatal error.
 */
async function loadMumbaiGeoJSON(url) { ... }

/**
 * Validates a single GeoJSON Feature.
 * Returns true if valid; logs a console warning and returns false if invalid.
 * @param {object} feature
 * @param {number} index
 * @returns {boolean}
 */
function validateGeoJSONFeature(feature, index) { ... }

/**
 * Checks all polygon pairs for overlap > 0.01% of smaller polygon area.
 * Logs a console warning for each violating pair.
 * @param {GeoJSONFeature[]} features
 */
function checkOverlaps(features) { ... }
```

**Validation rules:**
- Feature must have `geometry.type` of `"Polygon"` or `"MultiPolygon"`
- Feature must have `properties` with a non-empty `name` or `district` string field
- Invalid features are skipped (not fatal); a console warning identifies the feature index and missing field
- If the file is missing (HTTP 4xx/5xx) or top-level JSON is malformed, a console error is emitted and `null` is returned

### District_Projector

A pure-function module that converts between coordinate systems.

```javascript
// Mumbai bounding box constants
const MUMBAI_BBOX = {
  minLat: 18.89, maxLat: 19.27,
  minLon: 72.77, maxLon: 73.00,
};

/**
 * Projects a GPS coordinate to SVG viewport coordinates.
 * Uses equirectangular projection anchored to MUMBAI_BBOX.
 * @param {number} lat - WGS 84 latitude
 * @param {number} lon - WGS 84 longitude
 * @param {number} svgWidth - SVG viewport width in pixels
 * @param {number} svgHeight - SVG viewport height in pixels
 * @returns {{ x: number, y: number }}
 */
function projectToSVG(lat, lon, svgWidth, svgHeight) {
  const x = ((lon - MUMBAI_BBOX.minLon) / (MUMBAI_BBOX.maxLon - MUMBAI_BBOX.minLon)) * svgWidth;
  // SVG y-axis is inverted relative to latitude
  const y = ((MUMBAI_BBOX.maxLat - lat) / (MUMBAI_BBOX.maxLat - MUMBAI_BBOX.minLat)) * svgHeight;
  return { x, y };
}

/**
 * Projects a GPS centroid to Three.js scene coordinates.
 * Normalises Mumbai extent to [-38, +38] on X and Z axes.
 * @param {number} lat
 * @param {number} lon
 * @returns {{ x: number, z: number }}
 */
function projectToScene(lat, lon) {
  const SCENE_RANGE = 38;
  const x = ((lon - MUMBAI_BBOX.minLon) / (MUMBAI_BBOX.maxLon - MUMBAI_BBOX.minLon) - 0.5) * 2 * SCENE_RANGE;
  const z = ((MUMBAI_BBOX.maxLat - lat) / (MUMBAI_BBOX.maxLat - MUMBAI_BBOX.minLat) - 0.5) * 2 * SCENE_RANGE;
  return { x, z };
}

/**
 * Resolves centroid collisions in the 3D scene.
 * Translates the smaller district's centroid by the minimum vector
 * to achieve >= 3 scene-unit separation.
 * @param {District_Block[]} blocks - Mutable array; positions updated in-place
 * @param {number} minSeparation - Minimum separation in scene units (default 3)
 */
function resolveCollisions(blocks, minSeparation = 3) { ... }

/**
 * Computes the footprint size for a district in scene units.
 * @param {number} areaKm2 - District area in km²
 * @param {number} maxAreaKm2 - Maximum area among all districts
 * @returns {number} Footprint in scene units, clamped to [4, 14]
 */
function computeFootprint(areaKm2, maxAreaKm2) {
  return Math.max(4, Math.min(14, Math.sqrt(areaKm2 / maxAreaKm2) * 14));
}
```

### Updated `renderMap()` in `dashboard.js`

The existing `renderMap()` function is extended to render `<polygon>` or `<path>` SVG elements when GeoJSON data is available, falling back to `<rect>` squares otherwise.

```javascript
function renderMap() {
  // If GeoJSON features are loaded, render district polygons
  if (state.geoJSONFeatures && state.geoJSONFeatures.length > 0) {
    renderDistrictMap();
  } else {
    renderSyntheticGrid(); // existing <rect> rendering, unchanged
  }
}

function renderDistrictMap() {
  // For each block, find matching GeoJSON feature by id
  // Project polygon coordinates to SVG viewport
  // Render <polygon> or <path> with data-district-name, fill, stroke
  // Render centroid label (truncated to 12 chars)
}
```

### Updated `createBlocks()` in `map3d.js`

The existing `createBlocks()` function is updated to use `centroid_lat`/`centroid_lon` for positioning when available.

```javascript
function getBlockPosition(block) {
  // If block has centroid_lat and centroid_lon, use geographic projection
  if (Number.isFinite(block.centroid_lat) && Number.isFinite(block.centroid_lon)) {
    return projectToScene(block.centroid_lat, block.centroid_lon);
  }
  // Legacy fallback: col/row grid
  const spacing = 9;
  const xOffset = -2 * spacing;
  const zOffset = -1.5 * spacing;
  return { x: xOffset + block.col * spacing, z: zOffset + block.row * spacing };
}
```

### Updated `_build_actual_blocks()` in `api.py`

The backend function is extended to read district-level rows from the updated CSV and include the new geographic fields.

```python
def _build_actual_blocks(weather: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_csv(ACTUAL_BLOCKS_PATH)
    blocks = []
    for row in rows:
        # ... existing morphology calculations ...
        blocks.append({
            # existing fields preserved
            "id": row["block_id"],          # now a district slug, e.g. "andheri-east"
            "district_name": row.get("district_name", ""),
            "centroid_lat": _as_float(row.get("centroid_lat")),
            "centroid_lon": _as_float(row.get("centroid_lon")),
            "area_km2": _as_float(row.get("area_km2")),
            "col": int(_as_float(row.get("col", 0))),
            "row": int(_as_float(row.get("row", 0))),
            # geometry is joined from GeoJSON at startup, not stored in CSV
            "geometry": _district_geometry_cache.get(row["block_id"]),
            # ... all existing fields ...
        })
    return blocks
```

A module-level `_district_geometry_cache: dict[str, dict]` is populated at API startup by reading `mumbai_districts.geojson`.

---

## Data Models

### District_Block (JavaScript / JSON)

```typescript
interface DistrictBlock {
  // Identity
  id: string;                  // district slug, e.g. "andheri-east"
  district_name: string;       // human-readable, e.g. "Andheri East"

  // Zone classification
  zone_class: "WIND_CORRIDOR_CRITICAL" | "THERMAL_REMEDIATION" |
              "DENSITY_ADAPTIVE" | "BASELINE_UNCHANGED";

  // Morphology / climate metrics (all existing fields preserved)
  max_height_m: number;
  uhi_intensity: number;
  svf: number;
  lambda_p: number;
  hw_ratio: number;
  albedo: number;
  green_cover: number;
  surface_temp_c: number;
  air_temp_c: number;
  wind_speed: number;
  solar_radiation_wm2: number;

  // Geographic fields (NEW)
  centroid_lat: number;        // WGS 84 latitude of district centroid
  centroid_lon: number;        // WGS 84 longitude of district centroid
  area_km2: number;            // District area in km²
  geometry: GeoJSONPolygon;    // GeoJSON geometry object (type: "Polygon"), NOT a Feature wrapper

  // Legacy grid compatibility (retained)
  col: number;                 // integer in [0, 4]
  row: number;                 // integer in [0, 3]

  // Existing optional fields
  interventions: Intervention[];
  compliance_status?: string;
  // ... other existing fields ...
}

interface GeoJSONPolygon {
  type: "Polygon";
  coordinates: number[][][];   // [ring][point][lon, lat]
}
```

### Updated `actual_blocks.csv` Schema

The CSV gains five new columns while retaining all existing columns for backward compatibility:

| Column | Type | Description |
|---|---|---|
| `block_id` | string | District slug (e.g. `andheri-east`) |
| `district_name` | string | Human-readable district name |
| `centroid_lat` | float | WGS 84 latitude of centroid |
| `centroid_lon` | float | WGS 84 longitude of centroid |
| `area_km2` | float | District area in km² |
| `col` | int | Legacy grid column [0–4] |
| `row` | int | Legacy grid row [0–3] |
| *(all existing columns)* | | Unchanged |

### GeoJSON File Structure (`mumbai_districts.geojson`)

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "source": "OpenStreetMap contributors",
    "license": "ODbL 1.0",
    "retrieved_date": "2024-01-15"
  },
  "features": [
    {
      "type": "Feature",
      "id": "andheri-east",
      "properties": {
        "name": "Andheri East",
        "district": "andheri-east",
        "area_km2": 17.4
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[72.8777, 19.1136], [72.8900, 19.1136], ...]]
      }
    }
  ]
}
```

The `id` field on each Feature is the canonical district slug used to join with `actual_blocks.csv` and the API response.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: GeoJSON Feature Validation Filters Correctly

*For any* GeoJSON FeatureCollection containing a mix of valid and invalid features (where invalid means missing `geometry`, wrong geometry type, or missing `properties.name`/`properties.district`), the `GeoJSON_Loader` SHALL process exactly the valid features and emit exactly one console warning per invalid feature.

**Validates: Requirements 1.2**

---

### Property 2: GeoJSON Round-Trip Fidelity

*For any* valid GeoJSON FeatureCollection, parsing the JSON string, serialising it back to a JSON string, and parsing again SHALL produce a FeatureCollection with the same number of Features and coordinate values that differ by no more than 1×10⁻⁹ degrees from the originals.

**Validates: Requirements 1.5**

---

### Property 3: Overlap Detection Correctness

*For any* pair of district polygons, the overlap checker SHALL emit a console warning if and only if the shared interior area exceeds 0.01% of the smaller polygon's area, and SHALL not emit a warning for non-overlapping pairs.

**Validates: Requirements 1.4**

---

### Property 4: SVG Projection Stays Within Viewport

*For any* GPS coordinate (lat, lon) within the Mumbai bounding box (18.89°N–19.27°N, 72.77°E–73.00°E), the `District_Projector.projectToSVG` function SHALL return (x, y) values where x ∈ [0, svgWidth] and y ∈ [0, svgHeight].

**Validates: Requirements 2.1, 2.3**

---

### Property 5: District Label Truncation

*For any* district name string of any length, the rendered SVG label text SHALL be at most 12 characters long.

**Validates: Requirements 2.6**

---

### Property 6: 3D Scene Projection Stays Within Scene Range

*For any* GPS centroid (centroid_lat, centroid_lon) within the Mumbai bounding box, the `District_Projector.projectToScene` function SHALL return (x, z) values where both x and z are within [−38, +38].

**Validates: Requirements 3.1**

---

### Property 7: Footprint Formula Invariant

*For any* district area `area_km2 ≥ 0` and maximum area `max_area_km2 > 0`, the `computeFootprint` function SHALL return a value in the closed interval [4, 14] scene units, equal to `clamp(sqrt(area_km2 / max_area_km2) × 14, 4, 14)`.

**Validates: Requirements 3.3**

---

### Property 8: Collision Resolution Guarantees Minimum Separation

*For any* set of district blocks whose projected scene centroids include at least one pair closer than 3 scene units, after `resolveCollisions` is applied, every pair of district centroids SHALL be separated by at least 3 scene units.

**Validates: Requirements 3.4**

---

### Property 9: Zone Classifier Always Returns a Valid Zone Class

*For any* combination of OSM morphology input values (`osm_road_count`, `green_cover_pct`, `lambda_p`, `max_height_m`), the `_classify_actual_zone` function SHALL return exactly one of the four strings: `"WIND_CORRIDOR_CRITICAL"`, `"THERMAL_REMEDIATION"`, `"DENSITY_ADAPTIVE"`, or `"BASELINE_UNCHANGED"`.

**Validates: Requirements 4.2**

---

### Property 10: District_Block JSON Round-Trip Fidelity

*For any* `District_Block` record with any combination of field values (including a `geometry` field with an arbitrary polygon), `JSON.parse(JSON.stringify(block))` SHALL produce an object where every field value is equal to the original, and every coordinate in the `geometry.coordinates` array differs by no more than 1×10⁻⁹ degrees from the original.

**Validates: Requirements 4.4, 5.3**

---

### Property 11: Shared State Preserves Geometry Through Serialisation

*For any* array of `District_Block` records containing `geometry` fields with arbitrary polygon coordinates, calling `saveBlocksToSharedState` followed by `loadBlocksFromSharedState` SHALL return blocks where every coordinate in every `geometry.coordinates` ring differs by no more than 1×10⁻⁹ degrees from the original values.

**Validates: Requirements 5.1, 5.3**

---

### Property 12: GeoJSON Coordinate Precision

*For any* coordinate pair in the `mumbai_districts.geojson` file, both the latitude and longitude values SHALL have at least 4 decimal places of precision (i.e., the absolute difference between the value and its rounded-to-3-decimal-places counterpart SHALL be greater than 0 for at least one of the two values, or the value SHALL be exactly representable with 4+ decimal places).

**Validates: Requirements 6.2**

---

## Error Handling

### GeoJSON Loading Failures

| Failure Mode | Detection | Response |
|---|---|---|
| File not found (HTTP 404) | `fetch()` response status | `console.error` with path + status; fall back to synthetic grid |
| Network error | `fetch()` rejection | `console.error` with path + error message; fall back to synthetic grid |
| Malformed JSON | `JSON.parse()` throws | `console.error` with path + "JSON syntax error"; fall back to synthetic grid |
| Feature missing geometry | `validateGeoJSONFeature()` | `console.warn` with feature index + "missing geometry"; skip feature |
| Feature missing name/district | `validateGeoJSONFeature()` | `console.warn` with feature index + "missing name field"; skip feature |
| All features invalid | After validation loop | Fall back to synthetic grid; show error banner in `.city-map` |
| Polygon overlap detected | `checkOverlaps()` | `console.warn` with district names + overlap percentage; continue rendering |
| Degenerate polygon (< 3 points) | `District_Projector` | `console.warn` with district name + actual point count; skip polygon |

### API Data Failures

| Failure Mode | Detection | Response |
|---|---|---|
| `/v1/data/actual` unavailable | `fetch()` rejection or non-2xx | Fall back to `localStorage` state; if empty, fall back to synthetic grid |
| Missing `centroid_lat`/`centroid_lon` | `Number.isFinite()` check | Fall back to `col`/`row` grid positioning for that block |
| Missing `geometry` in API response | `geometry == null` check | Fall back to `col`/`row` grid positioning for that block |
| `actual_blocks.csv` missing | `path.exists()` check in `api.py` | HTTP 404 with descriptive message |
| GeoJSON metadata missing at startup | Field presence check | `logger.error` with missing field names; API continues with default data |

### 3D Scene Failures

| Failure Mode | Detection | Response |
|---|---|---|
| Centroid collision unresolvable | After N iterations | Log warning; use best-effort position |
| `area_km2` is zero or negative | `computeFootprint()` guard | Use minimum footprint of 4 scene units |
| `localStorage` parse error | `try/catch` in `loadBlocksFromSharedState` | Return `null`; trigger API fetch |

### Browser Compatibility

If `<script type="module">` is not supported, a `<noscript>`-equivalent `<div>` with the message "Please upgrade your browser to use this application" is rendered in place of the canvas element. This is implemented via a `<script nomodule>` tag that injects the message.

---

## Testing Strategy

### Unit Tests (Python — `pytest`)

Located in `tests/test_governance.py` and a new `tests/test_district_map.py`:

- **Zone classifier exhaustiveness**: Verify `_classify_actual_zone` returns a valid zone class for all boundary combinations of its four input variables.
- **Default fallback values**: Verify that a district row with all-empty morphology fields produces `BASELINE_UNCHANGED` with the specified default metric values.
- **GeoJSON metadata logging**: Verify that API startup logs `metadata.source` and `metadata.retrieved_date` at INFO level.
- **Missing GeoJSON at startup**: Verify that a missing GeoJSON file logs an ERROR and the API health endpoint still returns 200.
- **District_Block serialisation**: Verify that `_build_actual_blocks` output is JSON-serialisable and round-trips correctly.

### Unit Tests (JavaScript — Vitest or Jest)

Located in a new `src/dashboard/tests/` directory:

- **`validateGeoJSONFeature`**: Test valid features, features missing geometry, features with wrong geometry type, features missing name/district.
- **`projectToSVG`**: Test boundary coordinates (corners of Mumbai bounding box), interior coordinates, and coordinates at the exact bounding box edges.
- **`projectToScene`**: Test boundary coordinates produce ±38, interior coordinates produce values within range.
- **`computeFootprint`**: Test area = 0, area = max_area, area = max_area/4, area > max_area (edge case).
- **`resolveCollisions`**: Test no-collision case (unchanged), single collision, multiple collisions.
- **`renderMap` SVG output**: Test that rendered SVG contains `<polygon>`/`<path>` elements with correct `data-district-name` and fill attributes.
- **Label truncation**: Test names of length 0, 12, 13, 50.
- **`loadBlocksFromSharedState` with geometry**: Test that geometry field survives a localStorage round-trip.

### Property-Based Tests (JavaScript — fast-check)

Located in `src/dashboard/tests/properties.test.js`. Each property test runs a minimum of 100 iterations.

The property-based testing library chosen is **[fast-check](https://github.com/dubzzz/fast-check)** (MIT licence, actively maintained, works with Vitest and Jest, supports arbitrary generators for nested objects and arrays).

```javascript
// Tag format: Feature: mumbai-3d-district-map, Property N: <property_text>

// Feature: mumbai-3d-district-map, Property 2: GeoJSON round-trip fidelity
fc.assert(fc.property(arbitraryFeatureCollection(), (fc) => {
  const roundTripped = JSON.parse(JSON.stringify(fc));
  // verify feature count and coordinate precision
}), { numRuns: 100 });

// Feature: mumbai-3d-district-map, Property 4: SVG projection stays within viewport
fc.assert(fc.property(
  fc.float({ min: 18.89, max: 19.27 }),
  fc.float({ min: 72.77, max: 73.00 }),
  fc.integer({ min: 100, max: 2000 }),
  fc.integer({ min: 100, max: 2000 }),
  (lat, lon, w, h) => {
    const { x, y } = projectToSVG(lat, lon, w, h);
    return x >= 0 && x <= w && y >= 0 && y <= h;
  }
), { numRuns: 100 });

// Feature: mumbai-3d-district-map, Property 7: Footprint formula invariant
fc.assert(fc.property(
  fc.float({ min: 0, max: 500 }),
  fc.float({ min: 0.01, max: 500 }),
  (area, maxArea) => {
    const fp = computeFootprint(area, maxArea);
    return fp >= 4 && fp <= 14;
  }
), { numRuns: 100 });
```

### Integration Tests

- **`GET /v1/data/actual` returns district slugs**: Call the endpoint and verify all returned `id` values are present in `mumbai_districts.geojson`.
- **3D map first render timing**: Load `3d-map.html` in a headless browser (Playwright) and verify the first `requestAnimationFrame` callback fires within 3 seconds of `DOMContentLoaded`.
- **Loading indicator lifecycle**: Verify `#map3d-loading` is present during API fetch and removed after scene initialisation.
- **Storage event re-render**: Update `localStorage` while the 3D map is open and verify the scene updates within one animation frame.

### Smoke Tests

- GeoJSON file exists at `src/dashboard/data/mumbai_districts.geojson` and has ≥ 24 features.
- GeoJSON `metadata` object has non-empty `source`, `license`, and `retrieved_date` fields.
- `index.html` contains `<a class="nav-link-btn" href="3d-map.html">3D Map</a>`.
- API health endpoint returns `{"status": "healthy"}` after startup.
