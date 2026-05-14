/**
 * GeoJSON Loader module for Mumbai district boundary data.
 * Handles fetching, parsing, validation, and overlap detection.
 *
 * Requirements: 1.2, 1.3, 1.4
 */

/**
 * Validates a single GeoJSON Feature.
 *
 * Returns true if the feature has a geometry of type "Polygon" or "MultiPolygon"
 * AND has a properties object with a non-empty "name" or "district" string.
 * Returns false and emits a console.warn identifying the feature index and
 * the missing/invalid field for any failing check.
 *
 * @param {object} feature - A GeoJSON Feature object
 * @param {number} index   - Zero-based index of the feature in the collection
 * @returns {boolean}
 *
 * Requirements: 1.2
 */
export function validateGeoJSONFeature(feature, index) {
  // Check geometry presence and type
  if (
    !feature ||
    !feature.geometry ||
    (feature.geometry.type !== 'Polygon' && feature.geometry.type !== 'MultiPolygon')
  ) {
    const missingField =
      !feature || !feature.geometry ? 'missing geometry' : 'invalid geometry type';
    console.warn(
      `GeoJSON feature at index ${index}: ${missingField}`
    );
    return false;
  }

  // Check properties for a non-empty name or district string
  const props = feature.properties;
  const hasName =
    props &&
    typeof props.name === 'string' &&
    props.name.trim().length > 0;
  const hasDistrict =
    props &&
    typeof props.district === 'string' &&
    props.district.trim().length > 0;

  if (!hasName && !hasDistrict) {
    console.warn(
      `GeoJSON feature at index ${index}: missing name field`
    );
    return false;
  }

  return true;
}

/**
 * Returns the axis-aligned bounding box of a Polygon or MultiPolygon geometry.
 *
 * @param {object} geometry - GeoJSON geometry (Polygon or MultiPolygon)
 * @returns {{ minLon: number, maxLon: number, minLat: number, maxLat: number }}
 */
function getBoundingBox(geometry) {
  let minLon = Infinity;
  let maxLon = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;

  const rings =
    geometry.type === 'Polygon'
      ? geometry.coordinates
      : geometry.coordinates.flat(1); // MultiPolygon: flatten one level to get rings

  for (const ring of rings) {
    for (const [lon, lat] of ring) {
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
    }
  }

  return { minLon, maxLon, minLat, maxLat };
}

/**
 * Computes the area of an axis-aligned bounding box in squared degrees.
 *
 * @param {{ minLon: number, maxLon: number, minLat: number, maxLat: number }} bbox
 * @returns {number}
 */
function bboxArea(bbox) {
  const w = Math.max(0, bbox.maxLon - bbox.minLon);
  const h = Math.max(0, bbox.maxLat - bbox.minLat);
  return w * h;
}

/**
 * Computes the intersection area of two axis-aligned bounding boxes.
 *
 * @param {{ minLon: number, maxLon: number, minLat: number, maxLat: number }} a
 * @param {{ minLon: number, maxLon: number, minLat: number, maxLat: number }} b
 * @returns {number} Intersection area in squared degrees (0 if no overlap)
 */
function bboxIntersectionArea(a, b) {
  const overlapLon = Math.max(
    0,
    Math.min(a.maxLon, b.maxLon) - Math.max(a.minLon, b.minLon)
  );
  const overlapLat = Math.max(
    0,
    Math.min(a.maxLat, b.maxLat) - Math.max(a.minLat, b.minLat)
  );
  return overlapLon * overlapLat;
}

/**
 * Performs a pairwise bounding-box overlap check on all features.
 *
 * For each pair of features whose bounding boxes overlap, the shared area is
 * estimated as the bounding-box intersection area. If that shared area exceeds
 * 0.01% of the smaller polygon's bounding-box area, a console.warn is emitted
 * with both district names and the overlap percentage.
 *
 * @param {object[]} features - Array of validated GeoJSON Feature objects
 *
 * Requirements: 1.4
 */
export function checkOverlaps(features) {
  for (let i = 0; i < features.length; i++) {
    for (let j = i + 1; j < features.length; j++) {
      const featureA = features[i];
      const featureB = features[j];

      const bboxA = getBoundingBox(featureA.geometry);
      const bboxB = getBoundingBox(featureB.geometry);

      const intersection = bboxIntersectionArea(bboxA, bboxB);
      if (intersection <= 0) continue;

      const areaA = bboxArea(bboxA);
      const areaB = bboxArea(bboxB);
      const smallerArea = Math.min(areaA, areaB);

      if (smallerArea <= 0) continue;

      const overlapFraction = intersection / smallerArea;
      const overlapPercent = overlapFraction * 100;

      // Threshold: 0.01% of the smaller polygon's area
      if (overlapPercent > 0.01) {
        const nameA =
          (featureA.properties && (featureA.properties.name || featureA.properties.district)) ||
          `feature ${i}`;
        const nameB =
          (featureB.properties && (featureB.properties.name || featureB.properties.district)) ||
          `feature ${j}`;

        console.warn(
          `GeoJSON overlap detected between "${nameA}" and "${nameB}": ` +
            `${overlapPercent.toFixed(4)}% of smaller polygon area`
        );
      }
    }
  }
}

/**
 * Loads and validates the Mumbai districts GeoJSON file.
 *
 * - On HTTP 4xx/5xx or network error: emits console.error with path + failure
 *   description and returns null.
 * - On JSON parse failure: emits console.error with path + "JSON syntax error"
 *   and returns null.
 * - On success: validates each Feature via validateGeoJSONFeature, collects
 *   valid features, calls checkOverlaps, and returns { features, metadata }.
 * - If all features are invalid after validation: returns null.
 *
 * @param {string} url - URL or path to the GeoJSON file
 * @returns {Promise<{ features: object[], metadata: object } | null>}
 *
 * Requirements: 1.2, 1.3
 */
export async function loadMumbaiGeoJSON(url) {
  let response;

  // --- Fetch ---
  try {
    response = await fetch(url);
  } catch (networkError) {
    console.error(
      `GeoJSON_Loader: failed to fetch "${url}" — ${networkError.message || networkError}`
    );
    return null;
  }

  if (!response.ok) {
    console.error(
      `GeoJSON_Loader: failed to fetch "${url}" — HTTP ${response.status} ${response.statusText}`
    );
    return null;
  }

  // --- Parse ---
  let geojson;
  try {
    const text = await response.text();
    geojson = JSON.parse(text);
  } catch (_parseError) {
    console.error(
      `GeoJSON_Loader: failed to parse "${url}" — JSON syntax error`
    );
    return null;
  }

  // --- Validate features ---
  const rawFeatures = Array.isArray(geojson.features) ? geojson.features : [];
  const validFeatures = rawFeatures.filter((feature, index) =>
    validateGeoJSONFeature(feature, index)
  );

  if (validFeatures.length === 0) {
    console.error(
      `GeoJSON_Loader: no valid features found in "${url}"`
    );
    return null;
  }

  // --- Overlap check ---
  checkOverlaps(validFeatures);

  // --- Return result ---
  const metadata = geojson.metadata || {};
  return { features: validFeatures, metadata };
}
