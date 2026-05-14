/**
 * Unit tests for geojson-loader.js
 *
 * Tests validateGeoJSONFeature and loadMumbaiGeoJSON.
 * Requirements: 1.2, 1.3
 *
 * Run with:
 *   npm install
 *   npm test
 *
 * (Requires Node ≥18 for native fetch; Jest is invoked with
 *  --experimental-vm-modules to support ES module imports.)
 */

import { validateGeoJSONFeature, loadMumbaiGeoJSON } from '../geojson-loader.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal valid GeoJSON Feature. */
function makeValidFeature(overrides = {}) {
  return {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [72.8, 19.0],
          [72.9, 19.0],
          [72.9, 19.1],
          [72.8, 19.1],
          [72.8, 19.0],
        ],
      ],
    },
    properties: {
      name: 'Andheri East',
      district: 'andheri-east',
    },
    ...overrides,
  };
}

/** Build a minimal valid GeoJSON FeatureCollection. */
function makeFeatureCollection(features, metadata = {}) {
  return {
    type: 'FeatureCollection',
    metadata: {
      source: 'OpenStreetMap',
      license: 'ODbL',
      retrieved_date: '2024-01-15',
      ...metadata,
    },
    features,
  };
}

/** Create a mock fetch that resolves with the given GeoJSON object. */
function mockFetchOk(geojson) {
  return jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: 'OK',
    text: () => Promise.resolve(JSON.stringify(geojson)),
  });
}

/** Create a mock fetch that resolves with an HTTP error status. */
function mockFetchError(status, statusText) {
  return jest.fn().mockResolvedValue({
    ok: false,
    status,
    statusText,
    text: () => Promise.resolve(''),
  });
}

/** Create a mock fetch that resolves with non-JSON text. */
function mockFetchMalformedJSON() {
  return jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: 'OK',
    text: () => Promise.resolve('{ this is not valid json '),
  });
}

// ---------------------------------------------------------------------------
// validateGeoJSONFeature
// ---------------------------------------------------------------------------

describe('validateGeoJSONFeature', () => {
  let warnSpy;

  beforeEach(() => {
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  test('returns true for a valid Polygon feature with name and district', () => {
    const feature = makeValidFeature();
    expect(validateGeoJSONFeature(feature, 0)).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  test('returns true for a valid MultiPolygon feature with only name', () => {
    const feature = makeValidFeature({
      geometry: { type: 'MultiPolygon', coordinates: [[[[72.8, 19.0], [72.9, 19.0], [72.9, 19.1], [72.8, 19.0]]]] },
      properties: { name: 'Kurla' },
    });
    expect(validateGeoJSONFeature(feature, 1)).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  test('returns true for a valid feature with only district (no name)', () => {
    const feature = makeValidFeature({
      properties: { district: 'bandra-west' },
    });
    expect(validateGeoJSONFeature(feature, 2)).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  test('returns false and warns when geometry is missing', () => {
    const feature = { type: 'Feature', geometry: null, properties: { name: 'Test' } };
    expect(validateGeoJSONFeature(feature, 3)).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toContain('3');
  });

  test('returns false and warns when geometry type is "Point"', () => {
    const feature = makeValidFeature({
      geometry: { type: 'Point', coordinates: [72.85, 19.05] },
    });
    expect(validateGeoJSONFeature(feature, 4)).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toContain('4');
  });

  test('returns false and warns when geometry type is "LineString"', () => {
    const feature = makeValidFeature({
      geometry: { type: 'LineString', coordinates: [[72.8, 19.0], [72.9, 19.1]] },
    });
    expect(validateGeoJSONFeature(feature, 5)).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  test('returns false and warns when both name and district are missing from properties', () => {
    const feature = makeValidFeature({ properties: { area_km2: 42 } });
    expect(validateGeoJSONFeature(feature, 6)).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toContain('6');
  });

  test('returns false and warns when name is an empty string and district is absent', () => {
    const feature = makeValidFeature({ properties: { name: '' } });
    expect(validateGeoJSONFeature(feature, 7)).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toContain('7');
  });

  test('returns false and warns when name is whitespace-only and district is absent', () => {
    const feature = makeValidFeature({ properties: { name: '   ' } });
    expect(validateGeoJSONFeature(feature, 8)).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  test('returns false and warns when feature itself is null', () => {
    expect(validateGeoJSONFeature(null, 9)).toBe(false);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toContain('9');
  });

  test('warn message includes the feature index', () => {
    const feature = makeValidFeature({ geometry: null });
    validateGeoJSONFeature(feature, 42);
    expect(warnSpy.mock.calls[0][0]).toContain('42');
  });
});

// ---------------------------------------------------------------------------
// loadMumbaiGeoJSON
// ---------------------------------------------------------------------------

describe('loadMumbaiGeoJSON', () => {
  let errorSpy;
  let warnSpy;
  let originalFetch;

  beforeEach(() => {
    errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    originalFetch = global.fetch;
  });

  afterEach(() => {
    errorSpy.mockRestore();
    warnSpy.mockRestore();
    global.fetch = originalFetch;
  });

  // -------------------------------------------------------------------------
  // Successful load
  // -------------------------------------------------------------------------

  test('returns correct feature count on successful load', async () => {
    const features = [
      makeValidFeature({ properties: { name: 'Andheri East' } }),
      makeValidFeature({ properties: { name: 'Bandra West' } }),
      makeValidFeature({ properties: { name: 'Kurla' } }),
    ];
    const geojson = makeFeatureCollection(features);
    global.fetch = mockFetchOk(geojson);

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).not.toBeNull();
    expect(result.features).toHaveLength(3);
    expect(errorSpy).not.toHaveBeenCalled();
  });

  test('returns metadata from the GeoJSON file on successful load', async () => {
    const features = [makeValidFeature()];
    const geojson = makeFeatureCollection(features, {
      source: 'OpenStreetMap',
      license: 'ODbL',
      retrieved_date: '2024-06-01',
    });
    global.fetch = mockFetchOk(geojson);

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).not.toBeNull();
    expect(result.metadata.source).toBe('OpenStreetMap');
    expect(result.metadata.retrieved_date).toBe('2024-06-01');
  });

  test('filters out invalid features and returns only valid ones', async () => {
    const validFeature = makeValidFeature({ properties: { name: 'Andheri East' } });
    const invalidFeature = makeValidFeature({
      geometry: { type: 'Point', coordinates: [72.85, 19.05] },
    });
    const geojson = makeFeatureCollection([validFeature, invalidFeature]);
    global.fetch = mockFetchOk(geojson);

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).not.toBeNull();
    expect(result.features).toHaveLength(1);
  });

  // -------------------------------------------------------------------------
  // HTTP 404 → null
  // -------------------------------------------------------------------------

  test('returns null on HTTP 404 and emits console.error', async () => {
    global.fetch = mockFetchError(404, 'Not Found');

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).toBeNull();
    expect(errorSpy).toHaveBeenCalledTimes(1);
    const msg = errorSpy.mock.calls[0][0];
    expect(msg).toContain('./data/mumbai_districts.geojson');
    expect(msg).toContain('404');
  });

  test('returns null on HTTP 500 and emits console.error', async () => {
    global.fetch = mockFetchError(500, 'Internal Server Error');

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).toBeNull();
    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy.mock.calls[0][0]).toContain('500');
  });

  // -------------------------------------------------------------------------
  // Network error → null
  // -------------------------------------------------------------------------

  test('returns null on network error and emits console.error', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('Network failure'));

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).toBeNull();
    expect(errorSpy).toHaveBeenCalledTimes(1);
    const msg = errorSpy.mock.calls[0][0];
    expect(msg).toContain('./data/mumbai_districts.geojson');
    expect(msg).toContain('Network failure');
  });

  // -------------------------------------------------------------------------
  // Malformed JSON → null
  // -------------------------------------------------------------------------

  test('returns null on malformed JSON and emits console.error with "JSON syntax error"', async () => {
    global.fetch = mockFetchMalformedJSON();

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).toBeNull();
    expect(errorSpy).toHaveBeenCalledTimes(1);
    const msg = errorSpy.mock.calls[0][0];
    expect(msg).toContain('./data/mumbai_districts.geojson');
    expect(msg).toContain('JSON syntax error');
  });

  // -------------------------------------------------------------------------
  // All-invalid features → null
  // -------------------------------------------------------------------------

  test('returns null when all features fail validation', async () => {
    const allInvalid = [
      makeValidFeature({ geometry: { type: 'Point', coordinates: [72.85, 19.05] } }),
      makeValidFeature({ properties: { name: '' } }),
      makeValidFeature({ geometry: null }),
    ];
    const geojson = makeFeatureCollection(allInvalid);
    global.fetch = mockFetchOk(geojson);

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).toBeNull();
    expect(errorSpy).toHaveBeenCalledTimes(1);
  });

  test('returns null when features array is empty', async () => {
    const geojson = makeFeatureCollection([]);
    global.fetch = mockFetchOk(geojson);

    const result = await loadMumbaiGeoJSON('./data/mumbai_districts.geojson');

    expect(result).toBeNull();
    expect(errorSpy).toHaveBeenCalledTimes(1);
  });

  // -------------------------------------------------------------------------
  // Error message includes the URL
  // -------------------------------------------------------------------------

  test('error message includes the requested URL for HTTP errors', async () => {
    const url = 'https://example.com/districts.geojson';
    global.fetch = mockFetchError(403, 'Forbidden');

    await loadMumbaiGeoJSON(url);

    expect(errorSpy.mock.calls[0][0]).toContain(url);
  });

  test('error message includes the requested URL for JSON parse errors', async () => {
    const url = 'https://example.com/districts.geojson';
    global.fetch = mockFetchMalformedJSON();

    await loadMumbaiGeoJSON(url);

    expect(errorSpy.mock.calls[0][0]).toContain(url);
  });
});
