/**
 * Unit tests for district-projector.js
 *
 * Tests projectToSVG, projectToScene, computeFootprint, and resolveCollisions.
 * Requirements: 2.1, 3.1, 3.3, 3.4
 *
 * Run with:
 *   npm install
 *   npm test
 *
 * (Requires Node ≥18; Jest is invoked with --experimental-vm-modules to
 *  support ES module imports.)
 */

import {
  MUMBAI_BBOX,
  projectToSVG,
  projectToScene,
  computeFootprint,
  resolveCollisions,
} from '../district-projector.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const { minLat, maxLat, minLon, maxLon } = MUMBAI_BBOX;

/** Build a minimal District_Block for resolveCollisions tests. */
function makeBlock(centroid_lat, centroid_lon, area_km2, district_name) {
  return { centroid_lat, centroid_lon, area_km2, district_name };
}

/** Euclidean distance between two blocks' resolved scene positions. */
function sceneDist(a, b) {
  const dx = a._scene_x - b._scene_x;
  const dz = a._scene_z - b._scene_z;
  return Math.sqrt(dx * dx + dz * dz);
}

// ---------------------------------------------------------------------------
// projectToSVG
// ---------------------------------------------------------------------------

describe('projectToSVG', () => {
  const W = 800;
  const H = 600;

  test('bottom-left corner (minLat, minLon) maps to (0, svgHeight)', () => {
    const { x, y } = projectToSVG(minLat, minLon, W, H);
    expect(x).toBeCloseTo(0, 10);
    expect(y).toBeCloseTo(H, 10);
  });

  test('top-left corner (maxLat, minLon) maps to (0, 0)', () => {
    const { x, y } = projectToSVG(maxLat, minLon, W, H);
    expect(x).toBeCloseTo(0, 10);
    expect(y).toBeCloseTo(0, 10);
  });

  test('top-right corner (maxLat, maxLon) maps to (svgWidth, 0)', () => {
    const { x, y } = projectToSVG(maxLat, maxLon, W, H);
    expect(x).toBeCloseTo(W, 10);
    expect(y).toBeCloseTo(0, 10);
  });

  test('bottom-right corner (minLat, maxLon) maps to (svgWidth, svgHeight)', () => {
    const { x, y } = projectToSVG(minLat, maxLon, W, H);
    expect(x).toBeCloseTo(W, 10);
    expect(y).toBeCloseTo(H, 10);
  });

  test('centroid coordinate maps to interior of viewport', () => {
    // Approximate centroid of Mumbai bounding box
    const lat = (minLat + maxLat) / 2;
    const lon = (minLon + maxLon) / 2;
    const { x, y } = projectToSVG(lat, lon, W, H);
    expect(x).toBeCloseTo(W / 2, 10);
    expect(y).toBeCloseTo(H / 2, 10);
  });

  test('x is exactly 0 when lon equals minLon', () => {
    const { x } = projectToSVG(19.0, minLon, W, H);
    expect(x).toBe(0);
  });

  test('x is exactly svgWidth when lon equals maxLon', () => {
    const { x } = projectToSVG(19.0, maxLon, W, H);
    expect(x).toBe(W);
  });

  test('y is exactly 0 when lat equals maxLat', () => {
    const { y } = projectToSVG(maxLat, 72.85, W, H);
    expect(y).toBe(0);
  });

  test('y is exactly svgHeight when lat equals minLat', () => {
    const { y } = projectToSVG(minLat, 72.85, W, H);
    expect(y).toBe(H);
  });

  test('scales proportionally with svgWidth and svgHeight', () => {
    const lat = 19.0;
    const lon = 72.85;
    const r1 = projectToSVG(lat, lon, 400, 300);
    const r2 = projectToSVG(lat, lon, 800, 600);
    expect(r2.x).toBeCloseTo(r1.x * 2, 10);
    expect(r2.y).toBeCloseTo(r1.y * 2, 10);
  });
});

// ---------------------------------------------------------------------------
// projectToScene
// ---------------------------------------------------------------------------

describe('projectToScene', () => {
  test('bottom-left corner (minLat, minLon) maps to (-38, +38)', () => {
    const { x, z } = projectToScene(minLat, minLon);
    expect(x).toBeCloseTo(-38, 10);
    expect(z).toBeCloseTo(38, 10);
  });

  test('top-left corner (maxLat, minLon) maps to (-38, -38)', () => {
    const { x, z } = projectToScene(maxLat, minLon);
    expect(x).toBeCloseTo(-38, 10);
    expect(z).toBeCloseTo(-38, 10);
  });

  test('top-right corner (maxLat, maxLon) maps to (+38, -38)', () => {
    const { x, z } = projectToScene(maxLat, maxLon);
    expect(x).toBeCloseTo(38, 10);
    expect(z).toBeCloseTo(-38, 10);
  });

  test('bottom-right corner (minLat, maxLon) maps to (+38, +38)', () => {
    const { x, z } = projectToScene(minLat, maxLon);
    expect(x).toBeCloseTo(38, 10);
    expect(z).toBeCloseTo(38, 10);
  });

  test('centroid of bounding box maps to (0, 0)', () => {
    const lat = (minLat + maxLat) / 2;
    const lon = (minLon + maxLon) / 2;
    const { x, z } = projectToScene(lat, lon);
    expect(x).toBeCloseTo(0, 10);
    expect(z).toBeCloseTo(0, 10);
  });

  test('interior coordinate produces x within [-38, +38]', () => {
    const { x } = projectToScene(19.0, 72.85);
    expect(x).toBeGreaterThanOrEqual(-38);
    expect(x).toBeLessThanOrEqual(38);
  });

  test('interior coordinate produces z within [-38, +38]', () => {
    const { z } = projectToScene(19.0, 72.85);
    expect(z).toBeGreaterThanOrEqual(-38);
    expect(z).toBeLessThanOrEqual(38);
  });
});

// ---------------------------------------------------------------------------
// computeFootprint
// ---------------------------------------------------------------------------

describe('computeFootprint', () => {
  test('area = 0 returns minimum footprint of 4', () => {
    expect(computeFootprint(0, 100)).toBe(4);
  });

  test('area = maxArea returns maximum footprint of 14', () => {
    expect(computeFootprint(100, 100)).toBeCloseTo(14, 10);
  });

  test('area = maxArea/4 returns footprint of 7', () => {
    // sqrt(0.25) * 14 = 0.5 * 14 = 7
    expect(computeFootprint(25, 100)).toBeCloseTo(7, 10);
  });

  test('area > maxArea clamps to 14', () => {
    expect(computeFootprint(200, 100)).toBe(14);
  });

  test('maxAreaKm2 = 0 returns minimum footprint of 4', () => {
    expect(computeFootprint(50, 0)).toBe(4);
  });

  test('maxAreaKm2 < 0 returns minimum footprint of 4', () => {
    expect(computeFootprint(50, -10)).toBe(4);
  });

  test('very small area returns value close to 4 (clamped)', () => {
    const result = computeFootprint(0.001, 1000);
    expect(result).toBeGreaterThanOrEqual(4);
    expect(result).toBeLessThanOrEqual(14);
  });

  test('result is always within [4, 14] for valid inputs', () => {
    const cases = [
      [0, 100],
      [1, 100],
      [50, 100],
      [100, 100],
      [150, 100],
    ];
    for (const [area, max] of cases) {
      const result = computeFootprint(area, max);
      expect(result).toBeGreaterThanOrEqual(4);
      expect(result).toBeLessThanOrEqual(14);
    }
  });
});

// ---------------------------------------------------------------------------
// resolveCollisions
// ---------------------------------------------------------------------------

describe('resolveCollisions', () => {
  let logSpy;

  beforeEach(() => {
    logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  test('no-collision case leaves positions unchanged', () => {
    // Place two blocks far apart (different sides of Mumbai)
    const a = makeBlock(18.95, 72.80, 50, 'District A');
    const b = makeBlock(19.20, 72.95, 80, 'District B');

    resolveCollisions([a, b]);

    const posA = projectToScene(18.95, 72.80);
    const posB = projectToScene(19.20, 72.95);

    expect(a._scene_x).toBeCloseTo(posA.x, 10);
    expect(a._scene_z).toBeCloseTo(posA.z, 10);
    expect(b._scene_x).toBeCloseTo(posB.x, 10);
    expect(b._scene_z).toBeCloseTo(posB.z, 10);
    expect(logSpy).not.toHaveBeenCalled();
  });

  test('single collision achieves ≥3 unit separation after resolution', () => {
    // Place two blocks at the same centroid (worst-case collision)
    const a = makeBlock(19.08, 72.88, 100, 'Large District');
    const b = makeBlock(19.08, 72.88, 20, 'Small District');

    resolveCollisions([a, b], 3);

    const dist = sceneDist(a, b);
    expect(dist).toBeGreaterThanOrEqual(3);
  });

  test('single collision logs the translation', () => {
    const a = makeBlock(19.08, 72.88, 100, 'Large District');
    const b = makeBlock(19.08, 72.88, 20, 'Small District');

    resolveCollisions([a, b], 3);

    expect(logSpy).toHaveBeenCalledTimes(1);
    const msg = logSpy.mock.calls[0][0];
    expect(msg).toContain('Large District');
    expect(msg).toContain('Small District');
  });

  test('blocks very close together achieve ≥3 unit separation', () => {
    // Place blocks 0.5 scene units apart (well within minSeparation=3)
    // Use centroids that are geographically very close
    const a = makeBlock(19.08, 72.880, 80, 'District A');
    const b = makeBlock(19.08, 72.881, 30, 'District B');

    resolveCollisions([a, b], 3);

    const dist = sceneDist(a, b);
    expect(dist).toBeGreaterThanOrEqual(3);
  });

  test('multiple collisions: all pairs achieve ≥3 unit separation', () => {
    // Three blocks clustered at nearly the same point
    const a = makeBlock(19.08, 72.880, 100, 'District A');
    const b = makeBlock(19.08, 72.881, 60, 'District B');
    const c = makeBlock(19.08, 72.882, 20, 'District C');

    resolveCollisions([a, b, c], 3);

    expect(sceneDist(a, b)).toBeGreaterThanOrEqual(3);
    expect(sceneDist(a, c)).toBeGreaterThanOrEqual(3);
    expect(sceneDist(b, c)).toBeGreaterThanOrEqual(3);
  });

  test('larger area block is not moved when collision occurs', () => {
    const large = makeBlock(19.08, 72.88, 200, 'Large');
    const small = makeBlock(19.08, 72.88, 10, 'Small');

    resolveCollisions([large, small], 3);

    // Large block should stay at its original projected position
    const origPos = projectToScene(19.08, 72.88);
    expect(large._scene_x).toBeCloseTo(origPos.x, 10);
    expect(large._scene_z).toBeCloseTo(origPos.z, 10);
  });

  test('smaller area block is moved when collision occurs', () => {
    const large = makeBlock(19.08, 72.88, 200, 'Large');
    const small = makeBlock(19.08, 72.88, 10, 'Small');

    resolveCollisions([large, small], 3);

    // Small block must have moved
    const origPos = projectToScene(19.08, 72.88);
    const movedX = small._scene_x !== origPos.x || small._scene_z !== origPos.z;
    expect(movedX).toBe(true);
  });

  test('empty array does not throw', () => {
    expect(() => resolveCollisions([])).not.toThrow();
  });

  test('single block does not throw and initialises scene position', () => {
    const block = makeBlock(19.08, 72.88, 50, 'Solo');
    resolveCollisions([block]);
    const expected = projectToScene(19.08, 72.88);
    expect(block._scene_x).toBeCloseTo(expected.x, 10);
    expect(block._scene_z).toBeCloseTo(expected.z, 10);
  });

  test('custom minSeparation is respected', () => {
    const a = makeBlock(19.08, 72.880, 100, 'A');
    const b = makeBlock(19.08, 72.881, 30, 'B');

    resolveCollisions([a, b], 10);

    expect(sceneDist(a, b)).toBeGreaterThanOrEqual(10);
  });
});
