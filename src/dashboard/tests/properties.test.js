/**
 * Property-based tests for district-projector.js
 *
 * Uses fast-check to verify universal properties hold across all valid inputs.
 * This file is extended in tasks 6.5 and 8.7 with additional properties.
 *
 * Validates: Requirements 2.1, 2.3, 3.3
 *
 * Run with:
 *   npm test
 *
 * (Requires Node ≥18; Jest is invoked with --experimental-vm-modules to
 *  support ES module imports.)
 */

import fc from 'fast-check';
import { projectToSVG, computeFootprint, MUMBAI_BBOX } from '../district-projector.js';

// ---------------------------------------------------------------------------
// Property 4: SVG projection stays within viewport
//
// For any (lat, lon) within the Mumbai bounding box and any (svgWidth,
// svgHeight) in [100, 2000], projectToSVG returns x ∈ [0, svgWidth] and
// y ∈ [0, svgHeight].
//
// Validates: Requirements 2.1, 2.3
// ---------------------------------------------------------------------------

describe('Property 4: SVG projection stays within viewport', () => {
  test('projectToSVG returns x ∈ [0, svgWidth] and y ∈ [0, svgHeight] for all valid inputs', () => {
    // Feature: mumbai-3d-district-map, Property 4: SVG projection stays within viewport
    fc.assert(
      fc.property(
        fc.float({ min: MUMBAI_BBOX.minLat, max: MUMBAI_BBOX.maxLat, noNaN: true }),
        fc.float({ min: MUMBAI_BBOX.minLon, max: MUMBAI_BBOX.maxLon, noNaN: true }),
        fc.integer({ min: 100, max: 2000 }),
        fc.integer({ min: 100, max: 2000 }),
        (lat, lon, svgWidth, svgHeight) => {
          const { x, y } = projectToSVG(lat, lon, svgWidth, svgHeight);
          return x >= 0 && x <= svgWidth && y >= 0 && y <= svgHeight;
        }
      ),
      { numRuns: 100 }
    );
  });
});

// ---------------------------------------------------------------------------
// Property 7: Footprint formula invariant
//
// For any areaKm2 ≥ 0 and maxAreaKm2 > 0, computeFootprint returns a value
// in the closed interval [4, 14].
//
// Validates: Requirements 3.3
// ---------------------------------------------------------------------------

describe('Property 7: Footprint formula invariant', () => {
  test('computeFootprint returns a value in [4, 14] for all valid inputs', () => {
    // Feature: mumbai-3d-district-map, Property 7: Footprint formula invariant
    fc.assert(
      fc.property(
        fc.float({ min: 0, max: 10000, noNaN: true }),
        fc.float({ min: 0.001, max: 10000, noNaN: true }),
        (areaKm2, maxAreaKm2) => {
          const footprint = computeFootprint(areaKm2, maxAreaKm2);
          return footprint >= 4 && footprint <= 14;
        }
      ),
      { numRuns: 100 }
    );
  });
});
