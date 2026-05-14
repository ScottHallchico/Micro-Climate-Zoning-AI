/**
 * district-projector.js
 *
 * Shared module for converting WGS 84 lat/lon coordinates into 2D SVG viewport
 * coordinates or 3D Three.js scene coordinates for the Mumbai district map.
 *
 * Requirements: 2.1, 2.3, 3.1, 3.3, 3.4
 */

/**
 * The canonical bounding box for the Mumbai Metropolitan Region in WGS 84.
 * Used as the projection reference for all coordinate transformations.
 */
export const MUMBAI_BBOX = {
  minLat: 18.89,
  maxLat: 19.27,
  minLon: 72.77,
  maxLon: 73.00,
};

/**
 * Projects a WGS 84 lat/lon coordinate to SVG viewport coordinates using an
 * equirectangular projection anchored to MUMBAI_BBOX.
 *
 * x = ((lon - minLon) / (maxLon - minLon)) * svgWidth
 * y = ((maxLat - lat) / (maxLat - minLat)) * svgHeight
 *
 * @param {number} lat - Latitude in decimal degrees (WGS 84)
 * @param {number} lon - Longitude in decimal degrees (WGS 84)
 * @param {number} svgWidth - Width of the SVG viewport in pixels
 * @param {number} svgHeight - Height of the SVG viewport in pixels
 * @returns {{ x: number, y: number }} SVG coordinate pair
 *
 * Requirements: 2.1, 2.3
 */
export function projectToSVG(lat, lon, svgWidth, svgHeight) {
  const { minLat, maxLat, minLon, maxLon } = MUMBAI_BBOX;
  const x = ((lon - minLon) / (maxLon - minLon)) * svgWidth;
  const y = ((maxLat - lat) / (maxLat - minLat)) * svgHeight;
  return { x, y };
}

/**
 * Projects a WGS 84 lat/lon coordinate to a Three.js scene coordinate using an
 * equirectangular projection anchored to MUMBAI_BBOX, normalised so that the
 * full Mumbai extent maps to the range [−38, +38] on both the X and Z axes.
 *
 * x = ((lon - minLon) / (maxLon - minLon) - 0.5) * 2 * 38
 * z = ((maxLat - lat) / (maxLat - minLat) - 0.5) * 2 * 38
 *
 * @param {number} lat - Latitude in decimal degrees (WGS 84)
 * @param {number} lon - Longitude in decimal degrees (WGS 84)
 * @returns {{ x: number, z: number }} Scene coordinate pair (y is always 0)
 *
 * Requirements: 3.1
 */
export function projectToScene(lat, lon) {
  const { minLat, maxLat, minLon, maxLon } = MUMBAI_BBOX;
  const x = ((lon - minLon) / (maxLon - minLon) - 0.5) * 2 * 38;
  const z = ((maxLat - lat) / (maxLat - minLat) - 0.5) * 2 * 38;
  return { x, z };
}

/**
 * Resolves spatial collisions between district blocks in the 3D scene.
 *
 * Iterates all pairs of blocks. If two projected centroids are closer than
 * `minSeparation` scene units, the block with the smaller `area_km2` is
 * translated by the minimum vector required to achieve the target separation.
 * The resolved position is stored on the block as `_scene_x` and `_scene_z`.
 *
 * Mutates the blocks array in-place.
 *
 * @param {Array<Object>} blocks - Array of District_Block objects. Each block
 *   must have `centroid_lat`, `centroid_lon`, `area_km2`, and `district_name`.
 * @param {number} [minSeparation=3] - Minimum allowed distance (scene units)
 *   between any two block centroids.
 *
 * Requirements: 3.4
 */
export function resolveCollisions(blocks, minSeparation = 3) {
  // Initialise resolved scene positions from geographic centroids (or existing
  // overrides from a previous call).
  for (const block of blocks) {
    if (block._scene_x === undefined || block._scene_z === undefined) {
      const { x, z } = projectToScene(block.centroid_lat, block.centroid_lon);
      block._scene_x = x;
      block._scene_z = z;
    }
  }

  // Iterate all unique pairs and resolve collisions.
  for (let i = 0; i < blocks.length; i++) {
    for (let j = i + 1; j < blocks.length; j++) {
      const a = blocks[i];
      const b = blocks[j];

      const dx = b._scene_x - a._scene_x;
      const dz = b._scene_z - a._scene_z;
      const dist = Math.sqrt(dx * dx + dz * dz);

      if (dist < minSeparation) {
        // Determine which block to move: the one with the smaller area_km2.
        const areaA = a.area_km2 ?? 0;
        const areaB = b.area_km2 ?? 0;
        const moveBlock = areaA <= areaB ? a : b;
        const fixedBlock = areaA <= areaB ? b : a;

        // Compute the translation vector needed to push `moveBlock` to exactly
        // `minSeparation` units away from `fixedBlock`.
        let tx, tz;
        if (dist === 0) {
          // Coincident centroids — push along the +X axis as a tiebreaker.
          tx = minSeparation;
          tz = 0;
        } else {
          // Direction from fixedBlock to moveBlock (or opposite if moveBlock is b).
          const dirX = (moveBlock._scene_x - fixedBlock._scene_x) / dist;
          const dirZ = (moveBlock._scene_z - fixedBlock._scene_z) / dist;
          const overlap = minSeparation - dist;
          tx = dirX * overlap;
          tz = dirZ * overlap;
        }

        console.log(
          `[resolveCollisions] Collision between "${fixedBlock.district_name}" and ` +
          `"${moveBlock.district_name}": original separation = ${dist.toFixed(4)} scene units. ` +
          `Translating "${moveBlock.district_name}" by (${tx.toFixed(4)}, ${tz.toFixed(4)}).`
        );

        moveBlock._scene_x += tx;
        moveBlock._scene_z += tz;
      }
    }
  }
}

/**
 * Computes the ground footprint size (in scene units per side) for a district
 * block, scaled proportionally to its area relative to the largest district.
 *
 * Formula: clamp(sqrt(areaKm2 / maxAreaKm2) × 14, 4, 14)
 *
 * @param {number} areaKm2 - Area of the district in km²
 * @param {number} maxAreaKm2 - Area of the largest district in km²
 * @returns {number} Footprint size in scene units, clamped to [4, 14]
 *
 * Requirements: 3.3
 */
export function computeFootprint(areaKm2, maxAreaKm2) {
  if (maxAreaKm2 <= 0) {
    return 4;
  }
  return Math.max(4, Math.min(14, Math.sqrt(areaKm2 / maxAreaKm2) * 14));
}
