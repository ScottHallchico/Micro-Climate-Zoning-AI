# Research Publication Appendix: CFD Preprocessing Methodology

*Generated: 2026-06-14 01:36:53*

## S1. Geometry Generation Methodology
Building footprints sourced from MapPLUTO were projected to UTM Zone 18N (EPSG:32618) and extruded vertically using corresponding structural heights via the Mapbox Earcut triangulation engine. All geometry was validated to ensure strict watertight manifold topologies necessary for snappyHexMesh boundary layer insertion.

## S2. Terrain Methodology
A continuous terrain surface was generated utilizing a Delaunay triangulation across all internal building centroids mapped against their exact geographical ground elevations. The bounds were extended and leveled symmetrically to secure uniform flow limits along the computational domain boundaries.

## S3. Vegetation Methodology
Urban tree canopy obstructions were modeled via empirically derived heuristics scaling trunk diameter at breast height (DBH) to canopy radiuses and total elevations. Tree trunks are treated as solid slip boundaries, while spherical canopies are explicitly separated for explicit porous-media momentum sink parameterization (employing a uniform drag coefficient of 0.2 and LAD of 1.5 m²/m³).

## S4. ERA5 Extraction Methodology
Climatological boundary conditions were aggregated from the ECMWF ERA5 hourly dataset. The closest spatial grid cell for each respective urban archetype was isolated. Prevailing wind vectors were derived utilizing true circular magnitude-vector averaging to establish a non-biased seasonal flow profile alongside discrete 95th-percentile high-wind event isolations.

## S5. Wind Alignment Methodology
To optimize the rectangular computational fluid dynamics domain, all physical geometries (buildings, terrain, and vegetation) underwent a mathematical rotation matrix transformation about the local centroid prior to extrusion. The geometries were structurally rotated so that the prevailing meteorological wind flow enters normal to the `-X` boundary and propagates parallel to the `+X` axis, eliminating oblique boundary inflow errors.