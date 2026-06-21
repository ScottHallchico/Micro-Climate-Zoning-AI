# Phase UI-B: NYC Dataset Provenance Audit

## 1. Dataset Origin & Generation Method
*   **Target File:** `src/frontend/nyc_buildings.json`
*   **Origin / Pipeline:** The file was not generated via a local ETL script. It was directly downloaded via curl from a public visualization repository (`https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/trips/buildings.json`).
*   **Truncation Logic:** The source URL explicitly hosts a truncated/sample dataset meant for lightweight browser examples in the `deck.gl` repository.

## 2. NYC Building Dataset Inventory
A full scan of the workspace revealed the following core architectural datasets:
*   `data/raw/BUILDING_20260602.geojson`
    *   **File Size:** 962 MB
    *   **Feature Count:** 1,082,945 buildings
*   `src/frontend/nyc_buildings.json`
    *   **File Size:** 730 KB
    *   **Feature Count:** 999 buildings

## 3. Count Comparison & Retention
*   **Source Dataset Count:** 1,082,945 buildings
*   **Frontend Dataset Count:** 999 buildings
*   **Retention Percentage:** `0.0922%`

## 4. Final Certification
**Classification: C — Demo Sample / Truncated Export**

### Deliverable Summary
The current frontend is **not** rendering the complete Manhattan dataset. It is rendering a minuscule, hardcoded sample representing only `0.09%` of the actual NYC building inventory. The comprehensive dataset does exist on the server (`BUILDING_20260602.geojson`), but migrating it to the frontend necessitates a robust spatial tiling and streaming pipeline, which has been detailed in `reports/full_nyc_frontend_migration.md`.
