# API Reference

## `POST /v2/surrogate/predict`
**Inputs:** `coords` (N,3), `ws` (float), `wd` (float)
**Outputs:** 
- `wsi` (float)
- `corridor_score` (float)
- `route` ("SURROGATE" or "CFD")
- `confidence` (float)

## `POST /v2/surrogate/validate`
**Inputs:** `job_id`
**Outputs:** Status of CFD Fallback queue.
