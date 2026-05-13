# How to Run the MicroClimate Zoning AI Project

This project uses two local servers:

- Backend API: FastAPI on port `8000`
- Frontend dashboard: static dashboard on port `3000`

## 1. Start the Backend

Open PowerShell terminal 1:

```powershell
cd "C:\Users\jprit\OneDrive\Desktop\Main EL"
python -m uvicorn src.governance.api:app --reload
```

Keep this terminal open.

Check backend health:

```text
http://127.0.0.1:8000/v1/health
```

Check actual Mumbai data endpoint:

```text
http://127.0.0.1:8000/v1/data/actual?mode=design_peak
```

## 2. Start the Frontend

Open PowerShell terminal 2:

```powershell
cd "C:\Users\jprit\OneDrive\Desktop\Main EL\src\dashboard"
python -m http.server 3000
```

Keep this terminal open.

Open the dashboard:

```text
http://127.0.0.1:3000
```

Open the 3D map:

```text
http://127.0.0.1:3000/3d-map.html
```

## If `python` Does Not Work

Use `py` instead:

```powershell
py -m uvicorn src.governance.api:app --reload
```

```powershell
py -m http.server 3000
```

## Data Used

The project is currently configured for a Mumbai study area.

Actual data files:

- `data/actual_weather_measurements.csv`
- `data/actual_blocks.csv`
- `data/actual_data_sources.md`

Data sources:

- Open-Meteo for Mumbai weather
- OpenStreetMap for Mumbai building, road, and green-feature structure

## Notes

- The dashboard loads actual Mumbai data from the backend.
- The 3D map reads the same block state as the dashboard.
- If you change height, albedo, or green cover on the dashboard, refresh the 3D map to see the updated block values.
- Wind-corridor height overrides are allowed, but the block is marked `Non-compliant`.
- PINN is implemented and connected, but it is not yet trained on real OpenFOAM or satellite thermal datasets.

