from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import zlib
from typing import List, Optional
from src.services.surrogate_api import SurrogateAPI
from src.services.cfd_dispatcher import CFDDispatcher
import os

app = FastAPI(title="Micro-Climate Zoning AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.getenv("MODEL_PATH", "models/production/lightgbm_production.joblib")
surrogate = None
dispatcher = CFDDispatcher()

@app.on_event("startup")
def startup_event():
    global surrogate
    if os.path.exists(model_path):
        surrogate = SurrogateAPI(model_path)
    else:
        print(f"Warning: Model not found at {model_path}")

class PredictRequest(BaseModel):
    coords: List[List[float]]
    ws: float
    wd: float

@app.post("/v2/surrogate/predict")
def predict(req: PredictRequest):
    if surrogate is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return surrogate.predict(req.coords, req.ws, req.wd)

@app.get("/v2/surrogate/validate")
def validate(job_id: str):
    return dispatcher.get_job_status(job_id)

@app.get("/")
def read_root():
    return {
        "service": "Micro-Climate Zoning AI API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": surrogate is not None}

@app.get("/tiles/buildings/{z}/{x}/{y}")
async def get_building_tile(z: int, x: int, y: int):
    mbtiles_path = "data/tiles/buildings.mbtiles"
    if not os.path.exists(mbtiles_path):
        raise HTTPException(status_code=404, detail="Tileset not found")
    
    # Convert standard XYZ to TMS (Inverted Y)
    tms_y = (1 << z) - 1 - y
    
    try:
        conn = sqlite3.connect(mbtiles_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
            (z, x, tms_y)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return Response(content=b"", status_code=204) # No Content
        
        tile_data = row[0]
        headers = {
            "Content-Type": "application/x-protobuf",
            "Content-Encoding": "gzip",
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=86400"
        }
        return Response(content=tile_data, headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
