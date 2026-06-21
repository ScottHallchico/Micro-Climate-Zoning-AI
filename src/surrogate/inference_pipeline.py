import numpy as np
from .predictor import SurrogatePredictor

class InferencePipeline:
    def __init__(self, model_path):
        self.predictor = SurrogatePredictor(model_path)
        
    def run(self, coords, ws, wd):
        vel = self.predictor.predict(coords, ws, wd)
        # Compute WSI (Wake Severity Index) as ratio of points < 0.3*ws
        wsi = np.mean(vel < 0.3 * ws)
        
        # Corridors: points where velocity > 0.8 * ws
        corridors = vel > 0.8 * ws
        
        # Zoning classification
        if wsi > 0.5:
            zone = "Heat Stress / Wake Zone"
        elif np.mean(corridors) > 0.2:
            zone = "Ventilation Corridor Zone"
        else:
            zone = "Mixed Microclimate Zone"
            
        return {
            'velocity': vel,
            'wsi': wsi,
            'corridors': corridors,
            'zone': zone
        }
