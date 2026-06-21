import numpy as np
from src.surrogate.inference_pipeline import InferencePipeline

class SurrogateAPI:
    def __init__(self, model_path):
        self.pipeline = InferencePipeline(model_path)
        # Mock training centroids for epistemic uncertainty calculation
        self.train_centroid = np.random.randn(39) 
        
    def _compute_confidence(self, X_features):
        # Epistemic confidence proxy based on distance to known distribution
        dists = np.linalg.norm(X_features - self.train_centroid, axis=1)
        mean_dist = np.mean(dists)
        # Scale to [0, 1]
        confidence = np.clip(1.0 - (mean_dist / 100.0), 0.0, 1.0)
        
        # Inject OOD detection
        ood_flag = confidence < 0.80
        
        return float(confidence), bool(ood_flag)

    def predict(self, coords, ws, wd):
        # 1. Feature Generation & Prediction
        X_features = self.pipeline.predictor.builder.build(coords, ws, wd)
        if len(X_features) == 0:
            raise ValueError("No valid coordinates provided")
            
        vel = self.pipeline.predictor.model.predict(X_features)
        
        # 2. Epistemic Confidence Estimation
        confidence, is_ood = self._compute_confidence(X_features)
        
        # 3. Post-processing outputs
        wsi = float(np.mean(vel < 0.3 * ws))
        corridor_score = float(np.mean(vel > 0.8 * ws))
        uhi_risk = wsi * 1.5
        
        # 4. Confidence Routing Decision
        if confidence >= 0.80:
            route = "SURROGATE"
            route_reason = "High epistemic confidence"
        else:
            route = "CFD"
            route_reason = "Low confidence OOD sample detected"
            
        return {
            "route": route,
            "route_reason": route_reason,
            "predictions": {
                "wsi": wsi,
                "corridor_score": corridor_score,
                "uhi_risk": uhi_risk
            },
            "uncertainty": {
                "confidence_epistemic": confidence,
                "ood_detected": is_ood,
                "breakdown": {
                    "feature_distance": float(1.0 - confidence)
                }
            }
        }
