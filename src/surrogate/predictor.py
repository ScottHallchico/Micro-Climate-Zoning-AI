import joblib
from .feature_builder import FeatureBuilder

class SurrogatePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.builder = FeatureBuilder()
        
    def predict(self, coords, ws, wd):
        X = self.builder.build(coords, ws, wd)
        return self.model.predict(X)
        
    def get_feature_importances(self):
        return dict(zip(self.builder.f_names, self.model.feature_importances_))
