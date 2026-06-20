import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ML_DIR = PROJECT_ROOT / "data" / "ml"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "publication_figures"

class MCDropoutPINN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # For MC Dropout
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_features)
        )
        
    def forward(self, x):
        return self.net(x)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main():
    print("Phase 8A.3 Scientific Hardening - Surrogate PINN")
    df = pd.read_parquet(ML_DIR / "cfd_field_dataset_verified.parquet")
    cdf = pd.read_parquet(ML_DIR / "verified_cfd_dataset_v3.parquet")
    
    df = df.merge(cdf[['simulation_id', 'wind_speed', 'wind_direction']], on='simulation_id')
    
    # Simple feature set
    features = ['x', 'y', 'z', 'wind_speed', 'wind_direction']
    targets = ['u', 'v', 'w', 'p', 'k']
    
    # 8A.3.1 - Holdout Split
    sim_ids = df['simulation_id'].unique()
    np.random.seed(42)
    np.random.shuffle(sim_ids)
    
    n_train = int(len(sim_ids) * 0.8)
    train_ids = sim_ids[:n_train]
    test_ids = sim_ids[n_train:]
    
    train_df = df[df['simulation_id'].isin(train_ids)]
    test_df = df[df['simulation_id'].isin(test_ids)]
    
    X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[targets].values, dtype=torch.float32)
    X_test = torch.tensor(test_df[features].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[targets].values, dtype=torch.float32)
    
    # Standardize
    x_mean, x_std = X_train.mean(0), X_train.std(0) + 1e-6
    y_mean, y_std = y_train.mean(0), y_train.std(0) + 1e-6
    
    X_train_s = (X_train - x_mean) / x_std
    y_train_s = (y_train - y_mean) / y_std
    X_test_s = (X_test - x_mean) / x_std
    y_test_s = (y_test - y_mean) / y_std
    
    model = MCDropoutPINN(len(features), len(targets))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("Training surrogate...")
    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        pred = model(X_train_s)
        loss = nn.MSELoss()(pred, y_train_s)
        loss.backward()
        optimizer.step()
        
    print("Predicting with MC Dropout...")
    # 8A.3.4 Uncertainty Calibration (MC Dropout)
    model.eval()
    enable_dropout(model)
    preds = []
    for _ in range(30):
        with torch.no_grad():
            preds.append(model(X_test_s).numpy())
            
    preds = np.array(preds)
    mean_preds_s = np.mean(preds, axis=0)
    std_preds_s = np.std(preds, axis=0)
    
    mean_preds = mean_preds_s * y_std.numpy() + y_mean.numpy()
    std_preds = std_preds_s * y_std.numpy()
    
    true_vals = y_test.numpy()
    
    # 8A.3.1 Out-of-Sample Metrics
    md_oos = "# PINN Out-of-Sample Validation\n\n"
    md_oos += f"- **Training Simulations:** {len(train_ids)}\n"
    md_oos += f"- **Test Simulations:** {len(test_ids)}\n\n"
    md_oos += "| Variable | MAE | RMSE | R2 |\n"
    md_oos += "|---|---|---|---|\n"
    
    metrics = {}
    for i, t in enumerate(targets):
        mae = mean_absolute_error(true_vals[:, i], mean_preds[:, i])
        rmse = np.sqrt(mean_squared_error(true_vals[:, i], mean_preds[:, i]))
        r2 = r2_score(true_vals[:, i], mean_preds[:, i])
        md_oos += f"| {t} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |\n"
        metrics[t] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
    (REPORTS_DIR / "pinn_out_of_sample_validation.md").write_text(md_oos)
    
    # 8A.3.4 Uncertainty Calibration
    md_unc = "# PINN Uncertainty Calibration\n\n"
    md_unc += "Using Monte Carlo Dropout (30 forward passes).\n\n"
    md_unc += "| Variable | Variance-Error Correlation | Status |\n"
    md_unc += "|---|---|---|\n"
    
    for i, t in enumerate(targets):
        error = np.abs(true_vals[:, i] - mean_preds[:, i])
        unc = std_preds[:, i]
        if np.std(unc) > 1e-6 and np.std(error) > 1e-6:
            corr, _ = pearsonr(error, unc)
        else:
            corr = 0.0
        status = "PASS" if corr > 0.60 else "WARN"
        md_unc += f"| {t} | {corr:.3f} | {status} |\n"
        
    (REPORTS_DIR / "pinn_uncertainty_calibration.md").write_text(md_unc)
    
    # 8A.3.3 Benchmark
    md_bench = "# CFD vs PINN Benchmark\n\n"
    md_bench += "Aggregated surrogate fidelity across all archetypes.\n"
    md_bench += md_oos.split("\n\n")[-1] # Add the table
    (REPORTS_DIR / "pinn_vs_cfd_benchmark.md").write_text(md_bench)
    
    # 8A.3.2 OOD Detection
    print("Running Isolation Forest...")
    iso = IsolationForest(contamination=0.05, random_state=42)
    # Fit on train
    iso.fit(X_train_s.numpy())
    
    df['ood_score'] = iso.score_samples(((torch.tensor(df[features].values, dtype=torch.float32) - x_mean)/x_std).numpy())
    # OOD class: -1 is outlier, 1 is inlier. We use score_samples for a continuous metric.
    # Higher score -> more normal. Lower score -> more anomalous.
    
    def classify_ood(s):
        if s > -0.5: return "In Distribution"
        if s > -0.65: return "Borderline"
        return "Out of Distribution"
        
    df['ood_class'] = df['ood_score'].apply(classify_ood)
    df['prediction_confidence'] = 1.0 / (1.0 + np.exp(-df['ood_score'])) # Sigmoid scaled
    
    df[['simulation_id', 'x', 'y', 'z', 'ood_score', 'ood_class', 'prediction_confidence']].to_parquet(ML_DIR / "ood_predictions.parquet")
    
    md_ood = "# OOD Detection\n\n"
    md_ood += "Isolation Forest trained on base feature distribution.\n"
    md_ood += f"- **In Distribution:** {len(df[df['ood_class']=='In Distribution'])}\n"
    md_ood += f"- **Borderline:** {len(df[df['ood_class']=='Borderline'])}\n"
    md_ood += f"- **Out of Distribution:** {len(df[df['ood_class']=='Out of Distribution'])}\n"
    (REPORTS_DIR / "pinn_ood_detection.md").write_text(md_ood)
    
    print("Done.")

if __name__ == "__main__":
    main()
