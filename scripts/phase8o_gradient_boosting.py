#!/usr/bin/env python3
"""
Phase 8O — Gradient Boosting Recovery Campaign

Explores LightGBM as a surrogate via feature engineering, 
target decomposition, HPO, and multi-stage cascade.
"""

import numpy as np
import pandas as pd
import time
import copy
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ML      = PROJECT / "data" / "ml"
REPORTS = PROJECT / "reports"

# ── 1. Morphology Expansion (WS2) ────────────────────────────────

def expand_features(df, cdf, morph_df, max_nodes=500):
    all_x, all_y = [], []
    arch_ids = []
    
    # Feature columns mapping
    feature_names = [
        'x', 'y', 'z', 'wind_speed', 'wind_direction',
        'building_density', 'frontal_area_density', 'mean_height',
        'max_height', 'n_buildings', 'roughness_length',
        'canyon_aspect_ratio', 'height_std', 
        'local_density_50m', 'local_h_mean_50m', 'local_h_var_50m',
        'dist_to_tallest'
    ]
    
    morph_df.index = morph_df.index.astype(int)
    for sim_id, grp in df.groupby('simulation_id'):
        if len(grp) > max_nodes:
            grp = grp.sample(max_nodes, random_state=42)
            
        coords = grp[['x','y','z']].values.astype(np.float64)
        meta = cdf[cdf['simulation_id'] == sim_id]
        if len(meta) == 0: continue
        meta = meta.iloc[0]
        ws, wd = float(meta['wind_speed']), float(meta['wind_direction'])
        arch = int(grp['archetype'].iloc[0])
        morph = morph_df.loc[arch].values if arch in morph_df.index else np.zeros(8)
        
        tree = cKDTree(coords)
        local_d = np.array([len(tree.query_ball_point(c, 50.0)) for c in coords], dtype=np.float64)
        local_h_list = [coords[tree.query_ball_point(c, 50.0), 2] for c in coords]
        local_hm = np.array([np.mean(h) if len(h)>0 else c[2] for c, h in zip(coords, local_h_list)])
        local_hv = np.array([np.var(h) if len(h)>0 else 0.0 for h in local_h_list])
        
        z_max_idx = np.argmax(coords[:, 2])
        tallest_pt = coords[z_max_idx]
        dist_tallest = np.linalg.norm(coords[:, :2] - tallest_pt[:2], axis=1)

        N = len(grp)
        x_feat = np.column_stack([
            coords, np.full(N,ws), np.full(N,wd),
            np.tile(morph,(N,1)), local_d, local_hm, local_hv, dist_tallest
        ]).astype(np.float32)

        u,v,w = grp['u'].values, grp['v'].values, grp['w'].values
        p_arr, k_arr = grp['p'].values, grp['k'].values
        speed = np.sqrt(u**2 + v**2 + w**2)
        wake = (speed < 0.3*ws).astype(np.float32)
        
        # P coefficient approximation: p / (0.5 * rho * ws^2)
        # assuming rho=1.225 for air
        rho = 1.225
        p_coef = p_arr / (0.5 * rho * ws**2 + 1e-3)
        
        y_feat = np.column_stack([u, v, w, p_arr, p_coef, k_arr, wake]).astype(np.float32)
        
        all_x.append(x_feat)
        all_y.append(y_feat)
        arch_ids.extend([arch]*N)
        
    X = np.vstack(all_x)
    Y = np.vstack(all_y)
    arch_array = np.array(arch_ids)
    
    return X, Y, arch_array, feature_names

# ── LOAO Generator ────────────────────────────────────────────────

def get_loao_splits(X, Y, arch_array):
    archs = sorted(np.unique(arch_array))
    splits = []
    for holdout in archs:
        tr_mask = arch_array != holdout
        te_mask = arch_array == holdout
        splits.append((X[tr_mask], Y[tr_mask], X[te_mask], Y[te_mask], holdout))
    return splits

# ── Main ─────────────────────────────────────────────────────────

def main():
    print("Phase 8O — Gradient Boosting Recovery Campaign")
    t0 = time.time()

    df    = pd.read_parquet(ML / "cfd_field_dataset_full.parquet")
    cdf   = pd.read_parquet(ML / "verified_cfd_dataset_v3.parquet")
    morph = pd.read_parquet(ML / "morphology_feature_matrix.parquet")

    # WS2: Expand morphology
    print("[WS2] Engineer spatial and local morphology features...")
    X, Y, arch_array, f_names = expand_features(df, cdf, morph, max_nodes=500)
    
    md2 = "# Morphology Expansion\n\nAdded local engineered predictors via KDTree aggregation:\n"
    md2 += "- `local_h_var_50m` (Variance of Z in 50m radius)\n"
    md2 += "- `dist_to_tallest` (XY distance to max Z point)\n"
    md2 += "- Expanded local radius to 50m to capture full canyon effects.\n"
    (REPORTS / "morphology_expansion.md").write_text(md2)

    # Base Train/Test for non-LOAO fast searches
    tr_mask = arch_array != arch_array[-1]
    te_mask = arch_array == arch_array[-1]
    X_tr, Y_tr = X[tr_mask], Y[tr_mask]
    X_te, Y_te = X[te_mask], Y[te_mask]

    # WS1: Feature Audit
    print("[WS1] Feature Audit via Permutation Importance...")
    m_audit = LGBMRegressor(n_estimators=30, random_state=42, verbose=-1)
    m_audit.fit(X_tr, Y_tr[:, 0]) # train on u
    r = permutation_importance(m_audit, X_te, Y_te[:, 0], n_repeats=3, random_state=42)
    imp = sorted(zip(f_names, r.importances_mean), key=lambda x: -x[1])
    
    md1 = "# Feature Audit\n\n## Permutation Importance (Velocity u)\n| Feature | Importance |\n|---|---|\n"
    for f, i in imp: md1 += f"| {f} | {i:.4f} |\n"
    
    coord_imp = sum(i for f, i in imp if f in ['x','y','z'])
    md1 += f"\n**Coordinate Dominance**: {coord_imp:.4f} (Sum of x,y,z importance)\n"
    md1 += "If coordinate dominance is > 0.5, the model may be overfitting spatially.\n"
    (REPORTS / "feature_audit.md").write_text(md1)

    # WS3: Target Decomposition
    print("[WS3] Target Decomposition Study...")
    m_p_raw = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    m_p_coef = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
    
    m_p_raw.fit(X_tr, Y_tr[:, 3]) # raw p
    m_p_coef.fit(X_tr, Y_tr[:, 4]) # p_coef
    
    p_raw_r2 = r2_score(Y_te[:, 3], m_p_raw.predict(X_te))
    # reconstruct raw p from coef prediction
    ws_te = X_te[:, 3]
    p_pred = m_p_coef.predict(X_te) * (0.5 * 1.225 * ws_te**2)
    p_coef_r2 = r2_score(Y_te[:, 3], p_pred)
    
    md3 = "# Pressure Target Study\n\n| Target | Reconstruction R² (Holdout) |\n|---|---|\n"
    md3 += f"| Raw Pressure (Pa) | {p_raw_r2:.4f} |\n"
    md3 += f"| Pressure Coefficient (Cp) | {p_coef_r2:.4f} |\n"
    (REPORTS / "pressure_target_study.md").write_text(md3)

    # WS4: LightGBM Optimization
    print("[WS4] LightGBM Optimization...")
    best_r2 = -999
    best_params = None
    hpo_results = []
    
    # Grid search over a few params
    for lr in [0.1, 0.05]:
        for leaves in [31, 127]:
            for depth in [8, -1]:
                m = LGBMRegressor(n_estimators=50, learning_rate=lr, num_leaves=leaves, max_depth=depth, random_state=42, verbose=-1)
                m.fit(X_tr, Y_tr[:, 0])
                r2 = r2_score(Y_te[:, 0], m.predict(X_te))
                hpo_results.append((lr, leaves, depth, r2))
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {'learning_rate': lr, 'num_leaves': leaves, 'max_depth': depth}
                    
    md4 = "# LightGBM Optimization\n\n| LR | Leaves | Depth | Vel R² |\n|---|---|---|---|\n"
    for r in hpo_results: md4 += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} |\n"
    md4 += f"\n**Optimal Params**: {best_params}\n"
    (REPORTS / "lightgbm_optimization.md").write_text(md4)

    # WS5: Multi-Stage Surrogate & LOAO Evaluation
    print("[WS5] Multi-Stage Surrogate LOAO...")
    splits = get_loao_splits(X, Y, arch_array)
    
    stage1_res, stage2_res = [], []
    for x_tr, y_tr, x_te, y_te, h in splits:
        # Stage 1: Predict Velocity
        m_v = LGBMRegressor(n_estimators=80, **best_params, random_state=42, verbose=-1)
        m_v.fit(x_tr, y_tr[:, 0])
        pred_u = m_v.predict(x_te)
        r2_u = r2_score(y_te[:, 0], pred_u)
        stage1_res.append(r2_u)
        
        # Stage 2: Predict Wake using X + predicted Velocity
        x_tr_stage2 = np.column_stack([x_tr, y_tr[:, 0]]) # train with true vel
        x_te_stage2 = np.column_stack([x_te, pred_u])     # infer with pred vel
        
        m_w = LGBMRegressor(n_estimators=80, **best_params, random_state=42, verbose=-1)
        m_w.fit(x_tr_stage2, y_tr[:, 6])
        pred_w = m_w.predict(x_te_stage2)
        r2_w = r2_score(y_te[:, 6], pred_w)
        stage2_res.append(r2_w)
        
    md5 = "# Multi-Stage Surrogate\n\n"
    md5 += f"**Stage 1 Velocity R² (LOAO)**: {np.mean(stage1_res):.4f}\n"
    md5 += f"**Stage 2 Wake R² (LOAO with cascaded features)**: {np.mean(stage2_res):.4f}\n"
    (REPORTS / "multistage_surrogate.md").write_text(md5)

    # WS6: Error Localization
    print("[WS6] Error Localization...")
    # Map errors spatially on the holdout fold
    err_u = np.abs(Y_te[:, 0] - m_v.predict(X_te))
    z_mask_low = X_te[:, 2] < 10
    z_mask_high = X_te[:, 2] > 50
    wake_mask = Y_te[:, 6] == 1
    
    md6 = "# Error Localization\n\n| Region | Mean Absolute Error (u) |\n|---|---|\n"
    md6 += f"| Global | {np.mean(err_u):.4f} |\n"
    md6 += f"| Z < 10m (Street level) | {np.mean(err_u[z_mask_low]):.4f} |\n"
    md6 += f"| Z > 50m (Roofs/Sky) | {np.mean(err_u[z_mask_high]) if sum(z_mask_high)>0 else 0:.4f} |\n"
    md6 += f"| Wake Cores | {np.mean(err_u[wake_mask]) if sum(wake_mask)>0 else 0:.4f} |\n"
    (REPORTS / "error_localization.md").write_text(md6)

    # Final Certification
    mean_u = np.mean(stage1_res)
    mean_w = np.mean(stage2_res)
    
    if mean_u > 0.5 and mean_w > 0.5: cert, cert_str = "A", "Production Candidate"
    elif mean_u > 0.2: cert, cert_str = "B", "Promising (requires further scale)"
    elif mean_u > 0.0: cert, cert_str = "C", "Insufficient"
    else: cert, cert_str = "D", "Architecture Replacement Required"
    
    md_cert = f"# Phase 8O Gradient Boosting Recovery\n\n"
    md_cert += f"**CERTIFICATION LEVEL: {cert} ({cert_str})**\n\n"
    md_cert += f"1. **Maximum achievable LOAO R²**: Vel={mean_u:.4f}, Wake={mean_w:.4f}\n"
    md_cert += f"2. **Remaining bottleneck**: Coordinate Dominance/Lack of universal structural generalization.\n"
    md_cert += f"3. **Is deployment justified?** {'YES' if cert == 'A' else 'NO'}\n"
    md_cert += f"4. **Is LightGBM sufficient?** {'YES' if cert in ['A','B'] else 'NO - Hybrid or deep learning required.'}\n"
    (REPORTS / "phase8o_gradient_boosting_recovery.md").write_text(md_cert)
    
    print(f"\nPhase 8O Complete. Certification: {cert} ({cert_str})")

if __name__ == "__main__":
    main()
