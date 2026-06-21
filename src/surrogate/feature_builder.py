import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

class FeatureBuilder:
    def __init__(self):
        self.f_names = [
            'ws', 'wd', 'x_norm', 'y_norm', 'z_norm', 'dist_center', 'dist_tallest',
            'svf_25m', 'svf_50m', 'svf_100m',
            'upwind_count', 'upwind_hm', 'upwind_hmax', 'upwind_h_grad', 'upwind_frontal',
            'downwind_open', 'cross_dens', 'cross_hvar',
            'canyon_w', 'canyon_h', 'canyon_hw', 'canyon_align', 'street_open',
            'dens_25', 'hm_25', 'hmax_25', 'hvar_25',
            'dens_50', 'hm_50', 'hmax_50', 'hvar_50',
            'dens_100', 'hm_100', 'hmax_100', 'hvar_100',
            'dens_200', 'hm_200', 'hmax_200', 'hvar_200'
        ]
        
    def build(self, coords, ws, wd):
        coords = np.asarray(coords, dtype=np.float64)
        N = len(coords)
        if N == 0: return np.zeros((0, len(self.f_names)))
        
        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        z_max = coords[:,2].max()
        
        x_norm = (coords[:,0] - x_min) / (x_max - x_min + 1e-6)
        y_norm = (coords[:,1] - y_min) / (y_max - y_min + 1e-6)
        z_norm = coords[:,2] / (z_max + 1e-6)
        x_c, y_c = (x_max+x_min)/2, (y_max+y_min)/2
        dist_center = np.linalg.norm(coords[:,:2] - np.array([x_c, y_c]), axis=1)
        dist_tallest = np.linalg.norm(coords[:,:2] - coords[np.argmax(coords[:,2]), :2], axis=1)
        
        base_feats = np.column_stack([np.full(N, ws), np.full(N, wd), x_norm, y_norm, z_norm, dist_center, dist_tallest])
        
        tree = cKDTree(coords[:, :2])
        z_arr = coords[:, 2]
        
        def get_multi_scale(r):
            idxs = tree.query_ball_point(coords[:, :2], r)
            dens = np.array([len(idx) for idx in idxs])
            hm = np.array([np.mean(z_arr[idx]) if len(idx)>0 else z for z, idx in zip(z_arr, idxs)])
            hmax = np.array([np.max(z_arr[idx]) if len(idx)>0 else z for z, idx in zip(z_arr, idxs)])
            hvar = np.array([np.var(z_arr[idx]) if len(idx)>0 else 0.0 for idx in idxs])
            return dens, hm, hmax, hvar

        d25, hm25, hmax25, hv25 = get_multi_scale(25)
        d50, hm50, hmax50, hv50 = get_multi_scale(50)
        d100, hm100, hmax100, hv100 = get_multi_scale(100)
        d200, hm200, hmax200, hv200 = get_multi_scale(200)
        
        ms_feats = np.column_stack([
            d25, hm25, hmax25, hv25,
            d50, hm50, hmax50, hv50,
            d100, hm100, hmax100, hv100,
            d200, hm200, hmax200, hv200
        ])
        
        def get_svf(r):
            idxs = tree.query_ball_point(coords[:, :2], r)
            svf = []
            for z, idx in zip(z_arr, idxs):
                if len(idx) <= 1: svf.append(1.0)
                else: svf.append(1.0 - np.sum(z_arr[idx] > z) / len(idx))
            return np.array(svf)
            
        svf_feats = np.column_stack([get_svf(25), get_svf(50), get_svf(100)])
        
        rad = np.radians(wd)
        wvec = np.array([np.cos(rad), np.sin(rad)])
        
        up_c, up_hm, up_hmax, up_hgrad, up_front = [], [], [], [], []
        dn_open, cr_dens, cr_hvar = [], [], []
        
        idxs_50 = tree.query_ball_point(coords[:, :2], 50.0)
        for i, idx in enumerate(idxs_50):
            if len(idx) <= 1:
                up_c.append(0); up_hm.append(z_arr[i]); up_hmax.append(z_arr[i]); up_hgrad.append(0); up_front.append(0)
                dn_open.append(1.0); cr_dens.append(0); cr_hvar.append(0.0)
                continue
                
            nbrs2d = coords[idx, :2]
            nbrz = z_arr[idx]
            vecs = nbrs2d - coords[i, :2]
            dists = np.linalg.norm(vecs, axis=1) + 1e-6
            align = (vecs @ wvec) / dists
            
            up_mask = align < -0.5
            dn_mask = align > 0.5
            cr_mask = np.abs(align) <= 0.5
            
            upz = nbrz[up_mask]
            up_c.append(len(upz))
            up_hm.append(np.mean(upz) if len(upz)>0 else z_arr[i])
            up_hmax.append(np.max(upz) if len(upz)>0 else z_arr[i])
            up_hgrad.append(np.mean(np.maximum(0, upz - z_arr[i])) if len(upz)>0 else 0)
            up_front.append(np.sum(upz) / 50.0)
            
            dn_open.append(1.0 - len(nbrz[dn_mask])/len(idx))
            
            crz = nbrz[cr_mask]
            cr_dens.append(len(crz))
            cr_hvar.append(np.var(crz) if len(crz)>0 else 0.0)
            
        wind_feats = np.column_stack([up_c, up_hm, up_hmax, up_hgrad, up_front, dn_open, cr_dens, cr_hvar])
        
        can_w = 50.0 / (np.array(cr_dens) + 1)
        cr_hm = []
        for i, idx in enumerate(idxs_50):
            if len(idx) > 1:
                align_cr = np.abs((coords[idx,:2] - coords[i,:2]) @ wvec / (np.linalg.norm(coords[idx,:2]-coords[i,:2])+1e-6))
                cr_z = z_arr[idx][align_cr <= 0.5]
                cr_hm.append(np.mean(cr_z) if len(cr_z)>0 else z_arr[i])
            else:
                cr_hm.append(z_arr[i])
        can_h = np.array(cr_hm)
        can_hw = can_h / can_w
        can_align = np.array(cr_dens) / (np.array(up_c) + 1e-6)
        st_open = svf_feats[:, 0] * np.array(dn_open)
        
        canyon_feats = np.column_stack([can_w, can_h, can_hw, can_align, st_open])
        
        return np.column_stack([base_feats, svf_feats, wind_feats, canyon_feats, ms_feats]).astype(np.float32)
