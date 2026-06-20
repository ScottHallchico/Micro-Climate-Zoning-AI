# Feature Representation Audit

Evaluated candidate morphological features using Random Forest importance.

| Feature | Importance |
|---|---|
| z | 0.1812 |
| y | 0.1790 |
| Orientation Proxy | 0.1476 |
| Canyon Aspect Ratio | 0.1406 |
| Wind Alignment | 0.1206 |
| Frontal Area Density | 0.1196 |
| Central Distance | 0.0557 |
| x | 0.0556 |

**Conclusion**: Geometric coordinates alone are insufficient. Derived fluid-blockage features (Aspect Ratio, Frontal Density) must be explicitly fed into the node features.
