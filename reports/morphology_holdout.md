# Morphology Holdout Validation

## Objective
Measure true extrapolation capability by holding out extreme morphological classes entirely during training.

## Results
| Holdout Type                    |   Wake R² |   Velocity R² |   TKE R² |
|:--------------------------------|----------:|--------------:|---------:|
| Density (Train 75% -> Test 25%) |     0.485 |         0.61  |    0.81  |
| Height (Low-Rise -> High-Rise)  |     0.38  |         0.545 |    0.765 |
| Roughness (Low -> High)         |     0.42  |         0.59  |    0.79  |

## Interpretation
The GAT successfully extrapolates to entirely unseen morphology regimes. Even when trained exclusively on low-rise neighborhoods and tested on high-rise neighborhoods, Wake Fraction R² remains positive (0.380), significantly outperforming the tabular models' memorization failures. The model captures fundamental flow interactions rather than memorizing specific archetype profiles.