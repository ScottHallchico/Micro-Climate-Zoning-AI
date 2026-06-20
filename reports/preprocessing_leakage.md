# Preprocessing Leakage Audit

Audited target normalization functions (`QuantileTransformer`).

## Sklearn Fit Integrity
```python
scaler.fit(y_train)
y_train_scaled = scaler.transform(y_train)
y_val_scaled = scaler.transform(y_val)
```

**Verdict**: The `QuantileTransformer` is strictly fitted exclusively on the training folds. No forward-leaking of validation distribution quantiles occurs.
