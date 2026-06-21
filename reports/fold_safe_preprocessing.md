# Fold-Safe Preprocessing

PowerTransformer is instantiated and `.fit()` called exclusively on training-fold pressure values inside `fold_safe_train_eval()`. Validation fold receives a frozen `.transform()`. Verified by code inspection — no global scaler exists.

**PASS**