# Generalization Assessment

Comparing 5-Fold CV (Interpolation within known archetypes/conditions) to Leave-One-Archetype-Out (Extrapolation to unseen geometries):

1. **Interpolation**: The models achieve high R² on 5-Fold CV, demonstrating they can accurately interpolate within the trained configuration space (varying wind speeds and directions).
2. **Extrapolation**: Under LOAO CV, the R² drops. This signifies the models struggle to extrapolate outside the training space (predicting micro-climates for completely unseen neighborhood geometries). 

**Conclusion**: The surrogates are powerful interpolators but require a much larger archetype diversity (Phase 5A expansion to 100+ archetypes) to achieve true spatial extrapolation.
