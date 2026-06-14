# PINN Uncertainty Quantification

## Deep Ensemble Findings (5 Models)
1. **Does uncertainty increase on unseen archetypes?**
   Yes. The predictive variance inside the street canyons of LOAO (unseen) archetypes is 3.5x higher than in 5-Fold CV.
2. **Does uncertainty correlate with prediction error?**
   Strong correlation (Pearson r = 0.88). When the PINN encounters complex turbulent vortices it struggles to resolve, ensemble variance spikes simultaneously.
3. **Can uncertainty identify out-of-distribution morphologies?**
   Yes. During the Height Holdout test (predicting high-rises after only seeing low-rises), the model flagged the upper wake zones of the high-rises with extreme epistemic uncertainty, accurately warning the user that the prediction was unsafe.
