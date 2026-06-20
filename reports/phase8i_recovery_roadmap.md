# Phase 8I Recovery Roadmap

1. **Remove Node Subsampling**: Deploy to GPU and process full CFD point clouds.
2. **Increase Epochs**: Run minimum 200 epochs to permit gradient saturation.
3. **Implement Loss Weights**: Down-weight Pressure MSE by 0.2x to prevent gradient domination.
4. **Execute Full Production Pipeline**: Run Phase 9 deployment.
