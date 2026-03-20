# Stage 2 Evaluation Outputs

Expected generated artifacts:

- `reports/stage2_dr/history.json`
- `reports/stage2_dr/val_metrics.json`
- `reports/stage2_dr/test_metrics.json`
- `reports/stage2_hr/history.json`
- `reports/stage2_hr/val_metrics.json`
- `reports/stage2_hr/test_metrics.json`
- `reports/figures/stage2_dr/*.png`
- `reports/figures/stage2_hr/*.png`

The training scripts write:

- accuracy and loss curves
- DR and HR confusion matrices
- prediction example grids

Stage 1 confusion matrix plotting is supported through `src/eval/plots.py` once integrated predictions are available.
