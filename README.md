# Leakage-Aware SCADA Cyberattack Evaluation

Run the full project from this folder:

```bash
python3 -m src.run_all
```

The command loads the fifteen local CSV files, audits the dataset, trains Random Forest, XGBoost, and a feature-space 1D-CNN under random-split and leave-one-run-out protocols, writes figures/tables, and compiles the IEEE paper.

Main outputs:

- `outputs/results.json`
- `outputs/dataset_audit.json`
- `outputs/per_fold_metrics.csv`
- `outputs/predictions_naive.csv`
- `outputs/predictions_groupkfold.csv`
- `outputs/feature_importance.csv`
- `figures/`
- `paper_project/main.pdf`
