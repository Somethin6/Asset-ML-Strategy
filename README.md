# Asset ML Strategy

> **Legacy exploratory project.** This repository is retained as an early Python/ML application and is **not** presented as a validated forecasting or trading system. See [`LIMITATIONS.md`](LIMITATIONS.md) before interpreting model outputs.

A local desktop application for loading OHLCV-style financial data, computing simple derived features, fitting a Random Forest regression model, and visualizing basic diagnostics.

## What this project demonstrates

- Python data handling with pandas/NumPy
- scikit-learn model fitting
- sequential train/test splitting
- simple technical-feature construction
- matplotlib-based diagnostics
- tkinter desktop GUI construction
- Excel input validation and local processing

## What this project does **not** establish

- genuine out-of-sample future price forecasting skill
- profitable trading after costs/slippage
- causal relationships between features and returns
- production execution reliability
- generalization across assets or regimes

The original implementation includes an incomplete/placeholder future-prediction path, so this repository should be read as an exploratory software artifact rather than a finished quantitative-research result.

## Run locally

```bash
pip install -r requirements.txt
python asset_ml_strategy.py
```

Expected input columns:

```text
Date, Open, High, Low, Close, Adj Close, Volume
```

A sample workbook is included for basic application testing.

## Methodology note

For serious time-series research, the intended successor methodology is stricter: features must be known at the forecast origin, evaluation should use rolling/walk-forward origins, obvious baselines should be reported, and any trading interpretation should include costs, leakage controls, and null/robustness tests.

## License

See the repository license/files for reuse terms.
