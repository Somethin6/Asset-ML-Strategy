# Asset ML Strategy

> **Legacy exploratory project.** This repository is retained as an early local financial-data/ML application and is **not** presented as a validated trading or forecasting system.

Asset ML Strategy is a Python/tkinter application for loading OHLCV-style spreadsheet data, computing basic technical features, fitting a Random Forest regression model, and visualizing diagnostics.

## What this project demonstrates

- local Excel/financial-data ingestion with pandas/openpyxl
- feature construction from OHLCV-style inputs
- scikit-learn Random Forest integration
- sequential train/test splitting
- basic model diagnostics and feature importance
- matplotlib-based visualization
- a tkinter desktop interface

## Important limitations

This project predates my later causal/walk-forward market research tooling and should be read as an exploratory ML application rather than evidence of predictive market edge.

In particular:

- the current formulation is not a rigorous future-return forecasting protocol
- contemporaneous OHLCV variables can make apparent prediction performance difficult to interpret as genuine out-of-sample forecasting ability
- the repository does not establish profitability after fees, slippage, or realistic execution
- the future-prediction path is incomplete/placeholder-level
- no result here should be interpreted as investment advice or as evidence that the model generalizes to live markets

For quantitative research, a stronger evaluation standard is strict time causality, rolling/walk-forward origins, realistic baselines, transaction-cost sensitivity, null testing, and explicit leakage controls.

## Running the application

```bash
pip install -r requirements.txt
python asset_ml_strategy.py
```

Expected input columns:

```text
Date | Open | High | Low | Close | Adj Close | Volume
```

A sample workbook is included for exercising the application.

## Repository status

This repository is maintained primarily as historical evidence of an earlier stage of my Python/ML work. It is not a current flagship project.

## License

See the repository license files/metadata for applicable terms.
