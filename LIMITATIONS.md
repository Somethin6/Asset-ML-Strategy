# Limitations

This repository is retained as an early exploratory machine-learning application, not as evidence of a validated forecasting or trading system.

## Methodological limitations

- The original model was built around contemporaneous OHLCV-style variables and derived indicators. That setup does not by itself establish genuine future forecasting skill.
- A sequential train/test split is preferable to random shuffling for time series, but it is not a substitute for rolling-origin or walk-forward evaluation.
- The application does not establish transaction-cost, slippage, latency, market-impact, or execution realism.
- Model fit metrics such as R²/MSE do not imply economic usefulness.
- Feature importance from a Random Forest is descriptive of the fitted model, not causal evidence.
- The future-prediction path in the original code was incomplete/placeholder-stage.

## Appropriate interpretation

The project demonstrates early work with Python, pandas, scikit-learn, desktop GUI construction, data validation, plotting, and a basic ML workflow. It should not be interpreted as a profitable strategy, a production trading system, or evidence of out-of-sample alpha.

Later quantitative work should use causal feature construction, rolling/walk-forward origins, explicit baselines, cost sensitivity, leakage controls, and null testing.
