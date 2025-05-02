# TFT Model Evaluation Metrics

## Overview

This document outlines the standard evaluation metrics used for our TFT models. Having consistent evaluation criteria ensures proper comparison between model iterations and helps track progress over time.

## Core Metrics

### Regression Performance

| Metric | Formula | Interpretation | Target |
|--------|---------|----------------|--------|
| **MAE** (Mean Absolute Error) | `mean(abs(y_pred - y_true))` | Average magnitude of errors | Lower is better |
| **RMSE** (Root Mean Squared Error) | `sqrt(mean((y_pred - y_true)^2))` | Error magnitude with higher penalty for large errors | Lower is better |
| **Correlation** | `corr(y_pred, y_true)` | Linear relationship between predictions and actuals | Higher is better (target >0.1) |
| **Prediction Bias** | `mean(y_pred - y_true)` | Systematic over/underprediction | Target: close to 0 |

### Classification Performance

| Metric | Formula | Interpretation | Target |
|--------|---------|----------------|--------|
| **Direction Accuracy** | `mean(sign(y_pred) == sign(y_true))` | Percent of correctly predicted price movements | >50% (target >55%) |
| **Precision** (Up) | `TP / (TP + FP)` | Proportion of correct positive predictions | Higher is better |
| **Recall** (Up) | `TP / (TP + FN)` | Proportion of actual positives identified | Higher is better |
| **F1 Score** | `2 * (Precision * Recall) / (Precision + Recall)` | Harmonic mean of precision and recall | Higher is better |

## Trading Performance

| Metric | Formula | Interpretation | Target |
|--------|---------|----------------|--------|
| **Win Rate** | `winning_trades / total_trades` | Proportion of profitable trades | >50% (target >55%) |
| **Profit Factor** | `gross_profit / gross_loss` | Ratio of money made to money lost | >1.0 (target >1.5) |
| **Sharpe Ratio** | `(mean_returns - risk_free_rate) / std_returns` | Risk-adjusted return | >1.0 (target >2.0) |
| **Max Drawdown** | `max(cumulative_returns.cummax() - cumulative_returns)` | Largest peak-to-trough decline | Lower is better |
| **Calmar Ratio** | `annualized_return / max_drawdown` | Return relative to maximum risk | Higher is better |

## Evaluation Procedure

### Standard Validation Process

1. **Split Testing**:
   - Train models on training data
   - Tune hyperparameters on validation set
   - Final evaluation on completely unseen test data

2. **Walk-forward Analysis**:
   - Use expanding or rolling window for time series validation
   - Maintain temporal ordering of data (no random shuffling)
   - Re-normalize features for each window

3. **Multiple Seed Testing**:
   - Run models with different random seeds
   - Report mean and standard deviation of performance metrics
   - Assess model stability and robustness

### Trading Simulation Parameters

| Parameter | Description | Default Value | Rationale |
|-----------|-------------|---------------|-----------|
| `threshold` | Minimum prediction magnitude to enter position | 0.0001 | Filter noise, reduce false signals |
| `fee` | Trading fee per transaction | 0.001 | Realistic cost assumption |
| `position_size` | Trade size relative to capital | 1.0 (100%) | Maximum exposure for testing |
| `stop_loss` | Maximum loss per trade | None (future addition) | Risk management |

## Reporting Format

### Standard Metrics Report

```
Evaluation metrics:
MAE: 0.XXXXX
RMSE: 0.XXXXX
Correlation: 0.XXXXX
Prediction bias: 0.XXXXX

Direction accuracy: XX.XX%
Precision (up): XX.XX%
Recall (up): XX.XX%

Trading performance:
Total trades: XXX
Win rate: XX.XX%
Profit factor: X.XX
Sharpe ratio: X.XX
Max drawdown: XX.XX%
```

### Visualization Requirements

1. **Time Series Plot**:
   - Actual vs. predicted values over time
   - Zoomed view of representative segments

2. **Error Distribution**:
   - Histogram of prediction errors
   - Q-Q plot to assess normality

3. **Cumulative Returns**:
   - Equity curve of trading strategy
   - Comparison to buy-and-hold baseline

4. **Feature Importance**:
   - Attention weight visualization
   - Variable importance scores

## Benchmarks

| Model | MAE | RMSE | Direction Accuracy | Win Rate | Notes |
|-------|-----|------|-------------------|----------|-------|
| Random | 0.35 | 0.40 | 50% | 50% | Statistical baseline |
| ARIMA | 0.33 | 0.38 | 52% | 51% | Linear time series model |
| Current TFT v0.1 | 0.31 | 0.32 | TBD | TBD | Initial implementation |

## Result Interpretation Guidelines

1. **Statistical Significance**:
   - Compare to random baseline with hypothesis testing
   - Use bootstrapping for confidence intervals
   - Consider market regime effects on performance

2. **Overfitting Assessment**:
   - Compare training vs. testing performance
   - Examine learning curves over epochs
   - Assess sensitivity to hyperparameter changes

3. **Failure Analysis**:
   - Identify market conditions where model performs poorly
   - Analyze largest prediction errors
   - Track regime-dependent performance 