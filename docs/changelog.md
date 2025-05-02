# TFT Model Development Changelog

## Current Status

Our project is focused on developing Temporal Fusion Transformer (TFT) models for cryptocurrency price prediction using high-frequency market data. We're currently working with 10-second bar data to capture short-term price movements.

## Version History

### v0.1.0 (Current) - Initial TFT Model with Bias Correction
- **Model**: `tft_10sec_bias_corrected_20250501_230054.pt`
- **Architecture**: TFT with 21 time-varying features, 128 hidden dimensions, 4 attention heads
- **Data**: 10-second bars with OHLCV, order book, and derived features
- **Performance**:
  - MAE: ~0.31
  - RMSE: ~0.32
  - Consistent negative bias observed (~-0.31)
  - Initial trading simulation shows potential but needs optimization

### Future Improvements
- Investigate and correct the persistent negative bias in predictions
- Optimize trading strategy parameters and improve simulation
- Experiment with expanded feature set including market regime indicators
- Test alternative model architectures (larger hidden dimensions, more attention heads)
- Implement robust cross-validation methodology
- Conduct ablation studies to identify most important features 