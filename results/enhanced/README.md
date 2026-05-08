# Enhanced MARL-XGBoost Round

This directory contains the comparison output for the enhanced decentralized anomaly detection round.

## Command

```bash
'/Users/simonparris/Decentralized Federated Detection/threatmesh/.venv/bin/python' \
  Code/enhanced_marl_xgb.py \
  --datasets unsw ton cic \
  --agents 5 \
  --alpha 0.5 \
  --features 24 \
  --max-rows 8000
```

## Enhancements Tested

- Stable train-only preprocessing with `OrdinalEncoder` and `StandardScaler`.
- Dirichlet non-IID client partitioning.
- Class-imbalance-aware XGBoost agents using `scale_pos_weight`.
- Mutual-information feature policy plus epsilon-random feature policy competition.
- Trust-weighted decentralized probability aggregation.
- Global calibration split for threshold selection.
- Adaptive fail-closed collaboration: use the trust ensemble only when it beats the best validated peer.

## Result Summary

| Dataset | Best legacy F1 | Enhanced F1 | Delta F1 | Outcome |
| --- | ---: | ---: | ---: | --- |
| UNSW-NB15 | 0.893971 | 0.889195 | -0.004776 | Slightly worse |
| TON IoT | 1.000000 | 0.996678 | -0.003322 | Slightly worse; baseline saturated |
| CICIDS2017 | 0.971381 | 0.990373 | +0.018991 | Better |

## Interpretation

The enhanced algorithm improved CICIDS2017 substantially on this sampled run, but it did not universally improve all datasets. UNSW-NB15 appears sensitive to feature-policy choice and peer heterogeneity, while TON IoT is already saturated by the legacy baseline under this sample. The next useful iteration should focus on repeated seeds, larger sample sizes, and per-dataset policy selection rather than claiming a universal improvement from one configuration.
