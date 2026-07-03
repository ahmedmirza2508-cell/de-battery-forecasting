#!/usr/bin/env python
"""
Phase 3, step 5 — FINAL, ONE-TIME TEST evaluation for DE-LU day-ahead prices.

*** THIS IS THE SINGLE, FINAL TEST-SET EVALUATION. ***
It does NOT retrain with new hyperparameters, does NOT change any architecture,
does NOT tune anything, and MUST NOT be used to iterate on the models. Every
model and hyperparameter decision was locked on the val split. This script only:
  - reloads the frozen models:
      * persistence  = price_lag_24h (no parameters),
      * LightGBM     = retrained on train with the IDENTICAL fixed config from
                       train_baselines.py (deterministic, random_state=0),
      * quantile-LSTM = loaded from the epoch-4 best-val checkpoint
                       data/processed/lstm_quantile_best.pt (no re-training),
  - evaluates them ONCE on the held-out TEST split,
  - reports and persists the numbers to results/phase3/test_metrics.json.
No further model changes after this, per the Phase 3 plan.

Fair comparison: all models are scored on the SAME test rows — the timestamps
the LSTM can predict (a full 168h contiguous, same-split window). Scalers are
fit on TRAIN only, exactly as in training. Nothing is fit or selected on test.

Run: mamba run -n energy-ml python ml/evaluate_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lstm as base
import train_lstm_quantile as Q

TEST_JSON = base.OUT / "test_metrics.json"


def main() -> None:
    ds = base.load_dataset()
    feats = base.feature_cols(ds)
    tr = ds[ds["split"] == "train"]
    te = ds[ds["split"] == "test"]
    print(f"TEST split loaded: {len(te)} rows  [{te.index.min()} .. {te.index.max()}]")
    print("  (nominal window 2025-07-01..2026-06-30; start trimmed by the 8-day embargo)")

    # --- quantile-LSTM inputs: scalers fit on TRAIN only; build test sequences (same batching) ---
    fsc = StandardScaler().fit(tr[feats])
    tsc = StandardScaler().fit(tr[[base.TARGET]])
    te_scaled = te.copy()
    te_scaled[feats] = fsc.transform(te[feats])
    te_scaled[base.TARGET] = tsc.transform(te[[base.TARGET]])
    Xte, yte_s, end_times = base.build_sequences(te_scaled, feats, base.SEQ_LEN)

    # Common evaluation rows = timestamps the LSTM can predict (fair apples-to-apples).
    common = pd.DatetimeIndex(end_times)
    te_common = te.loc[common]
    y = te_common[base.TARGET].to_numpy()
    print(f"  scoring ALL models on the same {len(common)} LSTM-predictable test rows "
          f"[{common.min()} .. {common.max()}]")

    # (a) persistence
    p_pers = te_common["price_lag_24h"].to_numpy()

    # (b) LightGBM — retrained on TRAIN with the identical fixed config (deterministic)
    gbm = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbosity=-1,
    )
    gbm.fit(tr[feats], tr[base.TARGET])
    p_gbm = gbm.predict(te_common[feats])

    # (c) quantile-LSTM median — from the frozen epoch-4 checkpoint
    model = Q.QuantileLSTM(len(feats), len(Q.QUANTILES))
    model.load_state_dict(torch.load(Q.MODEL_PATH))
    model.eval()
    with torch.no_grad():
        pv = model(torch.from_numpy(Xte)).numpy()  # [N,3] scaled, aligned to end_times
    pv_eur = np.column_stack([tsc.inverse_transform(pv[:, i:i + 1]).ravel() for i in range(3)])
    q10, q50, q90 = pv_eur[:, 0], pv_eur[:, 1], pv_eur[:, 2]

    # safety: the LSTM's own targets must equal te_common's target (alignment check)
    assert np.allclose(tsc.inverse_transform(yte_s.reshape(-1, 1)).ravel(), y, atol=1e-3), \
        "LSTM sequence targets are not aligned with te_common — abort."

    cov = float(np.mean((y >= q10) & (y <= q90))) * 100.0
    cross = float(np.mean((q10 > q50) | (q50 > q90))) * 100.0

    res = {
        "note": "FINAL one-time TEST evaluation; no retuning or model changes after this.",
        "eval_split": "test",
        "test_rows_scored": int(len(common)),
        "test_range": [str(common.min()), str(common.max())],
        "persistence": {"mae": base.mae(y, p_pers), "rmse": base.rmse(y, p_pers)},
        "lightgbm": {"mae": base.mae(y, p_gbm), "rmse": base.rmse(y, p_gbm)},
        "quantile_lstm_median": {
            "mae": base.mae(y, q50), "rmse": base.rmse(y, q50),
            "coverage_10_90_pct": cov, "crossing_pct": cross,
        },
    }

    print("\n=== FINAL TEST METRICS (EUR/MWh, same rows for all models) ===")
    print(f"persistence:           MAE={res['persistence']['mae']:.3f}   RMSE={res['persistence']['rmse']:.3f}")
    print(f"LightGBM:              MAE={res['lightgbm']['mae']:.3f}   RMSE={res['lightgbm']['rmse']:.3f}")
    print(f"quantile-LSTM median:  MAE={res['quantile_lstm_median']['mae']:.3f}   RMSE={res['quantile_lstm_median']['rmse']:.3f}")
    print(f"quantile-LSTM 80% interval coverage={cov:.1f}%   crossing={cross:.2f}%")

    TEST_JSON.write_text(json.dumps(res, indent=2))
    print("\nsaved ->", TEST_JSON)


if __name__ == "__main__":
    main()
