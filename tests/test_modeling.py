from __future__ import annotations

import numpy as np
import pandas as pd


def _training_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    fraud = np.zeros(rows, dtype=int)
    fraud[::20] = 1
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{f"V{index}": rng.normal(size=rows) for index in range(1, 29)},
            "Amount": rng.lognormal(mean=3.0, sigma=1.0, size=rows),
            "Class": fraud,
        }
    )


def test_baseline_training_returns_leakage_safe_holdout_metrics() -> None:
    from fraudshield.modeling import train_baseline

    result = train_baseline(_training_frame(), test_fraction=0.25)

    assert result.train_rows == 180
    assert result.test_rows == 60
    assert set(result.metrics) == {
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    }
    assert all(0.0 <= value <= 1.0 for value in result.metrics.values())
    assert result.model.classes_.tolist() == [0, 1]


def test_baseline_exposes_chronological_holdout_probabilities_without_target_leakage() -> None:
    from fraudshield.modeling import train_baseline

    frame = _training_frame()
    result = train_baseline(frame, test_fraction=0.25)

    assert len(result.holdout_probabilities) == result.test_rows
    assert len(result.holdout_labels) == result.test_rows
    assert all(0.0 <= probability <= 1.0 for probability in result.holdout_probabilities)
    assert result.holdout_labels == tuple(frame.sort_values("Time").tail(60)["Class"].astype(int))
