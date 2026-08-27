from __future__ import annotations

import pandas as pd
import pytest


def test_validation_rejects_a_missing_model_feature() -> None:
    from fraudshield.data import DatasetSchemaError, validate_transactions

    frame = pd.DataFrame(
        {
            "Time": [0.0],
            **{f"V{index}": [0.0] for index in range(1, 28)},
            "Amount": [12.5],
            "Class": [0],
        }
    )

    with pytest.raises(DatasetSchemaError, match="V28"):
        validate_transactions(frame)


def test_validation_rejects_a_non_binary_target() -> None:
    from fraudshield.data import DatasetSchemaError, validate_transactions

    frame = pd.DataFrame(
        {
            "Time": [0.0, 1.0],
            **{f"V{index}": [0.0, 0.1] for index in range(1, 29)},
            "Amount": [12.5, 4.0],
            "Class": [0, 2],
        }
    )

    with pytest.raises(DatasetSchemaError, match="binary"):
        validate_transactions(frame)
