from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _valid_transactions(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": np.arange(rows, dtype=float),
            **{f"V{index}": np.full(rows, float(index)) for index in range(1, 29)},
            "Amount": np.full(rows, 12.5),
            "Class": [0, 1, *([0] * (rows - 2))],
        }
    )


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


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.iloc[0:0], "empty"),
        (lambda frame: frame.drop(columns=["Class"]), "Class"),
        (lambda frame: frame.assign(Class=[0, None, 1]), "Class.*null"),
        (lambda frame: frame.assign(V4=[0.1, None, 0.3]), "V4.*null"),
        (lambda frame: frame.assign(V7=[0.1, float("inf"), 0.3]), "V7.*finite"),
        (lambda frame: frame.assign(Amount=[12.0, "declined", 8.0]), "Amount.*numeric"),
        (lambda frame: frame.assign(Class=[0, 1, -1]), "Class.*0 and 1"),
    ],
)
def test_validation_rejects_dataset_quality_contract_violations(
    mutator: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    from fraudshield.data import DatasetSchemaError, validate_transactions

    with pytest.raises(DatasetSchemaError, match=message):
        validate_transactions(mutator(_valid_transactions()))


def test_prepare_kaggle_dataset_verifies_checksums_and_extracts_valid_csv(tmp_path: Path) -> None:
    from fraudshield.data import file_sha256, prepare_kaggle_dataset

    csv_text = _valid_transactions().to_csv(index=False)
    expected_csv_path = _write_bytes(tmp_path / "expected.csv", csv_text.encode("utf-8"))
    zip_path = tmp_path / "creditcardfraud.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("creditcard.csv", csv_text)

    raw_dir = tmp_path / "raw"
    result = prepare_kaggle_dataset(
        raw_dir=raw_dir,
        archive_path=zip_path,
        expected_archive_sha256=file_sha256(zip_path),
        expected_csv_sha256=file_sha256(expected_csv_path),
    )

    assert result.csv_path == raw_dir / "creditcard.csv"
    assert result.rows == 3
    assert result.fraud_cases == 1
    assert result.csv_sha256 == file_sha256(result.csv_path)


def test_prepare_kaggle_dataset_rejects_checksum_mismatch(tmp_path: Path) -> None:
    from fraudshield.data import DatasetIntegrityError, prepare_kaggle_dataset

    zip_path = tmp_path / "creditcardfraud.zip"
    zip_path.write_bytes(b"not the expected archive")

    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        prepare_kaggle_dataset(
            raw_dir=tmp_path / "raw",
            archive_path=zip_path,
            expected_archive_sha256="0" * 64,
            expected_csv_sha256="1" * 64,
        )


def _write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path
