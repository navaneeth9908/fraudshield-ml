"""Dataset acquisition, integrity checks, and validation contracts."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

FEATURE_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{index}" for index in range(1, 29)),
    "Amount",
)
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)

KAGGLE_DATASET_SLUG = "mlg-ulb/creditcardfraud"
KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
DEFAULT_ARCHIVE_NAME = "creditcardfraud.zip"
DEFAULT_CSV_NAME = "creditcard.csv"
DEFAULT_ARCHIVE_SHA256 = "a0360ce715992212e9ac72d8ccdca97f4be87dc1fdf2bed011358f7ab409a28a"
DEFAULT_CSV_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

DownloadFile = Callable[[str, Path], None]


class DatasetIntegrityError(ValueError):
    """Raised when downloaded or local dataset files fail integrity checks."""


class DatasetSchemaError(DatasetIntegrityError):
    """Raised when transaction data violates the modeling schema."""


@dataclass(frozen=True)
class PreparedDataset:
    """Verified Kaggle dataset snapshot prepared for local model development."""

    csv_path: Path
    rows: int
    fraud_cases: int
    fraud_rate: float
    csv_sha256: str
    archive_path: Path | None = None
    archive_sha256: str | None = None


@dataclass(frozen=True)
class DatasetSource:
    """Download and checksum contract for a Kaggle dataset snapshot."""

    slug: str
    download_url: str
    archive_name: str
    archive_sha256: str
    csv_name: str
    csv_sha256: str


DEFAULT_DATASET_SOURCE = DatasetSource(
    slug=KAGGLE_DATASET_SLUG,
    download_url=KAGGLE_DOWNLOAD_URL,
    archive_name=DEFAULT_ARCHIVE_NAME,
    archive_sha256=DEFAULT_ARCHIVE_SHA256,
    csv_name=DEFAULT_CSV_NAME,
    csv_sha256=DEFAULT_CSV_SHA256,
)


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate transaction schema and return the input frame."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetSchemaError(f"Missing required columns: {', '.join(missing)}")

    if len(frame) == 0:
        raise DatasetSchemaError("Dataset is empty; expected at least one transaction row")

    if frame[TARGET_COLUMN].isna().any():
        raise DatasetSchemaError("Class target contains null values")

    feature_nulls = [column for column in FEATURE_COLUMNS if frame[column].isna().any()]
    if feature_nulls:
        raise DatasetSchemaError(
            f"Model feature columns {', '.join(feature_nulls)} contain null values"
        )

    nonnumeric_features = [
        column for column in FEATURE_COLUMNS if not is_numeric_dtype(frame[column])
    ]
    if nonnumeric_features:
        raise DatasetSchemaError(
            f"Model feature columns {', '.join(nonnumeric_features)} must be numeric"
        )

    feature_values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    finite_mask = np.isfinite(feature_values)
    if not finite_mask.all():
        bad_column_indexes = np.flatnonzero(~finite_mask.all(axis=0))
        bad_columns = [FEATURE_COLUMNS[index] for index in bad_column_indexes]
        raise DatasetSchemaError(
            f"Model feature columns {', '.join(bad_columns)} must contain finite numeric values"
        )

    if not is_numeric_dtype(frame[TARGET_COLUMN]):
        raise DatasetSchemaError("Class must be a numeric binary target containing only 0 and 1")

    target_values = set(frame[TARGET_COLUMN].unique().tolist())
    if not target_values.issubset({0, 1}):
        raise DatasetSchemaError("Class must be a binary target containing only 0 and 1")

    return frame


def file_sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_file(
    dataset_path: Path, *, expected_csv_sha256: str | None = None
) -> PreparedDataset:
    """Verify a CSV checksum when provided, validate schema, and summarize the dataset."""
    csv_sha256 = _verify_sha256(dataset_path, expected_csv_sha256)
    frame = pd.read_csv(dataset_path)
    validate_transactions(frame)
    fraud_cases = int(frame[TARGET_COLUMN].sum())
    rows = len(frame)
    return PreparedDataset(
        csv_path=dataset_path,
        rows=rows,
        fraud_cases=fraud_cases,
        fraud_rate=fraud_cases / rows * 100.0,
        csv_sha256=csv_sha256,
    )


def prepare_kaggle_dataset(
    raw_dir: Path,
    *,
    archive_path: Path | None = None,
    expected_archive_sha256: str | None = DEFAULT_ARCHIVE_SHA256,
    expected_csv_sha256: str | None = DEFAULT_CSV_SHA256,
    source: DatasetSource = DEFAULT_DATASET_SOURCE,
    download_file: DownloadFile | None = None,
    force_download: bool = False,
) -> PreparedDataset:
    """Download or reuse, checksum, extract, and validate the Kaggle credit-card dataset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_to_use = archive_path or raw_dir / source.archive_name
    if archive_path is None and (force_download or not archive_to_use.exists()):
        downloader = download_file or _download_file
        downloader(source.download_url, archive_to_use)

    archive_sha256 = _verify_sha256(
        archive_to_use, expected_archive_sha256 or source.archive_sha256
    )
    csv_path = raw_dir / source.csv_name
    _extract_member(archive_to_use, member_name=source.csv_name, destination_dir=raw_dir)
    report = verify_dataset_file(
        csv_path,
        expected_csv_sha256=expected_csv_sha256 or source.csv_sha256,
    )
    return PreparedDataset(
        csv_path=report.csv_path,
        rows=report.rows,
        fraud_cases=report.fraud_cases,
        fraud_rate=report.fraud_rate,
        csv_sha256=report.csv_sha256,
        archive_path=archive_to_use,
        archive_sha256=archive_sha256,
    )


def download_creditcardfraud_dataset(
    raw_dir: Path,
    *,
    source: DatasetSource = DEFAULT_DATASET_SOURCE,
    download_file: DownloadFile | None = None,
    force_download: bool = False,
) -> PreparedDataset:
    """Download, extract, checksum, and validate the configured Kaggle dataset snapshot."""
    return prepare_kaggle_dataset(
        raw_dir,
        source=source,
        download_file=download_file,
        force_download=force_download,
    )


def _verify_sha256(path: Path, expected_sha256: str | None) -> str:
    if not path.is_file():
        raise DatasetIntegrityError(f"Dataset file does not exist: {path}")
    actual_sha256 = file_sha256(path)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise DatasetIntegrityError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256.lower()}, "
            f"got {actual_sha256.lower()}"
        )
    return actual_sha256


def _extract_member(archive_path: Path, *, member_name: str, destination_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        if member_name not in archive.namelist():
            raise DatasetIntegrityError(f"Archive {archive_path.name} is missing {member_name}")
        archive.extract(member_name, destination_dir)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120.0) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
