"""Locations, provenance constants and integrity digests of the scientific datasets.

The pooled V1 dataset (1,505 states x 8 modes = 12,040 rows) and its held-out benchmark set
are not distributed with this open-source core. They are licensed separately. The constants
below carry their version string, row counts and SHA-256 digests so that a licensed copy can
be verified byte-for-byte before use.

When no licensed dataset is present at the packaged location, ``V1Dataset`` raises
``DatasetNotFoundError`` rather than falling back to any smaller file. Substituting the
5-state fixture in ``tests/fixtures`` would produce numbers that look like results and are
not: that fixture is for fast unit tests only, never production or evaluation data.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# Full pooled production dataset (PER debruited by packet/error pooling; grid: elevation 1 deg,
# velocity 0.5 km/s). 1,505 states, 12,040 rows. Digests cover the canonical LF bytes enforced by
# .gitattributes, never a platform-specific CRLF working-tree representation.
POOLED_DATASET_VERSION = "v1-pooled-real-1"
POOLED_DATASET_SHA256 = "bbd0c4518c74e33ee9538be1483adc35f1f56b3a25b3b7ac4c74765777c51053"
POOLED_DATASET_STATES = 1505
POOLED_DATASET_ROWS = 12040

# Held-out states (never used to fit the KNN) for benchmarking. 301 states, 2,408 rows.
POOLED_HOLDOUT_SHA256 = "81081afa6efb9041512f843b302a81c6fcbccb977a29d7ecd29a99073c78f76c"


def _packaged(name: str) -> Path:
    return Path(str(files("meseleo_core") / "data" / name))


def pooled_dataset_path() -> Path:
    """Path where a licensed copy of the full pooled production dataset is expected."""
    return _packaged("v1_pooled_dataset.csv")


def pooled_holdout_path() -> Path:
    """Path where a licensed copy of the held-out benchmark dataset is expected."""
    return _packaged("v1_pooled_holdout_test.csv")
