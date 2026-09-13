"""
Build semantic sliding-window datasets for the KAVACH forecasting pipeline.

Input:
    data/processed/filtered/states_filtered.parquet
    data/processed/filtered/labels_filtered.parquet

Output:
    data/processed/windows/semantic/
        observation_windows.npy
        medium_context_windows.npy
        short_context_windows.npy
        target_features.npy
        target_stage_ids.npy
        rollout_stage_ids.npy
        rollout_valid_mask.npy
        target_timestamps.npy
        metadata.json

Window semantics:
    Short window  = latest 5 minutes
    Medium window = latest 10 minutes
    Long window   = latest 30 minutes

Important:
    The 10-minute window is medium historical context.
    It is not itself the decision-consistency verification layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "filtered"
    / "states_filtered.parquet"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "filtered"
    / "labels_filtered.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows"
    / "semantic"
)


# ---------------------------------------------------------------------
# Window configuration
# ---------------------------------------------------------------------

SHORT_WINDOW_LENGTH = 5
MEDIUM_WINDOW_LENGTH = 10
LONG_WINDOW_LENGTH = 30

# Number of future stages to save for rollout evaluation.
ROLLOUT_HORIZON = 10


# ---------------------------------------------------------------------
# Expected feature columns
# ---------------------------------------------------------------------

EXPECTED_FEATURE_COLUMNS = [
    "flow_count",
    "bytes_sent",
    "bytes_received",
    "unique_destinations",
    "unique_destination_ports",
    "new_peers",
    "syn_ratio",
    "rst_ratio",
    "mean_packet_size",
    "iat_mean",
    "iat_variance",
    "fan_out",
    "active_hosts",
    "total_flows",
    "total_bytes",
    "new_peer_rate",
    "network_fanout",
]


# ---------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------

STAGE_TO_ID = {
    "Benign": 0,
    "Cover up": 1,
    "Data Exfiltration": 2,
    "Establish Foothold": 3,
    "Lateral Movement": 4,
    "Reconnaissance": 5,
    "Unknown": 6,
}

ID_TO_STAGE = {
    str(stage_id): stage_name
    for stage_name, stage_id in STAGE_TO_ID.items()
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def find_timestamp_column(
    states: pd.DataFrame,
    labels: pd.DataFrame,
) -> Optional[str]:
    """
    Find a timestamp column if one exists.

    The current filtered files do not appear to contain a timestamp
    column. In that case, the row order is treated as the existing
    chronological order and row positions are used as timestamps.
    """

    candidates = [
        "timestamp",
        "time",
        "ts",
        "datetime",
        "date",
        "event_time",
        "window_end",
        "target_timestamp",
    ]

    for column in candidates:
        if column in states.columns:
            return column

    for column in candidates:
        if column in labels.columns:
            return column

    return None


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the filtered state and label files."""

    if not STATES_PATH.exists():
        raise FileNotFoundError(
            f"States file was not found:\n{STATES_PATH}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file was not found:\n{LABELS_PATH}"
        )

    states = pd.read_parquet(STATES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    print(f"Loaded states: {states.shape}")
    print(f"Loaded labels: {labels.shape}")

    if len(states) != len(labels):
        raise ValueError(
            "States and labels are not row-aligned:\n"
            f"  states rows = {len(states)}\n"
            f"  labels rows = {len(labels)}"
        )

    return states, labels


def prepare_temporal_order(
    states: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[str]]:
    """
    Sort by timestamp when available.

    If no timestamp column exists, preserve the current row order.
    """

    timestamp_column = find_timestamp_column(states, labels)

    if timestamp_column is not None:
        print(f"Using timestamp column: {timestamp_column}")

        if timestamp_column in states.columns:
            states = states.sort_values(
                timestamp_column
            ).reset_index(drop=True)

            labels = labels.loc[
                states.index
            ].reset_index(drop=True)

            timestamps = states[timestamp_column].to_numpy()

        else:
            labels = labels.sort_values(
                timestamp_column
            ).reset_index(drop=True)

            states = states.loc[
                labels.index
            ].reset_index(drop=True)

            timestamps = labels[timestamp_column].to_numpy()

    else:
        print(
            "No timestamp column found. "
            "Preserving existing chronological row order."
        )

        states = states.reset_index(drop=True)
        labels = labels.reset_index(drop=True)

        # Stable temporal identifiers based on row position.
        timestamps = np.arange(len(states), dtype=np.int64)

    return states, labels, timestamps, timestamp_column


def select_feature_columns(states: pd.DataFrame) -> list[str]:
    """
    Select the 17 behavioral features.

    Metadata columns such as forecast_ready and history_length are
    intentionally excluded because they are not behavioral inputs.
    """

    available_expected = [
        column
        for column in EXPECTED_FEATURE_COLUMNS
        if column in states.columns
    ]

    missing_expected = [
        column
        for column in EXPECTED_FEATURE_COLUMNS
        if column not in states.columns
    ]

    if missing_expected:
        raise ValueError(
            "The following expected feature columns are missing:\n"
            + "\n".join(f"  - {column}" for column in missing_expected)
        )

    if len(available_expected) != len(EXPECTED_FEATURE_COLUMNS):
        raise ValueError(
            "The number of selected features is not 17. "
            f"Found {len(available_expected)}."
        )

    print(f"Selected feature count: {len(available_expected)}")
    print("Selected features:")
    for index, column in enumerate(available_expected, start=1):
        print(f"  {index:02d}. {column}")

    return available_expected


def convert_features_to_float(
    states: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Convert selected state features into a numeric matrix."""

    feature_frame = states[feature_columns].copy()

    for column in feature_columns:
        feature_frame[column] = pd.to_numeric(
            feature_frame[column],
            errors="coerce",
        )

    if feature_frame.isna().any().any():
        nan_counts = feature_frame.isna().sum()
        problematic = {
            column: int(count)
            for column, count in nan_counts.items()
            if count > 0
        }

        raise ValueError(
            "Non-numeric or missing values were found in feature columns:\n"
            f"{problematic}"
        )

    features = feature_frame.to_numpy(dtype=np.float32)

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature matrix contains NaN, positive infinity, "
            "or negative infinity values."
        )

    return features


def find_stage_column(labels: pd.DataFrame) -> str:
    """Find the categorical attack-stage column."""

    for candidate in ["stage", "label", "attack_stage", "class"]:
        if candidate in labels.columns:
            return candidate

    raise ValueError(
        "Could not find an attack-stage column in labels. "
        f"Available columns: {list(labels.columns)}"
    )


def encode_stage_labels(
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Convert stage labels into integer IDs.

    Unknown or missing stage values are mapped to the Unknown class.
    """

    stage_column = find_stage_column(labels)

    raw_stages = (
        labels[stage_column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    unknown_values = sorted(
        value
        for value in raw_stages.unique()
        if value not in STAGE_TO_ID
    )

    if unknown_values:
        print(
            "The following stage values are not in the known mapping "
            "and will be mapped to Unknown:"
        )

        for value in unknown_values:
            print(f"  - {value}")

    stage_ids = np.array(
        [
            STAGE_TO_ID.get(stage, STAGE_TO_ID["Unknown"])
            for stage in raw_stages
        ],
        dtype=np.int64,
    )

    return raw_stages.to_numpy(dtype=object), stage_ids, stage_column


def build_window(
    features: np.ndarray,
    target_index: int,
    window_length: int,
) -> np.ndarray:
    """
    Extract one right-aligned historical window.

    Example:
        target_index = 100
        window_length = 5

    Uses rows:
        96, 97, 98, 99, 100
    """

    start_index = target_index - window_length + 1
    end_index = target_index + 1

    if start_index < 0:
        raise IndexError(
            f"Not enough history for target index {target_index} "
            f"and window length {window_length}."
        )

    return features[start_index:end_index]


def build_semantic_windows(
    features: np.ndarray,
    stage_ids: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build all semantic windows.

    A target is valid when at least 30 historical rows are available.

    Future rollout labels are padded with -1 when fewer than the complete
    future horizon is available. rollout_valid_mask identifies valid entries.
    """

    number_of_rows, feature_count = features.shape

    first_target_index = LONG_WINDOW_LENGTH - 1

    if number_of_rows <= first_target_index:
        raise ValueError(
            "Not enough rows to construct a 30-minute historical window. "
            f"Rows available: {number_of_rows}"
        )

    target_indices = np.arange(
        first_target_index,
        number_of_rows,
        dtype=np.int64,
    )

    sample_count = len(target_indices)

    observation_windows = np.empty(
        (
            sample_count,
            LONG_WINDOW_LENGTH,
            feature_count,
        ),
        dtype=np.float32,
    )

    medium_context_windows = np.empty(
        (
            sample_count,
            MEDIUM_WINDOW_LENGTH,
            feature_count,
        ),
        dtype=np.float32,
    )

    short_context_windows = np.empty(
        (
            sample_count,
            SHORT_WINDOW_LENGTH,
            feature_count,
        ),
        dtype=np.float32,
    )

    target_features = np.empty(
        (
            sample_count,
            feature_count,
        ),
        dtype=np.float32,
    )

    target_stage_ids = np.empty(
        sample_count,
        dtype=np.int64,
    )

    rollout_stage_ids = np.full(
        (
            sample_count,
            ROLLOUT_HORIZON,
        ),
        fill_value=-1,
        dtype=np.int64,
    )

    rollout_valid_mask = np.zeros(
        (
            sample_count,
            ROLLOUT_HORIZON,
        ),
        dtype=bool,
    )

    target_timestamps = np.empty(
        sample_count,
        dtype=timestamps.dtype,
    )

    for sample_index, target_index in enumerate(target_indices):
        observation_windows[sample_index] = build_window(
            features=features,
            target_index=target_index,
            window_length=LONG_WINDOW_LENGTH,
        )

        medium_context_windows[sample_index] = build_window(
            features=features,
            target_index=target_index,
            window_length=MEDIUM_WINDOW_LENGTH,
        )

        short_context_windows[sample_index] = build_window(
            features=features,
            target_index=target_index,
            window_length=SHORT_WINDOW_LENGTH,
        )

        target_features[sample_index] = features[target_index]
        target_stage_ids[sample_index] = stage_ids[target_index]
        target_timestamps[sample_index] = timestamps[target_index]

        # Future rollout starts after the current target row.
        future_start = target_index + 1
        future_end = min(
            target_index + 1 + ROLLOUT_HORIZON,
            number_of_rows,
        )

        available_future = stage_ids[future_start:future_end]
        available_length = len(available_future)

        if available_length > 0:
            rollout_stage_ids[
                sample_index,
                :available_length,
            ] = available_future

            rollout_valid_mask[
                sample_index,
                :available_length,
            ] = True

    return {
        "observation_windows": observation_windows,
        "medium_context_windows": medium_context_windows,
        "short_context_windows": short_context_windows,
        "target_features": target_features,
        "target_stage_ids": target_stage_ids,
        "rollout_stage_ids": rollout_stage_ids,
        "rollout_valid_mask": rollout_valid_mask,
        "target_timestamps": target_timestamps,
        "target_indices": target_indices,
    }


def save_outputs(
    arrays: dict[str, np.ndarray],
    feature_columns: list[str],
    stage_column: str,
    timestamp_column: Optional[str],
    number_of_rows: int,
) -> None:
    """Save arrays and metadata to the semantic windows directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUTPUT_DIR / "observation_windows.npy",
        arrays["observation_windows"],
    )

    np.save(
        OUTPUT_DIR / "medium_context_windows.npy",
        arrays["medium_context_windows"],
    )

    np.save(
        OUTPUT_DIR / "short_context_windows.npy",
        arrays["short_context_windows"],
    )

    np.save(
        OUTPUT_DIR / "target_features.npy",
        arrays["target_features"],
    )

    np.save(
        OUTPUT_DIR / "target_stage_ids.npy",
        arrays["target_stage_ids"],
    )

    np.save(
        OUTPUT_DIR / "rollout_stage_ids.npy",
        arrays["rollout_stage_ids"],
    )

    np.save(
        OUTPUT_DIR / "rollout_valid_mask.npy",
        arrays["rollout_valid_mask"],
    )

    np.save(
        OUTPUT_DIR / "target_timestamps.npy",
        arrays["target_timestamps"],
    )

    metadata = {
        "source_states": str(STATES_PATH),
        "source_labels": str(LABELS_PATH),
        "output_directory": str(OUTPUT_DIR),
        "timestamp_column": timestamp_column,
        "timestamp_mode": (
            "column"
            if timestamp_column is not None
            else "row_position_preserving_existing_order"
        ),
        "stage_column": stage_column,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "number_of_input_rows": number_of_rows,
        "number_of_samples": int(
            len(arrays["target_stage_ids"])
        ),
        "window_lengths": {
            "short": SHORT_WINDOW_LENGTH,
            "medium": MEDIUM_WINDOW_LENGTH,
            "long": LONG_WINDOW_LENGTH,
        },
        "window_alignment": "right_aligned_at_target_index",
        "rollout_horizon": ROLLOUT_HORIZON,
        "rollout_start": "one_row_after_current_target",
        "rollout_padding_value": -1,
        "stage_to_id": STAGE_TO_ID,
        "id_to_stage": ID_TO_STAGE,
        "array_shapes": {
            key: list(value.shape)
            for key, value in arrays.items()
            if isinstance(value, np.ndarray)
        },
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved outputs:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path.name}")


def validate_outputs(arrays: dict[str, np.ndarray]) -> None:
    """Run basic structural validation checks."""

    observation_windows = arrays["observation_windows"]
    medium_context_windows = arrays["medium_context_windows"]
    short_context_windows = arrays["short_context_windows"]
    target_features = arrays["target_features"]
    target_stage_ids = arrays["target_stage_ids"]
    rollout_stage_ids = arrays["rollout_stage_ids"]
    rollout_valid_mask = arrays["rollout_valid_mask"]

    sample_count = len(target_stage_ids)

    assert observation_windows.shape[0] == sample_count
    assert medium_context_windows.shape[0] == sample_count
    assert short_context_windows.shape[0] == sample_count
    assert target_features.shape[0] == sample_count
    assert rollout_stage_ids.shape[0] == sample_count
    assert rollout_valid_mask.shape[0] == sample_count

    assert observation_windows.shape[1] == LONG_WINDOW_LENGTH
    assert medium_context_windows.shape[1] == MEDIUM_WINDOW_LENGTH
    assert short_context_windows.shape[1] == SHORT_WINDOW_LENGTH

    assert observation_windows.shape[2] == len(
        EXPECTED_FEATURE_COLUMNS
    )

    assert medium_context_windows.shape[2] == len(
        EXPECTED_FEATURE_COLUMNS
    )

    assert short_context_windows.shape[2] == len(
        EXPECTED_FEATURE_COLUMNS
    )

    assert target_features.shape[1] == len(
        EXPECTED_FEATURE_COLUMNS
    )

    assert np.isfinite(observation_windows).all()
    assert np.isfinite(medium_context_windows).all()
    assert np.isfinite(short_context_windows).all()
    assert np.isfinite(target_features).all()

    assert np.isin(
        target_stage_ids,
        list(STAGE_TO_ID.values()),
    ).all()

    valid_rollout_values = rollout_stage_ids[rollout_valid_mask]

    if len(valid_rollout_values) > 0:
        assert np.isin(
            valid_rollout_values,
            list(STAGE_TO_ID.values()),
        ).all()

    print("\nValidation passed.")
    print(f"  Samples: {sample_count}")
    print(
        "  Observation windows:",
        observation_windows.shape,
    )
    print(
        "  Medium context windows:",
        medium_context_windows.shape,
    )
    print(
        "  Short context windows:",
        short_context_windows.shape,
    )
    print(
        "  Target features:",
        target_features.shape,
    )
    print(
        "  Rollout targets:",
        rollout_stage_ids.shape,
    )

    complete_rollout_count = int(
        rollout_valid_mask.all(axis=1).sum()
    )

    partial_rollout_count = int(
        (
            rollout_valid_mask.any(axis=1)
            & ~rollout_valid_mask.all(axis=1)
        ).sum()
    )

    print(
        f"  Complete {ROLLOUT_HORIZON}-step rollouts: "
        f"{complete_rollout_count}"
    )

    print(
        f"  Partial rollouts near dataset end: "
        f"{partial_rollout_count}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("KAVACH Semantic Window Builder")
    print("=" * 72)

    states, labels = load_data()

    states, labels, timestamps, timestamp_column = (
        prepare_temporal_order(
            states,
            labels,
        )
    )

    feature_columns = select_feature_columns(states)

    features = convert_features_to_float(
        states=states,
        feature_columns=feature_columns,
    )

    raw_stage_labels, stage_ids, stage_column = (
        encode_stage_labels(labels)
    )

    if len(features) != len(stage_ids):
        raise ValueError(
            "Feature and label lengths differ after preprocessing:\n"
            f"  features = {len(features)}\n"
            f"  labels = {len(stage_ids)}"
        )

    print(f"\nStage column: {stage_column}")
    print("Stage distribution:")

    unique_ids, counts = np.unique(
        stage_ids,
        return_counts=True,
    )

    for stage_id, count in zip(unique_ids, counts):
        stage_name = ID_TO_STAGE[str(int(stage_id))]
        print(
            f"  {int(stage_id)} - {stage_name}: {int(count)}"
        )

    arrays = build_semantic_windows(
        features=features,
        stage_ids=stage_ids,
        timestamps=timestamps,
    )

    validate_outputs(arrays)

    save_outputs(
        arrays=arrays,
        feature_columns=feature_columns,
        stage_column=stage_column,
        timestamp_column=timestamp_column,
        number_of_rows=len(states),
    )

    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()