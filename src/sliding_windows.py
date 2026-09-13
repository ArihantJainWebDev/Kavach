from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

STATES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "filtered"
    / "states_filtered.parquet"
)

LABELS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "filtered"
    / "labels_filtered.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "windows"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Sliding-window configuration
# ============================================================

WINDOWS = {
    "short": 5,       # Last 5 minutes
    "medium": 10,     # Last 10 minutes
    "long": 30,       # Last 30 minutes
}

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "Date",
    "Timestamp",
]

POSSIBLE_LABEL_COLUMNS = {
    "stage",
    "label",
    "attack_stage",
    "attack_stage_label",
    "class",
}

NON_FEATURE_COLUMNS = {
    "timestamp",
    "time",
    "datetime",
    "date",
    "forecast_ready",
    "history_length",
}


# ============================================================
# Utility functions
# ============================================================

def find_timestamp_column(df: pd.DataFrame) -> str | None:
    """
    Find a timestamp column.

    Returns None if no timestamp column exists.
    This is necessary because the filtered states file may have
    stored timestamp as its Parquet index.
    """
    for column in TIMESTAMP_CANDIDATES:
        if column in df.columns:
            return column

    return None


def recover_timestamp_column(
    states: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure both states and labels have a normalized 'timestamp' column.

    Priority:
    1. Use an existing timestamp column.
    2. Recover timestamp from a DatetimeIndex.
    3. Recover states timestamps from labels by row alignment.
    """

    states = states.copy()
    labels = labels.copy()

    # --------------------------------------------------------
    # Normalize labels timestamp
    # --------------------------------------------------------

    label_timestamp_column = find_timestamp_column(labels)

    if label_timestamp_column is not None:
        if label_timestamp_column != "timestamp":
            labels = labels.rename(
                columns={
                    label_timestamp_column: "timestamp"
                }
            )

    elif isinstance(labels.index, pd.DatetimeIndex):
        labels = labels.reset_index()

        first_column = labels.columns[0]

        labels = labels.rename(
            columns={
                first_column: "timestamp"
            }
        )

    else:
        raise ValueError(
            "Could not find a timestamp column in labels. "
            f"Available columns: {list(labels.columns)}"
        )

    # --------------------------------------------------------
    # Normalize states timestamp
    # --------------------------------------------------------

    state_timestamp_column = find_timestamp_column(states)

    if state_timestamp_column is not None:
        if state_timestamp_column != "timestamp":
            states = states.rename(
                columns={
                    state_timestamp_column: "timestamp"
                }
            )

    elif isinstance(states.index, pd.DatetimeIndex):
        states = states.reset_index()

        first_column = states.columns[0]

        states = states.rename(
            columns={
                first_column: "timestamp"
            }
        )

    else:
        # The states file has no timestamp column and no
        # DatetimeIndex. Recover timestamps from labels.
        if len(states) != len(labels):
            raise ValueError(
                "States and labels have different row counts, so "
                "timestamps cannot safely be recovered by row alignment.\n"
                f"States rows: {len(states)}\n"
                f"Labels rows: {len(labels)}"
            )

        states.insert(
            0,
            "timestamp",
            labels["timestamp"].to_numpy(),
        )

        print(
            "Timestamp recovery: states timestamps were recovered "
            "from labels using row alignment."
        )

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    states["timestamp"] = pd.to_datetime(
        states["timestamp"],
        errors="coerce",
    )

    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"],
        errors="coerce",
    )

    states = states.dropna(
        subset=["timestamp"]
    ).reset_index(drop=True)

    labels = labels.dropna(
        subset=["timestamp"]
    ).reset_index(drop=True)

    return states, labels


def find_label_column(df: pd.DataFrame) -> str:
    """
    Identify the attack-stage label column.
    """

    for column in df.columns:
        normalized = column.lower().strip()

        if normalized in POSSIBLE_LABEL_COLUMNS:
            return column

    # Fallback to a string/object column.
    object_columns = df.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    ).columns.tolist()

    # Do not accidentally select timestamp.
    object_columns = [
        column
        for column in object_columns
        if column != "timestamp"
    ]

    if not object_columns:
        raise ValueError(
            "Could not identify the label column. "
            f"Available columns: {list(df.columns)}"
        )

    return object_columns[0]


def get_feature_columns(states: pd.DataFrame) -> list[str]:
    """
    Select numeric behavioural features only.

    Filtering metadata such as forecast_ready and history_length
    is excluded from the model input.
    """

    feature_columns = []

    for column in states.columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            states[column]
        ):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError(
            "No numeric feature columns were found."
        )

    return feature_columns


def build_complete_minute_grid(
    states: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Build a complete one-minute timeline.

    Missing minutes are filled with zero-valued behavioural
    features. This is important because Phase 1 removed some
    inactive or irrelevant rows. Row-based slicing alone would
    incorrectly treat those gaps as consecutive minutes.
    """

    working = states[
        ["timestamp"] + feature_columns
    ].copy()

    working = working.sort_values(
        "timestamp"
    )

    working = working.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    working = working.set_index(
        "timestamp"
    )

    start_time = working.index.min().floor("min")
    end_time = working.index.max().floor("min")

    full_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="min",
    )

    feature_grid = working[
        feature_columns
    ].reindex(full_index)

    # Missing minutes represent inactive or filtered-out periods.
    feature_grid = feature_grid.fillna(0.0)

    # Ensure all values are finite.
    feature_grid = feature_grid.replace(
        [np.inf, -np.inf],
        0.0,
    )

    feature_grid = feature_grid.astype(
        np.float32
    )

    return feature_grid, full_index


def encode_labels(
    labels: pd.Series,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Encode string labels into integer class IDs.
    """

    labels = (
        labels
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    labels = labels.replace(
        "",
        "Unknown",
    )

    class_names = sorted(
        labels.unique().tolist()
    )

    class_to_id = {
        class_name: index
        for index, class_name in enumerate(
            class_names
        )
    }

    encoded = labels.map(
        class_to_id
    ).to_numpy(
        dtype=np.int64
    )

    return encoded, class_to_id


# ============================================================
# Main sliding-window pipeline
# ============================================================

def main() -> None:
    print("Loading filtered states and labels...")

    if not STATES_PATH.exists():
        raise FileNotFoundError(
            f"States file not found:\n{STATES_PATH}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found:\n{LABELS_PATH}"
        )

    states = pd.read_parquet(
        STATES_PATH
    )

    labels = pd.read_parquet(
        LABELS_PATH
    )

    print(
        f"States shape: {states.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    # --------------------------------------------------------
    # Recover timestamps
    # --------------------------------------------------------

    states, labels = recover_timestamp_column(
        states,
        labels,
    )

    print(
        f"States after timestamp recovery: {states.shape}"
    )

    print(
        f"Labels after timestamp recovery: {labels.shape}"
    )

    # --------------------------------------------------------
    # Identify label and feature columns
    # --------------------------------------------------------

    label_column = find_label_column(
        labels
    )

    feature_columns = get_feature_columns(
        states
    )

    print(
        f"Label column: {label_column}"
    )

    print(
        f"Feature count: {len(feature_columns)}"
    )

    print(
        f"Features: {feature_columns}"
    )

    # --------------------------------------------------------
    # Sort both datasets chronologically
    # --------------------------------------------------------

    states = states.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    labels = labels.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Build complete one-minute feature timeline
    # --------------------------------------------------------

    feature_grid, full_index = (
        build_complete_minute_grid(
            states=states,
            feature_columns=feature_columns,
        )
    )

    print(
        f"Complete timeline length: {len(full_index)}"
    )

    print(
        f"Timeline start: {full_index.min()}"
    )

    print(
        f"Timeline end: {full_index.max()}"
    )

    # --------------------------------------------------------
    # Prepare valid prediction targets
    # --------------------------------------------------------

    target_table = labels[
        ["timestamp", label_column]
    ].copy()

    target_table = target_table.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    target_table = target_table.set_index(
        "timestamp"
    )

    # Only rows marked forecast_ready are used as targets.
    # If the column is absent, all timestamps are eligible.
    if "forecast_ready" in states.columns:
        valid_target_states = states[
            states["forecast_ready"].astype(bool)
        ].copy()
    else:
        valid_target_states = states.copy()

    valid_target_timestamps = (
        valid_target_states["timestamp"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    target_table = target_table.reindex(
        valid_target_timestamps
    )

    target_table = target_table.dropna(
        subset=[label_column]
    )

    if target_table.empty:
        raise RuntimeError(
            "No valid target timestamps with labels were found."
        )

    encoded_labels, class_to_id = encode_labels(
        target_table[label_column]
    )

    timestamp_to_label_id = {
        timestamp: label_id
        for timestamp, label_id in zip(
            target_table.index,
            encoded_labels,
        )
    }

    # --------------------------------------------------------
    # Prepare feature matrix
    # --------------------------------------------------------

    feature_matrix = feature_grid[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    timestamp_positions = {
        timestamp: position
        for position, timestamp in enumerate(
            full_index
        )
    }

    # --------------------------------------------------------
    # Construct windows
    # --------------------------------------------------------

    window_sequences = {
        window_name: []
        for window_name in WINDOWS
    }

    target_timestamps = []
    target_label_ids = []

    longest_window = max(
        WINDOWS.values()
    )

    print("Constructing sliding windows...")

    for target_timestamp in target_table.index:
        target_timestamp = pd.Timestamp(
            target_timestamp
        ).floor("min")

        if target_timestamp not in timestamp_positions:
            continue

        if target_timestamp not in timestamp_to_label_id:
            continue

        target_position = timestamp_positions[
            target_timestamp
        ]

        # The long window needs 30 minutes of history.
        if target_position < longest_window - 1:
            continue

        for window_name, window_size in WINDOWS.items():
            start_position = (
                target_position
                - window_size
                + 1
            )

            end_position = (
                target_position
                + 1
            )

            sequence = feature_matrix[
                start_position:end_position
            ]

            expected_shape = (
                window_size,
                len(feature_columns),
            )

            if sequence.shape != expected_shape:
                raise RuntimeError(
                    f"Invalid {window_name} window shape: "
                    f"{sequence.shape}; "
                    f"expected {expected_shape}"
                )

            window_sequences[
                window_name
            ].append(sequence)

        target_timestamps.append(
            target_timestamp.isoformat()
        )

        target_label_ids.append(
            timestamp_to_label_id[
                target_timestamp
            ]
        )

    if not target_timestamps:
        raise RuntimeError(
            "No sliding-window sequences were created. "
            "Check timestamp alignment and dataset contents."
        )

    # --------------------------------------------------------
    # Save each window
    # --------------------------------------------------------

    for window_name, sequences in window_sequences.items():
        array = np.stack(
            sequences
        ).astype(
            np.float32
        )

        output_path = (
            OUTPUT_DIR
            / f"{window_name}_window_sequences.npz"
        )

        np.savez_compressed(
            output_path,
            X=array,
        )

        print(
            f"{window_name.capitalize()} window shape: "
            f"{array.shape}"
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Save targets
    # --------------------------------------------------------

    targets_path = (
        OUTPUT_DIR
        / "targets.npz"
    )

    np.savez_compressed(
        targets_path,
        y=np.asarray(
            target_label_ids,
            dtype=np.int64,
        ),
        timestamps=np.asarray(
            target_timestamps
        ),
    )

    print(
        f"Targets shape: "
        f"{len(target_label_ids)}"
    )

    print(
        f"Saved: {targets_path}"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "source_states": str(STATES_PATH),
        "source_labels": str(LABELS_PATH),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "windows": WINDOWS,
        "target_count": len(target_timestamps),
        "label_column": label_column,
        "class_to_id": class_to_id,
        "id_to_class": {
            str(index): class_name
            for class_name, index in class_to_id.items()
        },
        "timeline_frequency": "1min",
        "missing_minute_fill": 0.0,
        "window_alignment": (
            "right-aligned-at-target-timestamp"
        ),
        "target_timestamps_are_filtered_rows": True,
        "longest_history_minutes": longest_window,
    }

    metadata_path = (
        OUTPUT_DIR
        / "window_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Saved: {metadata_path}"
    )

    print(
        "\nSliding-window construction completed successfully."
    )

    print(
        f"Total prediction targets: "
        f"{len(target_timestamps)}"
    )


if __name__ == "__main__":
    main()