from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATES_PATH = PROJECT_ROOT / "data" / "processed" / "states" / "states.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "states" / "labels.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "filtered"
FILTERED_STATES_PATH = OUTPUT_DIR / "states_filtered.parquet"
FILTERED_LABELS_PATH = OUTPUT_DIR / "labels_filtered.parquet"
REPORT_PATH = OUTPUT_DIR / "filtering_report.json"


STAGE_COLUMNS = [
    "stage",
    "label",
    "attack_stage",
    "next_stage",
    "is_attack",
]

NON_NEGATIVE_COLUMNS = [
    "flow_count",
    "bytes_sent",
    "bytes_received",
    "unique_destinations",
    "unique_destination_ports",
    "new_peers",
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

RATIO_COLUMNS = [
    "syn_ratio",
    "rst_ratio",
]

BEHAVIOUR_COLUMNS = [
    "flow_count",
    "bytes_sent",
    "bytes_received",
    "unique_destinations",
    "unique_destination_ports",
    "new_peers",
    "fan_out",
    "active_hosts",
    "new_peer_rate",
    "network_fanout",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    states = pd.read_parquet(STATES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    if not isinstance(states.index, pd.DatetimeIndex):
        states.index = pd.to_datetime(states.index, errors="coerce")

    if not isinstance(labels.index, pd.DatetimeIndex):
        labels.index = pd.to_datetime(labels.index, errors="coerce")

    states = states.sort_index()
    labels = labels.sort_index()

    common_index = states.index.intersection(labels.index)

    states = states.loc[common_index].copy()
    labels = labels.loc[common_index].copy()

    return states, labels


def layer_1_data_quality(states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Layer 1:
    - Validate timestamps.
    - Convert all model features to numeric.
    - Remove NaN and infinite values.
    - Remove impossible negative values.
    - Remove invalid ratios outside [0, 1].
    """

    result = states.copy()

    initial_rows = len(result)

    valid_timestamp = ~result.index.isna()

    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    finite_mask = np.isfinite(result.to_numpy()).all(axis=1)

    non_negative_mask = np.ones(len(result), dtype=bool)

    for column in NON_NEGATIVE_COLUMNS:
        if column in result.columns:
            non_negative_mask &= result[column].to_numpy() >= 0

    ratio_mask = np.ones(len(result), dtype=bool)

    for column in RATIO_COLUMNS:
        if column in result.columns:
            values = result[column].to_numpy()
            ratio_mask &= (values >= 0) & (values <= 1)

    keep_mask = (
        valid_timestamp
        & finite_mask
        & non_negative_mask
        & ratio_mask
    )

    result = result.loc[keep_mask].copy()

    report = {
        "initial_rows": int(initial_rows),
        "removed_rows": int(initial_rows - len(result)),
        "remaining_rows": int(len(result)),
        "removed_invalid_timestamps": int((~valid_timestamp).sum()),
        "removed_non_finite_rows": int((~finite_mask).sum()),
        "removed_negative_value_rows": int((~non_negative_mask).sum()),
        "removed_invalid_ratio_rows": int((~ratio_mask).sum()),
    }

    return result, report


def layer_2_behavioural_relevance(
    states: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Layer 2:
    Retain:
    - All known attack-labelled windows.
    - Windows containing meaningful network activity.
    - Windows showing significant behavioural change.
    - Context windows around attack-stage transitions.

    Empty windows with no activity and no attack evidence are removed.
    """

    states = states.copy()
    labels = labels.loc[states.index].copy()

    initial_rows = len(states)

    stage_series = None

    for candidate in ["stage", "label", "attack_stage"]:
        if candidate in labels.columns:
            stage_series = labels[candidate].astype(str)
            break

    if stage_series is None:
        stage_series = pd.Series(
            "Unknown",
            index=states.index,
            dtype="object",
        )

    known_attack_mask = (
        ~stage_series.str.lower().isin(
            {"benign", "unknown", "nan", "none"}
        )
    )

    activity_columns = [
        column
        for column in [
            "flow_count",
            "bytes_sent",
            "bytes_received",
            "unique_destinations",
            "unique_destination_ports",
            "new_peers",
            "active_hosts",
            "total_flows",
            "total_bytes",
        ]
        if column in states.columns
    ]

    activity_mask = states[activity_columns].fillna(0).sum(axis=1) > 0

    # Compute changes against a short rolling behavioural baseline.
    behaviour = states[
        [column for column in BEHAVIOUR_COLUMNS if column in states.columns]
    ].copy()

    log_behaviour = np.log1p(behaviour.clip(lower=0))

    rolling_baseline = (
        log_behaviour
        .rolling(window=5, min_periods=1, center=True)
        .median()
    )

    relative_change = (log_behaviour - rolling_baseline).abs()

    behaviour_change_mask = (
        relative_change.max(axis=1) >= 0.50
    )

    # Preserve context around attack transitions.
    transition_mask = known_attack_mask.astype(int).diff().abs().fillna(0) > 0

    transition_context_mask = (
        transition_mask
        .rolling(window=5, min_periods=1, center=True)
        .max()
        .astype(bool)
    )

    keep_mask = (
        known_attack_mask
        | activity_mask
        | behaviour_change_mask
        | transition_context_mask
    )

    filtered_states = states.loc[keep_mask].copy()
    filtered_labels = labels.loc[keep_mask].copy()

    report = {
        "initial_rows": int(initial_rows),
        "remaining_rows": int(len(filtered_states)),
        "removed_rows": int(initial_rows - len(filtered_states)),
        "known_attack_rows": int(known_attack_mask.sum()),
        "active_rows": int(activity_mask.sum()),
        "behaviour_change_rows": int(behaviour_change_mask.sum()),
        "transition_context_rows": int(transition_context_mask.sum()),
        "retention_rate": round(
            float(len(filtered_states) / max(initial_rows, 1)),
            6,
        ),
    }

    return filtered_states, filtered_labels, report


def layer_3_forecast_readiness(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Layer 3 foundation:
    - Mark rows with sufficient historical context.
    - Mark rows that are safe for model forecasting.
    - Do not use model predictions here.
    - Do not delete the first lookback-1 rows; they remain useful for analysis.
    """

    states = states.copy()
    labels = labels.loc[states.index].copy()

    states["forecast_ready"] = False
    states["history_length"] = np.arange(1, len(states) + 1)

    states.loc[
        states["history_length"] >= lookback,
        "forecast_ready",
    ] = True

    finite_features = np.isfinite(
        states.drop(
            columns=["forecast_ready", "history_length"],
            errors="ignore",
        ).to_numpy()
    ).all(axis=1)

    states["forecast_ready"] &= finite_features

    report = {
        "lookback": int(lookback),
        "total_rows": int(len(states)),
        "forecast_ready_rows": int(states["forecast_ready"].sum()),
        "warmup_rows": int((~states["forecast_ready"]).sum()),
    }

    return states, labels, report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    states, labels = load_data()

    print(f"Loaded states: {states.shape}")
    print(f"Loaded labels: {labels.shape}")

    filtered_states_1, report_1 = layer_1_data_quality(states)

    labels_after_1 = labels.loc[filtered_states_1.index].copy()

    filtered_states_2, filtered_labels_2, report_2 = (
        layer_2_behavioural_relevance(
            filtered_states_1,
            labels_after_1,
        )
    )

    filtered_states_3, filtered_labels_3, report_3 = (
        layer_3_forecast_readiness(
            filtered_states_2,
            filtered_labels_2,
            lookback=10,
        )
    )

    # Metadata columns are useful for diagnostics but should not be fed
    # directly into the model as behavioural features.
    filtered_states_3.to_parquet(FILTERED_STATES_PATH)
    filtered_labels_3.to_parquet(FILTERED_LABELS_PATH)

    report = {
        "source": {
            "states": str(STATES_PATH),
            "labels": str(LABELS_PATH),
        },
        "output": {
            "states": str(FILTERED_STATES_PATH),
            "labels": str(FILTERED_LABELS_PATH),
        },
        "input_shape": {
            "states": list(states.shape),
            "labels": list(labels.shape),
        },
        "layers": {
            "layer_1_data_quality": report_1,
            "layer_2_behavioural_relevance": report_2,
            "layer_3_forecast_readiness": report_3,
        },
        "final_shape": {
            "states": list(filtered_states_3.shape),
            "labels": list(filtered_labels_3.shape),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFiltering completed.")
    print(f"Layer 1 remaining rows: {report_1['remaining_rows']}")
    print(f"Layer 2 remaining rows: {report_2['remaining_rows']}")
    print(
        "Layer 3 forecast-ready rows: "
        f"{report_3['forecast_ready_rows']}"
    )

    print("\nSaved:")
    print(FILTERED_STATES_PATH)
    print(FILTERED_LABELS_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()