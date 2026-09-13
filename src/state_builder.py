from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATE_COLUMNS = [
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


# Columns actually required from the 89-column Unraveled flow schema.
REQUIRED_COLUMNS = [
    "src_ip",
    "dst_ip",
    "dst_port",
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "bidirectional_packets",
    "bidirectional_bytes",
    "src2dst_bytes",
    "dst2src_bytes",
    "src2dst_packets",
    "dst2src_packets",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_min_ps",
    "bidirectional_max_ps",
    "bidirectional_syn_packets",
    "bidirectional_rst_packets",
    "Activity",
    "Stage",
    "DefenderResponse",
    "Signature",
]


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected columns to numeric without crashing on malformed values."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

STAGE_PRIORITY = {
    "Benign": 0,
    "Reconnaissance": 1,
    "Establish Foothold": 2,
    "Lateral Movement": 3,
    "Data Exfiltration": 4,
    "Cover up": 5,
}


def _select_stage(values: pd.Series) -> str:
    """
    Select the most advanced known attack stage present in a window.

    Unknown/unmapped values are not interpreted as attack stages.
    """
    values = (
        values.dropna()
        .astype(str)
        .str.strip()
    )

    known = [
        value
        for value in values
        if value in STAGE_PRIORITY
    ]

    if not known:
        return "Unknown"

    return max(
        known,
        key=lambda value: STAGE_PRIORITY[value],
    )


def _select_activity(values: pd.Series) -> str:
    """
    Preserve an attack activity when one is present.

    Normal is treated as benign/background activity.
    Unknown and unmapped values are not interpreted.
    """
    values = (
        values.dropna()
        .astype(str)
        .str.strip()
    )

    attack_values = [
        value
        for value in values
        if value not in {"Normal", "Unknown"}
    ]

    if attack_values:
        return attack_values[0]

    if "Normal" in values.values:
        return "Normal"

    return "Unknown"


def _select_label(values: pd.Series) -> str:
    """
    Generic label aggregation for metadata where no explicit
    semantic priority has been defined.
    """
    values = (
        values.dropna()
        .astype(str)
        .str.strip()
    )

    if values.empty:
        return "Unknown"

    return values.mode().iloc[0]

def build_states_from_flows(
    df: pd.DataFrame,
    window: str = "60s",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw Unraveled network-flow rows into temporal behavioural states.

    Returns:
        states:
            One row per temporal window containing model features.

        labels:
            One row per temporal window containing evaluation metadata.
            These labels are NEVER used as model input.
    """

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required Unraveled columns: "
            + ", ".join(missing)
        )

    x = df.copy()

    # Numeric conversion.
    _numeric(
        x,
        [
            "dst_port",
            "bidirectional_first_seen_ms",
            "bidirectional_last_seen_ms",
            "bidirectional_packets",
            "bidirectional_bytes",
            "src2dst_bytes",
            "dst2src_bytes",
            "src2dst_packets",
            "dst2src_packets",
            "bidirectional_mean_ps",
            "bidirectional_stddev_ps",
            "bidirectional_min_ps",
            "bidirectional_max_ps",
            "bidirectional_syn_packets",
            "bidirectional_rst_packets",
        ],
    )

    # Timestamp comes from the flow start time.
    x["Timestamp"] = pd.to_datetime(
        x["bidirectional_first_seen_ms"],
        unit="ms",
        errors="coerce",
    )

    x = x.dropna(subset=["Timestamp"])
    x = x.sort_values("Timestamp")

    if x.empty:
        return (
            pd.DataFrame(columns=STATE_COLUMNS),
            pd.DataFrame(),
        )

    x = x.set_index("Timestamp")

    grouped = x.resample(window)

    states = pd.DataFrame(index=grouped.size().index)

    # Basic volume.
    states["flow_count"] = grouped.size()

    states["bytes_sent"] = grouped["src2dst_bytes"].sum()
    states["bytes_received"] = grouped["dst2src_bytes"].sum()

    states["total_flows"] = states["flow_count"]
    states["total_bytes"] = (
        states["bytes_sent"] + states["bytes_received"]
    )

    # Diversity / discovery behaviour.
    states["unique_destinations"] = (
        grouped["dst_ip"].nunique()
    )

    states["unique_destination_ports"] = (
        grouped["dst_port"].nunique()
    )

    # Number of distinct source hosts active in a window.
    states["active_hosts"] = grouped["src_ip"].nunique()

    # Fan-out.
    states["fan_out"] = (
        states["unique_destinations"]
        / states["active_hosts"].replace(0, np.nan)
    )

    states["network_fanout"] = (
        states["unique_destinations"]
        / states["flow_count"].replace(0, np.nan)
    )

    # TCP behaviour.
    states["syn_ratio"] = _safe_ratio(
        grouped["bidirectional_syn_packets"].sum(),
        grouped["bidirectional_packets"].sum(),
    )

    states["rst_ratio"] = _safe_ratio(
        grouped["bidirectional_rst_packets"].sum(),
        grouped["bidirectional_packets"].sum(),
    )

    # Packet-size behaviour.
    states["mean_packet_size"] = (
        grouped["bidirectional_mean_ps"].mean()
    )

    # Approximate inter-arrival behaviour.
    # Derive IAT from flow start times within each source host.
    #
    # We use a temporary row ID so duplicate timestamps cannot cause
    # pandas to reindex incorrectly.

    temp = x.reset_index()[["Timestamp", "src_ip"]].copy()
    temp["_row_id"] = np.arange(len(temp))

    temp = temp.sort_values(
        ["src_ip", "Timestamp", "_row_id"]
    )

    temp["_iat_seconds"] = (
        temp.groupby("src_ip")["Timestamp"]
        .diff()
        .dt.total_seconds()
        .clip(lower=0)
        .fillna(0.0)
    )

    # Restore the original row order before putting IAT back into x.
    temp = temp.sort_values("_row_id")

    x = x.copy()
    x["_iat_seconds"] = temp["_iat_seconds"].to_numpy()

    iat = x["_iat_seconds"]

    states["iat_mean"] = iat.resample(window).mean()
    states["iat_variance"] = (
        iat.resample(window).var().fillna(0.0)
    )

    # "New peers" means destinations that were not observed in
    # the immediately preceding temporal window.
    destination_sets = []

    for timestamp, group in grouped:
        destination_sets.append(
            (
                timestamp,
                set(group["dst_ip"].dropna().astype(str))
            )
        )

    previous_peers: set[str] = set()
    new_peer_values = []

    for timestamp, peers in destination_sets:
        new_peers = len(peers - previous_peers)
        new_peer_values.append(
            (timestamp, new_peers)
        )
        previous_peers = peers

    new_peer_series = pd.Series(
        dict(new_peer_values),
        dtype=float,
    )

    states["new_peers"] = (
        new_peer_series.reindex(states.index)
        .fillna(0)
    )

    states["new_peer_rate"] = _safe_ratio(
        states["new_peers"],
        states["flow_count"],
    )

    # Clean numerical issues.
    states = (
        states
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Ensure all required model columns exist and are ordered.
    states = states[STATE_COLUMNS]

    # ---------------------------------------------------------
    # Evaluation metadata
    # ---------------------------------------------------------

    labels = pd.DataFrame(index=states.index)

    labels["flow_count"] = states["flow_count"]

    # ---------------------------------------------------------
    # Evaluation metadata
    # ---------------------------------------------------------

    labels = pd.DataFrame(index=states.index)

    labels["flow_count"] = states["flow_count"]

    # Activity:
    # Preserve an explicit non-Normal activity if one occurs
    # in the temporal window.
    labels["activity"] = (
        grouped["Activity"]
        .agg(_select_activity)
    )

    # Stage:
    # Preserve the most advanced known attack stage occurring
    # anywhere inside the temporal window.
    labels["stage"] = (
        grouped["Stage"]
        .agg(_select_stage)
    )

    # Defender response and signature do not currently have
    # a validated semantic priority, so retain their mode.
    labels["defender_response"] = (
        grouped["DefenderResponse"]
        .agg(_select_label)
    )

    labels["signature"] = (
        grouped["Signature"]
        .agg(_select_label)
    )

    return states, labels


def discover_flow_files(dataset_root: str | Path) -> list[Path]:
    """Find every Unraveled network-flow CSV recursively."""

    root = Path(dataset_root)

    files = sorted(
        root.rglob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {root}"
        )

    return files


def build_states_from_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    window: str = "60s",
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the complete Unraveled network-flow dataset.

    Processing is performed file-by-file and chunk-by-chunk so that
    the complete dataset does not need to fit in memory.
    """

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = discover_flow_files(dataset_root)

    print(f"Found {len(files)} CSV files.")

    all_states = []
    all_labels = []

    total_rows = 0

    usecols = [
        c for c in REQUIRED_COLUMNS
    ]

    for i, csv_file in enumerate(files, start=1):

        print(
            f"[{i}/{len(files)}] "
            f"{csv_file.name}"
        )

        chunks = []

        try:
            reader = pd.read_csv(
                csv_file,
                usecols=lambda c: c in usecols,
                chunksize=chunksize,
                engine="python",
                on_bad_lines="warn",
            )

            for chunk in reader:
                total_rows += len(chunk)
                chunks.append(chunk)

        except Exception as exc:
            print(
                f"WARNING: failed to read {csv_file}: {exc}"
            )
            continue

        if not chunks:
            continue

        df = pd.concat(
            chunks,
            ignore_index=True,
        )

        states, labels = build_states_from_flows(
            df,
            window=window,
        )

        if not states.empty:
            all_states.append(states)
            all_labels.append(labels)

    if not all_states:
        raise RuntimeError(
            "No states were generated."
        )

    states = pd.concat(all_states)
    labels = pd.concat(all_labels)

    # Multiple source files can produce the same temporal index.
    # Aggregate them into a single network-wide state.
    states = (
        states
        .groupby(level=0)
        .mean()
        .sort_index()
    )

    # Keep the corresponding label metadata.
   # Keep the strongest label observed across all source files
# contributing to the same temporal network state.
    labels = (
        labels
        .groupby(level=0)
        .agg(
            {
                "flow_count": "sum",
                "activity": _select_activity,
                "stage": _select_stage,
                "defender_response": _select_label,
                "signature": _select_label,
            }
        )
        .sort_index()
    )

    # Align exactly.
    common_index = states.index.intersection(
        labels.index
    )

    states = states.loc[common_index]
    labels = labels.loc[common_index]

    states_path = output_dir / "states.parquet"
    labels_path = output_dir / "labels.parquet"

    states.to_parquet(states_path)
    labels.to_parquet(labels_path)

    print()
    print("State generation complete.")
    print(f"Raw rows processed: {total_rows:,}")
    print(f"Temporal states: {len(states):,}")
    print(f"Features: {len(STATE_COLUMNS)}")
    print(f"States: {states_path}")
    print(f"Labels: {labels_path}")

    return states, labels