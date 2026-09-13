from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"


STAGE_NAMES = {
    0: "Benign",
    1: "Cover up",
    2: "Data Exfiltration",
    3: "Establish Foothold",
    4: "Lateral Movement",
    5: "Reconnaissance",
    6: "Unknown",
}


def load_split(split):
    current_path = SPLIT_DIR / f"target_stage_ids_{split}.npy"
    rollout_path = SPLIT_DIR / f"rollout_stage_ids_{split}.npy"

    current = np.load(current_path)
    rollout = np.load(rollout_path)

    # First rollout step = next stage
    next_stage = rollout[:, 0]

    # Keep only valid rows
    valid = (
        (current >= 0)
        & (next_stage >= 0)
    )

    return current[valid], next_stage[valid]


def print_distribution(title, counter, total):
    print(f"\n{title}")
    print("-" * len(title))

    for stage_id, count in sorted(counter.items()):
        name = STAGE_NAMES.get(stage_id, f"Unknown ID {stage_id}")
        percentage = 100 * count / total if total else 0
        print(
            f"{stage_id}: {name:<22} "
            f"{count:>8} ({percentage:6.2f}%)"
        )


def analyze_split(split):
    current, next_stage = load_split(split)

    same = current == next_stage
    transitions = ~same

    print("\n" + "=" * 70)
    print(f"SPLIT: {split.upper()}")
    print("=" * 70)

    print(f"Total valid samples: {len(current)}")
    print(f"Same-stage next step: {same.sum()} ({100 * same.mean():.2f}%)")
    print(
        f"Actual transition:   {transitions.sum()} "
        f"({100 * transitions.mean():.2f}%)"
    )

    print_distribution(
        "Current-stage distribution",
        Counter(current.tolist()),
        len(current),
    )

    print_distribution(
        "Next-stage distribution",
        Counter(next_stage.tolist()),
        len(next_stage),
    )

    transition_counts = Counter(
        zip(current[transitions].tolist(), next_stage[transitions].tolist())
    )

    print("\nTransition pairs")
    print("----------------")

    for (source, destination), count in transition_counts.most_common():
        source_name = STAGE_NAMES.get(source, str(source))
        destination_name = STAGE_NAMES.get(destination, str(destination))

        print(
            f"{source_name:<22} -> "
            f"{destination_name:<22} : {count:>8}"
        )

    print("\nTransition matrix")
    print("-----------------")

    matrix = defaultdict(Counter)

    for source, destination in zip(current.tolist(), next_stage.tolist()):
        matrix[source][destination] += 1

    stage_ids = sorted(STAGE_NAMES.keys())

    header = "Current \\ Next".ljust(24)
    header += "".join(f"{STAGE_NAMES[s][:10]:>12}" for s in stage_ids)
    print(header)

    for source in stage_ids:
        row = STAGE_NAMES[source].ljust(24)
        row += "".join(
            f"{matrix[source][destination]:>12}"
            for destination in stage_ids
        )
        print(row)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        analyze_split(split)