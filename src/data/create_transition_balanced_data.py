from pathlib import Path
from collections import Counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "data" / "processed" / "transition_balanced"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split):
    observations = np.load(
        SPLIT_DIR / f"observation_windows_{split}.npy"
    )

    current_stage = np.load(
        SPLIT_DIR / f"target_stage_ids_{split}.npy"
    )

    rollout_stage = np.load(
        SPLIT_DIR / f"rollout_stage_ids_{split}.npy"
    )

    next_stage = rollout_stage[:, 0]

    valid = (
        (current_stage >= 0)
        & (next_stage >= 0)
    )

    return (
        observations[valid],
        current_stage[valid],
        next_stage[valid],
    )


def describe_pairs(current_stage, next_stage, title):
    pairs = Counter(
        zip(
            current_stage.tolist(),
            next_stage.tolist(),
        )
    )

    print(f"\n{title}")
    print("-" * 70)

    for (source, destination), count in pairs.most_common():
        print(
            f"{source} -> {destination}: {count}"
        )


def main():
    observations, current_stage, next_stage = load_split("train")

    transition_mask = current_stage != next_stage
    persistence_mask = ~transition_mask

    transition_indices = np.flatnonzero(transition_mask)
    persistence_indices = np.flatnonzero(persistence_mask)

    print(f"Original training samples: {len(current_stage)}")
    print(f"Transition samples:        {len(transition_indices)}")
    print(f"Persistence samples:       {len(persistence_indices)}")

    describe_pairs(
        current_stage[transition_mask],
        next_stage[transition_mask],
        "Original transition pairs",
    )

    # Keep all transition examples.
    selected_transition_indices = transition_indices

    # Keep a 3:1 ratio of persistence to transition samples.
    # This prevents the balanced set from becoming dominated by persistence.
    desired_persistence_count = min(
        len(persistence_indices),
        len(selected_transition_indices) * 3,
    )

    rng = np.random.default_rng(42)

    selected_persistence_indices = rng.choice(
        persistence_indices,
        size=desired_persistence_count,
        replace=False,
    )

    selected_indices = np.concatenate([
        selected_transition_indices,
        selected_persistence_indices,
    ])

    rng.shuffle(selected_indices)

    balanced_observations = observations[selected_indices]
    balanced_current_stage = current_stage[selected_indices]
    balanced_next_stage = next_stage[selected_indices]

    np.save(
        OUTPUT_DIR / "observation_windows_train.npy",
        balanced_observations,
    )

    np.save(
        OUTPUT_DIR / "current_stage_train.npy",
        balanced_current_stage,
    )

    np.save(
        OUTPUT_DIR / "next_stage_train.npy",
        balanced_next_stage,
    )

    transition_labels = (
        balanced_current_stage != balanced_next_stage
    ).astype(np.float32)

    np.save(
        OUTPUT_DIR / "transition_labels_train.npy",
        transition_labels,
    )

    print("\nCreated transition-balanced training data")
    print("-" * 70)
    print(f"Samples: {len(balanced_next_stage)}")
    print(
        f"Transitions: {transition_labels.sum():.0f} "
        f"({100 * transition_labels.mean():.2f}%)"
    )
    print(
        f"Persistence: {(1 - transition_labels).sum():.0f} "
        f"({100 * (1 - transition_labels.mean()):.2f}%)"
    )

    describe_pairs(
        balanced_current_stage,
        balanced_next_stage,
        "Balanced transition pairs",
    )


if __name__ == "__main__":
    main()