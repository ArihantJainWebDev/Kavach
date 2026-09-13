from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


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
    current = np.load(
        SPLIT_DIR / f"target_stage_ids_{split}.npy"
    )
    rollout = np.load(
        SPLIT_DIR / f"rollout_stage_ids_{split}.npy"
    )

    next_stage = rollout[:, 0]

    valid = (
        (current >= 0)
        & (next_stage >= 0)
    )

    return current[valid], next_stage[valid]


def build_transition_table(current, next_stage):
    table = defaultdict(Counter)

    for source, destination in zip(current, next_stage):
        table[int(source)][int(destination)] += 1

    return table


def most_likely_destination(table, source):
    counts = table.get(int(source), Counter())

    if not counts:
        return int(source)

    return counts.most_common(1)[0][0]


def evaluate(y_true, y_pred, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.4f}"
    )
    print(
        f"Macro F1:          "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(STAGE_NAMES.keys()),
            target_names=list(STAGE_NAMES.values()),
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(STAGE_NAMES.keys()),
        )
    )


if __name__ == "__main__":
    train_current, train_next = load_split("train")
    test_current, test_next = load_split("test")

    transition_table = build_transition_table(
        train_current,
        train_next,
    )

    print("Training transition rules:")
    print("-" * 70)

    for source in sorted(STAGE_NAMES):
        source_name = STAGE_NAMES[source]
        counts = transition_table.get(source, Counter())

        if not counts:
            print(f"{source_name}: no training examples")
            continue

        print(f"\n{source_name}")
        total = sum(counts.values())

        for destination, count in counts.most_common():
            probability = count / total
            destination_name = STAGE_NAMES.get(
                destination,
                f"Unknown ID {destination}",
            )

            print(
                f"  -> {destination_name:<22} "
                f"{count:>6} "
                f"({probability:.4f})"
            )

    predictions = np.array([
        most_likely_destination(transition_table, source)
        for source in test_current
    ])

    evaluate(
        test_next,
        predictions,
        "Most-likely transition baseline",
    )

    persistence_predictions = test_current.copy()

    evaluate(
        test_next,
        persistence_predictions,
        "Persistence baseline",
    )