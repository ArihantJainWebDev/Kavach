from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"

SPLITS = ["train", "val", "test"]
HORIZONS = range(1, 11)

for split in SPLITS:
    current = np.load(
        SPLIT_DIR / f"target_stage_ids_{split}.npy"
    )
    future = np.load(
        SPLIT_DIR / f"rollout_stage_ids_{split}.npy"
    )
    valid = np.load(
        SPLIT_DIR / f"rollout_valid_mask_{split}.npy"
    )

    print(f"\n{split.upper()}")
    print("-" * 72)
    print(
        f"{'Horizon':>7} "
        f"{'Samples':>8} "
        f"{'Accuracy':>10} "
        f"{'Bal.Acc':>10} "
        f"{'Macro F1':>10}"
    )

    for h in HORIZONS:
        index = h - 1

        mask = valid[:, index].astype(bool)
        y_true = future[mask, index]
        y_pred = current[mask]

        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=np.unique(y_true),
            average="macro",
            zero_division=0,
        )

        print(
            f"{h:7d} "
            f"{len(y_true):8d} "
            f"{accuracy:10.4f} "
            f"{balanced_acc:10.4f} "
            f"{macro_f1:10.4f}"
        )