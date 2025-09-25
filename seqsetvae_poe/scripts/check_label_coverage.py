#!/usr/bin/env python3
import os
import glob
import os.path as op
import sys


def main() -> None:
    label_csv = os.environ.get("LABEL_CSV")
    data_dir = os.environ.get("DATA_DIR")
    assert label_csv and data_dir, "请先设置 LABEL_CSV 和 DATA_DIR 环境变量"

    # Use the normalized mapping logic from classifier
    sys.path.insert(0, "/workspace")
    from seqsetvae_poe.classifier import _read_label_map  # type: ignore

    id_to_label = _read_label_map(label_csv)

    def stem(p: str) -> str:
        return op.splitext(op.basename(p))[0]

    for split in ["train", "valid", "test"]:
        files = sorted(glob.glob(op.join(data_dir, split, "*.parquet")))
        ids = [stem(p) for p in files]
        labeled_ids = [pid for pid in ids if pid in id_to_label]
        missing = [pid for pid in ids if pid not in id_to_label]
        y = [id_to_label[pid] for pid in labeled_ids]
        n_pos = sum(1 for v in y if v == 1)
        n_neg = sum(1 for v in y if v == 0)
        print(
            f"{split}: total_files={len(ids)} labeled={len(labeled_ids)} missing={len(missing)} pos={n_pos} neg={n_neg}"
        )
        if len(missing) > 0:
            print("  sample_missing:", missing[:10])


if __name__ == "__main__":
    main()

