#!/usr/bin/env python3
"""
find_bad_s1rtc_files.py
--------------------------------
Quick utility to traverse the Sentinel-1 RTC dataset and pinpoint the file(s)
that crash dataloading with the infamous
``RuntimeError: Trying to resize storage that is not resizable``.

Usage
-----
python find_bad_s1rtc_files.py \
    --cfg path/to/your/s1rtc_config.yaml \
    [--batch_size 4] [--num_workers 0]

The script deliberately uses PyTorch's *default* collate function (no custom
work-arounds) so that any problematic sample will surface immediately.  When a
faulty batch is detected it falls back to checking each of its samples one by
one and reports the offending grid-cell / product ID.

Nothing is written to disk; you only get console output.
"""

import argparse
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ddm.utils import construct_class_by_name
from train_vae import load_conf


def parse_args():
    p = argparse.ArgumentParser(description="Locate problematic S1RTC files")
    p.add_argument("--cfg", required=True, help="YAML config that defines the S1RTC dataset")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for probing")
    p.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 keeps debugging simple)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_conf(args.cfg)

    # Force preprocessing off – we only care about I/O + tensor creation speed.
    data_cfg = cfg["data"].copy()
    # data_cfg.setdefault("preprocess_bands", False)

    print("\nInitialising dataset …")
    dataset = construct_class_by_name(**data_cfg)
    print(f"Dataset length: {len(dataset)} samples")

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        # **NO** custom collate_fn on purpose → we want to expose the error.
    )

    print("\nIterating through DataLoader …")
    batch_idx = -1  # ensures variable exists if the first batch already fails
    try:
        for batch_idx, batch in enumerate(dl):
            # Simply iterating is enough – if default_collate fails we'll jump to except.
            if (batch_idx + 1) % 500 == 0:
                print(f"✔ Processed {batch_idx + 1} batches without issues …")
            # Nothing else to do – just fetch the next batch.

        # If we make it here the whole dataset was traversed without an issue.
        print("\n✅ Reached end of loader – no issues detected.")
        return

    # -------- error handling -------
    # Any RuntimeError raised inside the DataLoader loop will be caught below.
    # pylint: disable=bare-except
    except Exception as err:  # noqa: E722
        print("\n🛑 Error detected while collating a batch!")
        traceback.print_exc()

        # Identify which sample(s) inside the failing batch are responsible.
        start_idx = max(batch_idx, 0) * args.batch_size  # batch_idx could be -1
        end_idx = min(start_idx + args.batch_size, len(dataset))
        print(f"\n→ Narrowing down offending sample between indices {start_idx} and {end_idx - 1} …")

        for idx in range(start_idx, end_idx):
            try:
                sample = dataset[idx]
                # mimic default_collate trigger on a single sample by stacking it with itself
                torch.stack([sample["image"], sample["image"]])
            except Exception as sample_err:
                print("\n❌ Problematic sample detected!")
                print(f"   dataset index : {idx}")
                print(f"   grid_cell     : {sample['filename']}")
                print("   error message :", sample_err)
                # Optionally break after first hit.
                break
        else:
            print("Could not isolate a single failing sample – the issue might be intermittent.")


if __name__ == "__main__":
    main() 