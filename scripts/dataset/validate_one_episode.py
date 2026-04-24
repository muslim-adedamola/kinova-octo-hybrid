"""Validate a single episode from kinova_dataset.

Usage:
  python3 validate_one_episode.py --data_dir /path/to/tensorflow_datasets --config default

Notes:
  - data_dir should be the same directory you used when running `tfds build`.
  - config can be "default" (deg->rad) or "radians".
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow_datasets as tfds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="TFDS data_dir used for tfds build / tfds.load")
    parser.add_argument("--config", default="default", choices=["default", "radians"])
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ds = tfds.load(
        f"kinova_dataset/{args.config}",
        split=args.split,
        data_dir=args.data_dir,
        shuffle_files=False,
    )

    first = next(iter(ds))
    steps = list(tfds.as_numpy(first["steps"]))

    print("Episode metadata:")
    print("  episode_dir:", first["episode_metadata"]["episode_dir"].numpy().decode("utf-8"))
    print("  num_steps:", len(steps))

    # Flags sanity
    flags = [(i, s["is_first"], s["is_last"], s["is_terminal"]) for i, s in enumerate(steps)]
    print("\nFirst/Last/Terminal flags (index, is_first, is_last, is_terminal):")
    print("  first step:", flags[0])
    print("  last step:", flags[-1])

    # Action stats
    actions = np.stack([s["action"] for s in steps], axis=0)  # (T, 8)
    mins = actions.min(axis=0)
    maxs = actions.max(axis=0)
    print("\nAction ranges per dimension:")
    names = ["dx","dy","dz","droll","dpitch","dyaw","gripper","terminate"]
    for n, mn, mx in zip(names, mins, maxs):
        print(f"  {n:9s} min={mn:+.6f}  max={mx:+.6f}")

    # Image decode check
    img0 = steps[0]["observation"]["image"]
    print("\nImage decode check:")
    print("  dtype:", img0.dtype, "shape:", img0.shape, "min:", img0.min(), "max:", img0.max())
    assert img0.ndim == 3 and img0.shape[-1] == 3, "Expected HxWx3 image"
    assert img0.dtype == np.uint8, "Expected uint8 image"

    # Terminate flag sanity
    term = actions[:, 7]
    print("\nTerminate values (unique):", np.unique(term))
    print("Done.")


if __name__ == "__main__":
    main()
