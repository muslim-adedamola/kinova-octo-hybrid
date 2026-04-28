"""
TFDS/RLDS builder for Kinova teleoperation episodes.

Expected raw layout:
  <data_dir>/downloads/manual/kinova_dataset_2/episodes/<episode_id>/
      episode.csv
      rgb/<image files referenced by episode.csv>

The builder converts each teleoperation episode into RLDS-style steps containing:
  - observation.image: resized RGB image,
  - observation.state: measured + commanded tool pose,
  - action: 8-D vector [dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, gripper, terminate],
  - reward/discount/is_first/is_last/is_terminal metadata.

It also creates a reproducible episode-level train/val split using split_seed.
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from dataclasses import dataclass

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

_DESCRIPTION = "Kinova teleop episodes converted to RLDS format (TFDS)."
_HOMEPAGE = "https://github.com/kpertsch/rlds_dataset_builder"


def _safe_float(row, key, default=float("nan")):
    """Parse a float from a CSV row while tolerating missing/bad values."""
    v = row.get(key, "")
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(row, key, default=0):
    """Parse an int from a CSV row while tolerating missing/bad values."""
    v = row.get(key, "")
    try:
        return int(float(v))
    except Exception:
        return default


def _read_csv_rows_gfile(csv_path):
    """Read episode.csv through tf.io.gfile so local/GCS paths both work."""
    import csv
    import tensorflow as tf

    with tf.io.gfile.GFile(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def _load_and_resize_image_rgb_gfile(path, image_size=(256, 256)):
    """Load an RGB frame, validate it, and resize to the TFDS image size."""
    import io
    import numpy as np
    import tensorflow as tf
    from PIL import Image, UnidentifiedImageError

    with tf.io.gfile.GFile(path, "rb") as f:
        image_bytes = f.read()

    if not image_bytes:
        raise ValueError(f"Empty image file: {path}")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        img = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Unidentified/corrupt image file: {path}") from e
    except Exception as e:
        raise ValueError(f"Failed reading image file {path}: {e}") from e

    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)



def _parse_episode_dir(ep_dir: str, angles_in_degrees: bool, image_size=(256, 256)):
    """Parse one raw episode folder into a TFDS/RLDS sample.

    Returns None when required files are missing or all frames are invalid, so
    Beam can filter bad episodes without crashing the whole build.
    """
    import os
    import numpy as np
    import tensorflow as tf

    csv_path = os.path.join(ep_dir, "episode.csv")
    rgb_dir = os.path.join(ep_dir, "rgb")

    if not tf.io.gfile.exists(csv_path):
        return None

    rows = _read_csv_rows_gfile(csv_path)
    if len(rows) == 0:
        return None

    steps = []
    lang_instr = ""
    lang_emb = np.zeros((512,), dtype=np.float32)
    deg2rad = np.pi / 180.0

    for row in rows:
        img_file = (row.get("img_file") or "").strip()
        if not img_file:
            continue

        img_path = os.path.join(rgb_dir, img_file)
        if not tf.io.gfile.exists(img_path):
            print(f"[WARN] Missing image file: {img_path}")
            continue

        try:
            image = _load_and_resize_image_rgb_gfile(img_path, image_size=image_size)
        except Exception as e:
            print(f"[WARN] Skipping bad image: {img_path} | {e}")
            continue

        state_keys = [
            "tool_pose_x", "tool_pose_y", "tool_pose_z",
            "tool_pose_theta_x", "tool_pose_theta_y", "tool_pose_theta_z",
            "cmd_tool_pose_x", "cmd_tool_pose_y", "cmd_tool_pose_z",
            "cmd_tool_pose_theta_x", "cmd_tool_pose_theta_y", "cmd_tool_pose_theta_z",
        ]
        state = np.array([_safe_float(row, k) for k in state_keys], dtype=np.float32)

        if angles_in_degrees:
            state[3:6] *= deg2rad
            state[9:12] *= deg2rad

        dx = _safe_float(row, "a_dx", 0.0)
        dy = _safe_float(row, "a_dy", 0.0)
        dz = _safe_float(row, "a_dz", 0.0)

        dtx = _safe_float(row, "a_dtheta_x", 0.0)
        dty = _safe_float(row, "a_dtheta_y", 0.0)
        dtz = _safe_float(row, "a_dtheta_z", 0.0)

        if angles_in_degrees:
            dtx *= deg2rad
            dty *= deg2rad
            dtz *= deg2rad

        gripper = np.float32(_safe_int(row, "gripper_state_bin", 0))
        terminate = np.float32(0.0)

        action = np.array(
            [dx, dy, dz, dtx, dty, dtz, gripper, terminate],
            dtype=np.float32,
        )

        steps.append({
            "observation": {
                "image": image,
                "state": state,
            },
            "action": action,
            "discount": np.float32(1.0),
            "reward": np.float32(0.0),
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
            "language_instruction": lang_instr,
            "language_embedding": lang_emb,
        })

    if not steps:
        return None

    for j, st in enumerate(steps):
        st["is_first"] = (j == 0)
        st["is_last"] = (j == len(steps) - 1)
        st["is_terminal"] = (j == len(steps) - 1)

    steps[-1]["action"][-1] = np.float32(1.0)
    steps[-1]["reward"] = np.float32(1.0)

    key = os.path.basename(ep_dir.rstrip("/"))
    sample = {
        "steps": steps,
        "episode_metadata": {
            "episode_dir": ep_dir,
        },
    }
    return key, sample


@dataclass
class KinovaConfig(tfds.core.BuilderConfig):
    """Builder config controlling angular units and the train/val split."""
    angles_in_degrees: bool = True
    val_fraction: float = 0.10
    split_seed: int = 42


class KinovaDataset(tfds.core.BeamBasedBuilder):
    VERSION = tfds.core.Version("0.1.5")

    BUILDER_CONFIGS = [
        KinovaConfig(
            name="default",
            description="Assumes a_dtheta_* are in degrees; converts to radians.",
            angles_in_degrees=True,
            val_fraction=0.10,
            split_seed=42,
        ),
        KinovaConfig(
            name="radians",
            description="Assumes a_dtheta_* are already in radians; no conversion.",
            angles_in_degrees=False,
            val_fraction=0.10,
            split_seed=42,
        ),
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Place your raw Kinova episodes under:
      <data_dir>/downloads/manual/kinova_dataset_2/episodes/<episode_id>/
    """

    IMAGE_SIZE = (256, 256)

    def _info(self) -> tfds.core.DatasetInfo:
        state_dim = 12
        h, w = self.IMAGE_SIZE
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(h, w, 3),
                            dtype=np.uint8,
                            encoding_format="png",
                        ),
                        "state": tfds.features.Tensor(shape=(state_dim,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                    "language_embedding": tfds.features.Tensor(shape=(512,), dtype=np.float32),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_dir": tfds.features.Text(),
                }),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Discover manual episodes and create deterministic train/val splits."""
        manual_dir = dl_manager.manual_dir
        root_dir = os.path.join(manual_dir, "kinova_dataset_2", "episodes")

        if not tf.io.gfile.exists(root_dir):
            raise FileNotFoundError(f"Could not find '{root_dir}'")

        episode_dirs = sorted(
            os.path.join(root_dir, p)
            for p in tf.io.gfile.listdir(root_dir)
            if tf.io.gfile.isdir(os.path.join(root_dir, p))
        )

        if len(episode_dirs) < 2:
            raise ValueError(
                f"Need at least 2 episodes to create train/val split, found {len(episode_dirs)}"
            )

        # Episode-level split: 90/10
        val_fraction = getattr(self.builder_config, "val_fraction", 0.10)
        split_seed = getattr(self.builder_config, "split_seed", 42)

        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(episode_dirs))
        shuffled_episode_dirs = [episode_dirs[i] for i in perm]

        val_count = max(1, int(round(val_fraction * len(shuffled_episode_dirs))))
        train_count = len(shuffled_episode_dirs) - val_count

        train_episode_dirs = shuffled_episode_dirs[:train_count]
        val_episode_dirs = shuffled_episode_dirs[train_count:]

        #save train and val episodes data to txt
        train_list_path = os.path.join(manual_dir, "train_episodes.txt")
        val_list_path = os.path.join(manual_dir, "val_episodes.txt")

        with tf.io.gfile.GFile(train_list_path, "w") as f:
            for p in train_episode_dirs:
                f.write(os.path.basename(p.rstrip("/")) + "\n")

        with tf.io.gfile.GFile(val_list_path, "w") as f:
            for p in val_episode_dirs:
                f.write(os.path.basename(p.rstrip("/")) + "\n")

        print(
            f"[INFO] RLDS split seed={split_seed} | "
            f"{len(train_episode_dirs)} train episodes, {len(val_episode_dirs)} val episodes"
        )

        return {
            "train": self._build_pcollection(
                pipeline=beam.Create(train_episode_dirs),
                angles_in_degrees=getattr(self.builder_config, "angles_in_degrees", True),
            ),
            "val": self._build_pcollection(
                pipeline=beam.Create(val_episode_dirs),
                angles_in_degrees=getattr(self.builder_config, "angles_in_degrees", True),
            ),
        }

    def _build_pcollection(self, pipeline, angles_in_degrees: bool):
        """Beam transform that parses and filters episode directories."""
        return (
            pipeline
            | "ParseEpisodes" >> beam.Map(
                lambda ep_dir: _parse_episode_dir(
                    ep_dir,
                    angles_in_degrees=angles_in_degrees,
                    image_size=self.IMAGE_SIZE,
                )
            )
            | "FilterInvalid" >> beam.Filter(lambda x: x is not None)
        )

# tfds build kinova_dataset  --overwrite  --data_dir /link_to_your/tensorflow_datasets  --beam_pipeline_options="direct_running_mode=multi_threading,direct_num_workers=10"
