import os
import json
import argparse
from collections import deque

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import flax.linen as nn
from flax.training import checkpoints

from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.data.dataset import make_single_dataset
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
tf.config.set_visible_devices([], "GPU")


# -----------------------------------------------------------------------------
# Hybrid BCE gripper head
# -----------------------------------------------------------------------------
class BinaryGripperHead(nn.Module):
    action_horizon: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, emb):
        # emb: [B, D]
        x = nn.Dense(self.hidden_dim)(emb)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        logits = nn.Dense(self.action_horizon)(x)  # [B, H]
        return logits


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_dataset_statistics(
    stats_json=None,
    tfds_data_dir=None,
    dataset_name="kinova_dataset",
    dataset_config="default",
    dataset_version="0.1.5",
):
    if stats_json is not None:
        with open(stats_json, "r") as f:
            return json.load(f)

    if tfds_data_dir is None:
        return None

    pattern = os.path.join(
        tfds_data_dir,
        dataset_name,
        dataset_config,
        dataset_version,
        "dataset_statistics_*.json",
    )
    matches = sorted(tf.io.gfile.glob(pattern))
    if not matches:
        return None

    with tf.io.gfile.GFile(matches[-1], "r") as f:
        return json.load(f)


def tree_lists_to_arrays(x):
    if isinstance(x, dict):
        return {k: tree_lists_to_arrays(v) for k, v in x.items()}
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    return x


def extract_action_stats(model, dataset_stats=None):
    candidates = []

    if getattr(model, "dataset_statistics", None) is not None:
        candidates.append(model.dataset_statistics)

    if dataset_stats is not None:
        candidates.append(dataset_stats)

    for stats in candidates:
        if not isinstance(stats, dict):
            continue
        if "action" in stats:
            return stats["action"]
        for _, value in stats.items():
            if isinstance(value, dict) and "action" in value:
                return value["action"]

    return None


def extract_readout_tensor(x):
    if hasattr(x, "shape") and len(x.shape) >= 3:
        y = x
    elif isinstance(x, dict):
        preferred_keys = [
            "readout_action",
            "action_readout",
            "readout",
            "readouts",
            "obs_primary",
            "obs",
        ]
        y = None
        for key in preferred_keys:
            if key in x:
                candidate = x[key]
                if hasattr(candidate, "tokens"):
                    y = candidate.tokens
                    break
                if hasattr(candidate, "shape") and len(candidate.shape) >= 3:
                    y = candidate
                    break
        if y is None:
            raise TypeError(f"Could not extract readout tensor from dict keys={list(x.keys())}")
    else:
        if hasattr(x, "tokens"):
            y = x.tokens
        else:
            raise TypeError(f"Could not extract readout tensor. type={type(x)}")

    if len(y.shape) == 4:
        y = y[:, :, 0, :]

    if len(y.shape) != 3:
        raise ValueError(f"Expected readout tensor with 3 dims after normalization, got shape={y.shape}")

    return y


def standardize_episode_from_rlds_np(ep):
    step_list = list(ep["steps"])
    if len(step_list) == 0:
        raise ValueError("Episode has no steps.")

    images = []
    actions = []
    dones = []

    for st in step_list:
        img = np.asarray(st["observation"]["image"], dtype=np.uint8)
        act = np.asarray(st["action"], dtype=np.float32)

        action_dim = act.shape[-1]
        if action_dim == 8:
            act_7 = act[:7].astype(np.float32)
        elif action_dim == 7:
            act_7 = act.astype(np.float32)
        else:
            raise ValueError(f"Unexpected action dimension: {action_dim}")

        if "is_terminal" in st:
            done = bool(st["is_terminal"])
        elif "is_last" in st:
            done = bool(st["is_last"])
        else:
            if action_dim == 8:
                done = bool(act[7] > 0.5)
            else:
                done = False

        images.append(img)
        actions.append(act_7)
        dones.append(done)

    image = np.stack(images, axis=0)
    action_7 = np.stack(actions, axis=0)
    done = np.asarray(dones, dtype=bool)

    final_image = image[-1]
    goal_image = np.repeat(final_image[None, ...], repeats=image.shape[0], axis=0)

    return {
        "image_primary": image,
        "goal_image": goal_image,
        "action": action_7,
        "done": done,
    }


def build_goal_only_task_from_episode(goal_rgb):
    goal_rgb = cv2.resize(goal_rgb, (256, 256), interpolation=cv2.INTER_AREA).astype(np.uint8)
    return {
        "image_primary": goal_rgb[None],
        "pad_mask_dict": {
            "image_primary": np.ones((1,), dtype=bool),
        },
    }


def build_obs_from_window(image_window, window_size):
    frames = [
        cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.uint8)
        for img in image_window
    ]
    return {
        "image_primary": np.stack(frames, axis=0)[None],  # [1, W, H, W, C]
        "timestep": np.arange(window_size, dtype=np.int32)[None],
        "task_completed": np.zeros((1, window_size, 4), dtype=bool),
        "timestep_pad_mask": np.ones((1, window_size), dtype=bool),
        "pad_mask_dict": {
            "image_primary": np.ones((1, window_size), dtype=bool),
            "timestep": np.ones((1, window_size), dtype=bool),
        },
    }


def mse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def compute_binary_metrics(pred_prob, gt_raw, threshold=0.5, gt_open_threshold=0.5):
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt_raw = np.asarray(gt_raw, dtype=np.float32)

    pred_bin = (pred_prob >= threshold).astype(np.int32)
    gt_bin = (gt_raw >= gt_open_threshold).astype(np.int32)

    acc = float(np.mean(pred_bin == gt_bin))

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-8))

    pred_open_rate = float(np.mean(pred_bin))
    gt_open_rate = float(np.mean(gt_bin))

    pred_transitions = int(np.sum(pred_bin[1:] != pred_bin[:-1]))
    gt_transitions = int(np.sum(gt_bin[1:] != gt_bin[:-1]))
    transition_acc = (
        float(np.mean((pred_bin[1:] != pred_bin[:-1]) == (gt_bin[1:] != gt_bin[:-1])))
        if len(pred_bin) > 1 else 1.0
    )

    return {
        "grip_acc": acc,
        "grip_open_precision": precision,
        "grip_open_recall": recall,
        "grip_open_f1": f1,
        "pred_open_rate": pred_open_rate,
        "gt_open_rate": gt_open_rate,
        "pred_transitions": pred_transitions,
        "gt_transitions": gt_transitions,
        "transition_acc": transition_acc,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--tfds_data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="kinova_dataset")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_version", type=str, default="0.1.5")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--stats_json", type=str, default=None)

    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--gripper_head_hidden_dim", type=int, default=256)
    parser.add_argument("--gripper_eval_threshold", type=float, default=0.5)
    parser.add_argument("--gripper_label_open_threshold", type=float, default=0.5)

    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./offline_eval_results_hybrid")
    parser.add_argument("--csv_name", type=str, default="per_episode_metrics.csv")
    parser.add_argument("--summary_name", type=str, default="summary.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("[INFO] Loading hybrid checkpoint...")
    ckpt = checkpoints.restore_checkpoint(args.checkpoint_path, target=None)
    params = ckpt["params"]
    ckpt_step = ckpt.get("step", None)

    print("[INFO] Loading dataset statistics...")
    dataset_stats = load_dataset_statistics(
        stats_json=args.stats_json,
        tfds_data_dir=args.tfds_data_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_version=args.dataset_version,
    )

    print("[INFO] Building example batch for model init...")
    dataset_kwargs = dict(
        name=f"{args.dataset_name}/{args.dataset_config}",
        data_dir=args.tfds_data_dir,
        standardize_fn=ModuleSpec.create(kinova_rlds_to_octo),
        image_obs_keys={
            "primary": "image_primary",
            "goal": "goal_image",
        },
        action_normalization_mask=[True, True, True, True, True, True, False],
        dataset_statistics=dataset_stats,
    )

    traj_transform_kwargs = dict(
        window_size=args.window_size,
        action_horizon=args.action_horizon,
        subsample_length=None,
    )

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),
            "goal": (256, 256),
        },
        image_dropout_prob=0.0,
    )

    ds_for_example = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        train=True,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
    ).flatten().batch(1)

    ex = next(ds_for_example.as_numpy_iterator())
    obs = dict(ex["observation"])
    goal_img = obs.pop("image_goal")[:, 0]
    obs_pad = dict(obs.get("pad_mask_dict", {}))
    goal_pad = obs_pad.pop("image_goal")[:, 0] if "image_goal" in obs_pad else np.ones((1,), dtype=bool)
    obs["pad_mask_dict"] = obs_pad

    example_batch = {
        "observation": obs,
        "task": {
            "image_primary": goal_img,
            "pad_mask_dict": {"image_primary": goal_pad},
        },
        "action": ex["action"],
        "action_pad_mask": ex["action_pad_mask"],
    }

    print("[INFO] Loading Octo structure...")
    pretrained = OctoModel.load_pretrained(args.pretrained_path)
    model = OctoModel.from_config(
        pretrained.config,
        example_batch,
        text_processor=None,
        verbose=False,
        dataset_statistics=dataset_stats,
    )
    model = model.replace(params=params["octo"])

    action_stats = extract_action_stats(model, dataset_stats)
    if action_stats is None:
        print("[WARN] No action statistics found. Continuous predictions may remain normalized.")
    else:
        action_stats = tree_lists_to_arrays(action_stats)

    gripper_head = BinaryGripperHead(
        action_horizon=args.action_horizon,
        hidden_dim=args.gripper_head_hidden_dim,
    )

    @jax.jit
    def predict_step(octo_params, gripper_params, obs, task, rng):
        bound = model.module.bind({"params": octo_params}, rngs={"dropout": rng})

        emb = bound.octo_transformer(
            obs,
            task,
            obs["timestep_pad_mask"],
            train=False,
        )

        cont_actions = model.sample_actions(
            obs,
            task,
            unnormalization_statistics=action_stats,
            rng=rng,
        )[..., :6]  # [B, H, 6] or [B, W, H, 6] depending on Octo version

        cont_actions = jnp.asarray(cont_actions)
        if cont_actions.ndim == 4:
            cont_actions = cont_actions[:, -1, :, :]  # [B, H, 6]
        elif cont_actions.ndim != 3:
            raise ValueError(f"Unexpected continuous action shape: {cont_actions.shape}")

        readout_emb = extract_readout_tensor(emb)      # [B, W, D]
        readout_emb = readout_emb[:, -1, :]            # [B, D]

        gripper_logits = gripper_head.apply(
            {"params": gripper_params},
            readout_emb,
        )                                              # [B, H]

        gripper_prob = jax.nn.sigmoid(gripper_logits)[..., None]  # [B, H, 1]
        actions = jnp.concatenate([cont_actions, gripper_prob], axis=-1)  # [B, H, 7]
        return actions

    print("[INFO] Building TFDS dataset...")
    builder_name = f"{args.dataset_name}/{args.dataset_config}:{args.dataset_version}"
    ds = tfds.load(
        builder_name,
        split=args.split,
        data_dir=args.tfds_data_dir,
        shuffle_files=False,
    )

    rng = jax.random.PRNGKey(args.seed)
    rows = []

    print(f"[INFO] Evaluating split='{args.split}' ...")

    for ep_idx, ep in enumerate(tfds.as_numpy(ds)):
        if args.max_episodes is not None and ep_idx >= args.max_episodes:
            break

        ep_std = standardize_episode_from_rlds_np(ep)
        images = ep_std["image_primary"]
        goal_images = ep_std["goal_image"]
        actions_gt = ep_std["action"]
        done = ep_std["done"]

        T = images.shape[0]
        if T == 0:
            continue

        goal_rgb = goal_images[0]
        task = build_goal_only_task_from_episode(goal_rgb)

        image_window = deque(maxlen=args.window_size)
        pred_actions = []
        gt_actions = []

        for t in range(T):
            image_window.append(images[t])

            while len(image_window) < args.window_size:
                image_window.appendleft(image_window[0])

            obs = build_obs_from_window(image_window, args.window_size)

            obs_jax = jax.tree_util.tree_map(jnp.asarray, obs)
            task_jax = jax.tree_util.tree_map(jnp.asarray, task)

            rng, sample_rng = jax.random.split(rng)
            pred = predict_step(
                params["octo"],
                params["gripper_head"],
                obs_jax,
                task_jax,
                sample_rng,
            )

            pred = np.asarray(pred)
            pred_7 = pred[0, 0].astype(np.float32)   # first action of current predicted chunk
            gt_7 = actions_gt[t].astype(np.float32)

            pred_actions.append(pred_7)
            gt_actions.append(gt_7)

        pred_actions = np.stack(pred_actions, axis=0)
        gt_actions = np.stack(gt_actions, axis=0)

        grip_metrics = compute_binary_metrics(
            pred_actions[:, 6],
            gt_actions[:, 6],
            threshold=args.gripper_eval_threshold,
            gt_open_threshold=args.gripper_label_open_threshold,
        )

        ep_row = {
            "episode_index": ep_idx,
            "num_steps": int(T),
            "terminal_steps": int(np.sum(done)),
            "mse_total": mse(pred_actions, gt_actions),
            "mse_pos": mse(pred_actions[:, :3], gt_actions[:, :3]),
            "mse_rot": mse(pred_actions[:, 3:6], gt_actions[:, 3:6]),
            "mse_gripper": mse(pred_actions[:, 6], gt_actions[:, 6]),
            "mae_total": mae(pred_actions, gt_actions),
            "mae_pos": mae(pred_actions[:, :3], gt_actions[:, :3]),
            "mae_rot": mae(pred_actions[:, 3:6], gt_actions[:, 3:6]),
            "mae_gripper": mae(pred_actions[:, 6], gt_actions[:, 6]),
            "gt_gripper_mean": float(np.mean(gt_actions[:, 6])),
            "pred_gripper_mean": float(np.mean(pred_actions[:, 6])),
            **grip_metrics,
        }

        rows.append(ep_row)

        if (ep_idx + 1) % args.print_every == 0:
            print(
                f"[{ep_idx + 1}] "
                f"steps={T} "
                f"mse_total={ep_row['mse_total']:.6f} "
                f"mse_pos={ep_row['mse_pos']:.6f} "
                f"mse_rot={ep_row['mse_rot']:.6f} "
                f"mse_gripper={ep_row['mse_gripper']:.6f} "
                f"grip_acc={ep_row['grip_acc']:.3f} "
                f"open_f1={ep_row['grip_open_f1']:.3f}"
            )

    if len(rows) == 0:
        raise RuntimeError("No episodes were evaluated.")

    df = pd.DataFrame(rows)
    df = df.sort_values("mse_total", ascending=True).reset_index(drop=True)

    csv_path = os.path.join(args.save_dir, args.csv_name)
    df.to_csv(csv_path, index=False)

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_step": ckpt_step,
        "split": args.split,
        "num_episodes": int(len(df)),
        "window_size": int(args.window_size),
        "action_horizon": int(args.action_horizon),
        "mean_mse_total": float(df["mse_total"].mean()),
        "mean_mse_pos": float(df["mse_pos"].mean()),
        "mean_mse_rot": float(df["mse_rot"].mean()),
        "mean_mse_gripper": float(df["mse_gripper"].mean()),
        "mean_mae_total": float(df["mae_total"].mean()),
        "mean_mae_pos": float(df["mae_pos"].mean()),
        "mean_mae_rot": float(df["mae_rot"].mean()),
        "mean_mae_gripper": float(df["mae_gripper"].mean()),
        "mean_grip_acc": float(df["grip_acc"].mean()),
        "mean_grip_open_precision": float(df["grip_open_precision"].mean()),
        "mean_grip_open_recall": float(df["grip_open_recall"].mean()),
        "mean_grip_open_f1": float(df["grip_open_f1"].mean()),
        "mean_pred_open_rate": float(df["pred_open_rate"].mean()),
        "mean_gt_open_rate": float(df["gt_open_rate"].mean()),
        "mean_pred_transitions": float(df["pred_transitions"].mean()),
        "mean_gt_transitions": float(df["gt_transitions"].mean()),
        "mean_transition_acc": float(df["transition_acc"].mean()),
        "median_mse_total": float(df["mse_total"].median()),
        "best_episode_index": int(df.iloc[0]["episode_index"]),
        "best_episode_mse_total": float(df.iloc[0]["mse_total"]),
        "worst_episode_index": int(df.iloc[-1]["episode_index"]),
        "worst_episode_mse_total": float(df.iloc[-1]["mse_total"]),
        "top_5_episode_indices": [int(x) for x in df.head(5)["episode_index"].tolist()],
        "bottom_5_episode_indices": [int(x) for x in df.tail(5)["episode_index"].tolist()],
    }

    summary_path = os.path.join(args.save_dir, args.summary_name)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[INFO] Evaluation complete.")
    print(f"[INFO] Per-episode CSV saved to: {csv_path}")
    print(f"[INFO] Summary JSON saved to: {summary_path}")

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))

    print("\n===== TOP 10 EPISODES (lowest mse_total) =====")
    print(
        df[
            [
                "episode_index",
                "num_steps",
                "mse_total",
                "mse_pos",
                "mse_rot",
                "mse_gripper",
                "grip_acc",
                "grip_open_f1",
            ]
        ].head(10).to_string(index=False)
    )

    print("\n===== WORST 10 EPISODES (highest mse_total) =====")
    print(
        df[
            [
                "episode_index",
                "num_steps",
                "mse_total",
                "mse_pos",
                "mse_rot",
                "mse_gripper",
                "grip_acc",
                "grip_open_f1",
            ]
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()