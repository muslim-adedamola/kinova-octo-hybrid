"""
Offline gripper-trace evaluator for the Kinova + Octo hybrid policy.

This script replays TFDS/RLDS episodes frame by frame, runs the hybrid policy,
and compares the predicted gripper probability against the ground-truth gripper
label. It can save per-episode CSV traces and PNG plots, which are useful for
checking whether the BCE gripper head closes at the right time in a trajectory.
"""

import os
import json
import argparse

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import flax.linen as nn
from flax.training import checkpoints

from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.data.dataset import make_single_dataset
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
tf.config.set_visible_devices([], "GPU")


class BinaryGripperHead(nn.Module):
    """Small MLP head that predicts OPEN probability logits for each chunk step."""
    action_horizon: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, emb):
        x = nn.Dense(self.hidden_dim)(emb)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        logits = nn.Dense(self.action_horizon)(x)
        return logits


def load_dataset_statistics(
    stats_json=None,
    tfds_data_dir=None,
    dataset_name="kinova_dataset",
    dataset_config="default",
    dataset_version="0.1.5",
):
    """Load cached TFDS dataset statistics used for Octo action unnormalization."""
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
    """Convert one raw RLDS episode to NumPy arrays for simple step replay."""
    steps = list(ep["steps"].as_numpy_iterator()) if hasattr(ep["steps"], "as_numpy_iterator") else list(ep["steps"])

    image = np.stack([np.asarray(s["observation"]["image"], dtype=np.uint8) for s in steps], axis=0)
    action = np.stack([np.asarray(s["action"], dtype=np.float32) for s in steps], axis=0)

    if action.shape[-1] == 8:
        action = action[..., :7]

    return {
        "image": image,
        "action": action,
        "length": image.shape[0],
    }


def build_obs_and_task(ep_std, t, window_size):
    """Build the model-facing observation window and final-frame goal task.

    This evaluator uses the final episode frame as the goal image for trace
    analysis, matching the common offline diagnostic setup.
    """
    image = ep_std["image"]
    start = max(0, t - window_size + 1)
    frames = image[start:t + 1]

    if frames.shape[0] < window_size:
        pad_count = window_size - frames.shape[0]
        pad = np.repeat(frames[:1], repeats=pad_count, axis=0)
        frames = np.concatenate([pad, frames], axis=0)

    frames = np.stack(
        [cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA) for f in frames],
        axis=0,
    ).astype(np.uint8)

    goal = cv2.resize(image[-1], (256, 256), interpolation=cv2.INTER_AREA).astype(np.uint8)

    obs = {
        "image_primary": frames[None],
        "timestep": np.arange(window_size, dtype=np.int32)[None],
        "task_completed": np.zeros((1, window_size, 4), dtype=bool),
        "timestep_pad_mask": np.ones((1, window_size), dtype=bool),
        "pad_mask_dict": {
            "image_primary": np.ones((1, window_size), dtype=bool),
            "timestep": np.ones((1, window_size), dtype=bool),
        },
    }

    task = {
        "image_primary": goal[None],
        "pad_mask_dict": {
            "image_primary": np.ones((1,), dtype=bool),
        },
    }

    return obs, task


def should_save_trace(ep_idx, args):
    if not args.save_gripper_traces:
        return False
    if args.trace_episode_indices is not None and len(args.trace_episode_indices) > 0:
        return ep_idx in set(args.trace_episode_indices)
    return ep_idx < args.num_trace_episodes


def save_gripper_trace_csv(save_dir, ep_idx, pred_gripper, gt_gripper, threshold=0.5, gt_open_threshold=0.5):
    """Write timestep-wise predicted/ground-truth gripper values to CSV."""
    os.makedirs(save_dir, exist_ok=True)

    pred_gripper = np.asarray(pred_gripper, dtype=np.float32)
    gt_gripper = np.asarray(gt_gripper, dtype=np.float32)

    pred_bin = (pred_gripper >= threshold).astype(np.int32)
    gt_bin = (gt_gripper >= gt_open_threshold).astype(np.int32)

    df = pd.DataFrame({
        "timestep": np.arange(len(pred_gripper), dtype=np.int32),
        "pred_gripper_prob": pred_gripper,
        "gt_gripper_raw": gt_gripper,
        "pred_gripper_bin": pred_bin,
        "gt_gripper_bin": gt_bin,
    })

    path = os.path.join(save_dir, f"episode_{ep_idx:03d}_gripper_trace.csv")
    df.to_csv(path, index=False)
    return path


def save_gripper_trace_plot(save_dir, ep_idx, pred_gripper, gt_gripper, threshold=0.5, gt_open_threshold=0.5):
    """Save a quick visual comparison of predicted and target gripper traces."""
    os.makedirs(save_dir, exist_ok=True)

    pred_gripper = np.asarray(pred_gripper, dtype=np.float32)
    gt_gripper = np.asarray(gt_gripper, dtype=np.float32)
    t = np.arange(len(pred_gripper), dtype=np.int32)

    plt.figure(figsize=(10, 4))
    plt.plot(t, pred_gripper, label="pred_gripper_prob")
    plt.plot(t, gt_gripper, label="gt_gripper_raw")
    plt.axhline(threshold, linestyle="--", linewidth=1, label=f"pred_threshold={threshold}")
    plt.axhline(gt_open_threshold, linestyle=":", linewidth=1, label=f"gt_open_threshold={gt_open_threshold}")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("timestep")
    plt.ylabel("gripper value")
    plt.title(f"Episode {ep_idx} gripper trace")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, f"episode_{ep_idx:03d}_gripper_trace.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def compute_binary_metrics(pred_prob, gt_raw, threshold=0.5, gt_open_threshold=0.5):
    """Compute gripper classification and transition metrics for one episode."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--tfds_data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="kinova_dataset")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_version", type=str, default="0.1.5")
    parser.add_argument("--stats_json", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--gripper_eval_threshold", type=float, default=0.5)
    parser.add_argument("--gripper_label_open_threshold", type=float, default=0.5)
    parser.add_argument("--gripper_head_hidden_dim", type=int, default=256)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--save_gripper_traces", action="store_true")
    parser.add_argument("--num_trace_episodes", type=int, default=5)
    parser.add_argument("--trace_episode_indices", type=int, nargs="*", default=None)
    parser.add_argument("--max_episodes", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("[INFO] Loading hybrid checkpoint...")
    ckpt = checkpoints.restore_checkpoint(args.checkpoint_path, target=None)
    params = ckpt["params"]
    ckpt_step = ckpt.get("step", None)

    dataset_stats = load_dataset_statistics(
        stats_json=args.stats_json,
        tfds_data_dir=args.tfds_data_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_version=args.dataset_version,
    )

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
    if action_stats is not None:
        action_stats = tree_lists_to_arrays(action_stats)

    gripper_head = BinaryGripperHead(
        action_horizon=args.action_horizon,
        hidden_dim=args.gripper_head_hidden_dim,
    )

    @jax.jit
    def predict_step(octo_params, gripper_params, obs, task, rng):
        """Return a hybrid action chunk for one observation/task pair."""
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
        )[..., :6]

        cont_actions = jnp.asarray(cont_actions)
        if cont_actions.ndim == 4:
            cont_actions = cont_actions[:, -1, :, :]
        elif cont_actions.ndim != 3:
            raise ValueError(f"Unexpected continuous action shape: {cont_actions.shape}")

        readout_emb = extract_readout_tensor(emb)
        readout_emb = readout_emb[:, -1, :]

        gripper_logits = gripper_head.apply({"params": gripper_params}, readout_emb)
        gripper_prob = jax.nn.sigmoid(gripper_logits)[..., None]

        actions = jnp.concatenate([cont_actions, gripper_prob], axis=-1)
        return actions

    print(f"[INFO] Building TFDS dataset for split='{args.split}' ...")
    builder = tfds.builder(
        f"{args.dataset_name}/{args.dataset_config}:{args.dataset_version}",
        data_dir=args.tfds_data_dir,
    )
    raw_ds = builder.as_dataset(split=args.split)

    rng = jax.random.PRNGKey(0)
    rows = []

    print(f"[INFO] Evaluating split='{args.split}' ...")
    for ep_idx, ep in enumerate(raw_ds):
        if args.max_episodes is not None and ep_idx >= args.max_episodes:
            break

        ep_std = standardize_episode_from_rlds_np(ep)
        T = ep_std["length"]

        pred_actions = []
        gt_actions = []

        for t in range(T - 1):
            obs_np, task_np = build_obs_and_task(ep_std, t, args.window_size)

            obs_jax = jax.tree_util.tree_map(jnp.asarray, obs_np)
            task_jax = jax.tree_util.tree_map(jnp.asarray, task_np)

            rng, step_rng = jax.random.split(rng)
            pred = predict_step(
                params["octo"],
                params["gripper_head"],
                obs_jax,
                task_jax,
                step_rng,
            )

            pred = np.asarray(pred)[0, 0]  # [7]
            gt = ep_std["action"][t]

            pred_actions.append(pred)
            gt_actions.append(gt)

        pred_actions = np.stack(pred_actions, axis=0)
        gt_actions = np.stack(gt_actions, axis=0)

        mse_total = float(np.mean((pred_actions - gt_actions) ** 2))
        mse_pos = float(np.mean((pred_actions[:, :3] - gt_actions[:, :3]) ** 2))
        mse_rot = float(np.mean((pred_actions[:, 3:6] - gt_actions[:, 3:6]) ** 2))
        mse_gripper = float(np.mean((pred_actions[:, 6] - gt_actions[:, 6]) ** 2))

        grip_metrics = compute_binary_metrics(
            pred_actions[:, 6],
            gt_actions[:, 6],
            threshold=args.gripper_eval_threshold,
            gt_open_threshold=args.gripper_label_open_threshold,
        )

        row = {
            "episode_index": ep_idx,
            "num_steps": int(pred_actions.shape[0]),
            "mse_total": mse_total,
            "mse_pos": mse_pos,
            "mse_rot": mse_rot,
            "mse_gripper": mse_gripper,
            **grip_metrics,
        }
        rows.append(row)

        print(
            f"[{ep_idx + 1}] "
            f"steps={row['num_steps']} "
            f"mse_total={row['mse_total']:.6f} "
            f"mse_gripper={row['mse_gripper']:.6f} "
            f"grip_acc={row['grip_acc']:.3f} "
            f"open_f1={row['grip_open_f1']:.3f} "
            f"pred_open={row['pred_open_rate']:.3f} "
            f"gt_open={row['gt_open_rate']:.3f} "
            f"pred_trans={row['pred_transitions']} "
            f"gt_trans={row['gt_transitions']}"
        )

        if should_save_trace(ep_idx, args):
            trace_dir = os.path.join(args.save_dir, "gripper_traces")
            csv_path = save_gripper_trace_csv(
                trace_dir,
                ep_idx,
                pred_actions[:, 6],
                gt_actions[:, 6],
                args.gripper_eval_threshold,
                args.gripper_label_open_threshold,
            )
            png_path = save_gripper_trace_plot(
                trace_dir,
                ep_idx,
                pred_actions[:, 6],
                gt_actions[:, 6],
                args.gripper_eval_threshold,
                args.gripper_label_open_threshold,
            )
            print(f"[TRACE] saved gripper trace CSV: {csv_path}")
            print(f"[TRACE] saved gripper trace PNG: {png_path}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.save_dir, "per_episode_metrics.csv"), index=False)

    summary = {
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_step": ckpt_step,
        "split": args.split,
        "num_episodes": int(len(df)),
        "window_size": args.window_size,
        "action_horizon": args.action_horizon,
        "gripper_eval_threshold": args.gripper_eval_threshold,
        "gripper_label_open_threshold": args.gripper_label_open_threshold,
        "mean_mse_total": float(df["mse_total"].mean()),
        "mean_mse_pos": float(df["mse_pos"].mean()),
        "mean_mse_rot": float(df["mse_rot"].mean()),
        "mean_mse_gripper": float(df["mse_gripper"].mean()),
        "mean_grip_acc": float(df["grip_acc"].mean()),
        "mean_grip_open_precision": float(df["grip_open_precision"].mean()),
        "mean_grip_open_recall": float(df["grip_open_recall"].mean()),
        "mean_grip_open_f1": float(df["grip_open_f1"].mean()),
        "mean_pred_open_rate": float(df["pred_open_rate"].mean()),
        "mean_gt_open_rate": float(df["gt_open_rate"].mean()),
        "mean_pred_transitions": float(df["pred_transitions"].mean()),
        "mean_gt_transitions": float(df["gt_transitions"].mean()),
        "mean_transition_acc": float(df["transition_acc"].mean()),
        "best_episode_index": int(df["mse_total"].idxmin()),
        "best_episode_mse_total": float(df["mse_total"].min()),
        "worst_episode_index": int(df["mse_total"].idxmax()),
        "worst_episode_mse_total": float(df["mse_total"].max()),
    }

    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[INFO] Evaluation complete.")
    print(f"[INFO] Per-episode CSV saved to: {os.path.join(args.save_dir, 'per_episode_metrics.csv')}")
    print(f"[INFO] Summary JSON saved to: {os.path.join(args.save_dir, 'summary.json')}")
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()