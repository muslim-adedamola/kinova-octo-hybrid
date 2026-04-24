from absl import app, flags, logging
import copy
import json
import os

import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo
from octo.data.dataset import make_single_dataset

FLAGS = flags.FLAGS

# Flags
flags.DEFINE_string(
    "pretrained_path",
    "hf://rail-berkeley/octo-small-1.5",
    "Pretrained Octo checkpoint.",
)
flags.DEFINE_string(
    "tfds_data_dir",
    None,
    "TFDS data_dir that contains kinova_dataset.",
)
flags.DEFINE_string(
    "save_dir",
    "./finetune_ckpts_hybrid_arm_diffusion_gripper_bce",
    "Where to save checkpoints.",
)
flags.DEFINE_string(
    "dataset_name",
    "kinova_dataset",
    "TFDS dataset name.",
)
flags.DEFINE_string(
    "dataset_config",
    "default",
    "TFDS builder config.",
)
flags.DEFINE_string(
    "dataset_version",
    "0.1.5",
    "TFDS dataset version. Use the rebuilt version with seeded randomized train/val split.",
)

# Core training
flags.DEFINE_integer("batch_size", 64, "Global batch size.")
flags.DEFINE_integer("window_size", 2, "Observation history length.")
flags.DEFINE_integer("action_horizon", 4, "Action chunk length.")
flags.DEFINE_integer("steps", 50000, "Finetuning steps.")
flags.DEFINE_integer("warmup_steps", 2000, "Linear warmup steps.")
flags.DEFINE_float("learning_rate", 3e-5, "Peak finetuning learning rate.")
flags.DEFINE_float("weight_decay", 0.1, "AdamW weight decay.")
flags.DEFINE_float("grad_clip", 1.0, "Gradient clipping norm.")
flags.DEFINE_integer("shuffle_buffer", 256, "Shuffle buffer size.")
flags.DEFINE_integer("save_interval", 10000, "Checkpoint save interval.")
flags.DEFINE_integer("log_interval", 1000, "Train logging interval.")
flags.DEFINE_integer("eval_interval", 2000, "Validation interval.")
flags.DEFINE_integer("val_batches", 20, "Number of validation batches to average.")
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether to freeze transformer weights. "
    "Currently not implemented for this hybrid checkpoint format.",
)


# Gripper BCE head
flags.DEFINE_float(
    "gripper_bce_weight",
    5.0,
    "Weight on binary gripper BCE loss.",
)
flags.DEFINE_integer(
    "gripper_head_hidden_dim",
    256,
    "Hidden dimension for BCE gripper head.",
)
flags.DEFINE_float(
    "gripper_label_open_threshold",
    0.5,
    "Ground-truth gripper values >= this are treated as OPEN (1), else CLOSED (0).",
)
flags.DEFINE_float(
    "gripper_eval_threshold",
    0.5,
    "Threshold used to convert predicted sigmoid probability into binary OPEN/CLOSED during metrics.",
)

# Logging
flags.DEFINE_string(
    "wandb_project",
    "octo_kinova_goal_image_only",
    "Weights & Biases project name.",
)
flags.DEFINE_string(
    "wandb_run_name",
    "finetune_kinova_hybrid_arm_diffusion_gripper_bce",
    "Weights & Biases run name.",
)


# Utils
def load_tfds_dataset_statistics(
    tfds_data_dir: str,
    dataset_name: str,
    builder_config: str,
    version: str,
):
    stats_glob = os.path.join(
        tfds_data_dir,
        dataset_name,
        builder_config,
        version,
        "dataset_statistics_*.json",
    )
    matches = tf.io.gfile.glob(stats_glob)

    if not matches:
        print(f"[WARN] No dataset_statistics_*.json found. Looked for: {stats_glob}")
        print("[WARN] Falling back to Octo auto-computed dataset statistics.")
        return None

    matches_sorted = sorted(
        matches,
        key=lambda p: tf.io.gfile.stat(p).mtime_nsec,
        reverse=True,
    )
    stats_path = matches_sorted[0]
    tf.print("Using dataset statistics:", stats_path)

    with tf.io.gfile.GFile(stats_path, "r") as f:
        return json.load(f)


def make_lr_schedule(total_steps: int, warmup_steps: int, peak_lr: float):
    warmup_steps = min(warmup_steps, total_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )

    cosine = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=cosine_steps,
        alpha=0.0,
    )

    if warmup_steps == 0:
        return cosine

    return optax.join_schedules(
        schedules=[warmup, cosine],
        boundaries=[warmup_steps],
    )


def remove_unused_tokenizers(config: dict) -> dict:
    """
    Keep the model schema explicit and minimal:
      - observation tokenizer: only primary image
      - task tokenizer: only goal image carried as task['image_primary']
    """
    config = copy.deepcopy(config)

    obs_toks = config["model"].get("observation_tokenizers", {})
    task_toks = config["model"].get("task_tokenizers", {})

    print("Observation tokenizer keys before pruning:", list(obs_toks.keys()))
    print("Task tokenizer keys before pruning:", list(task_toks.keys()))

    for key in list(obs_toks.keys()):
        if key not in ["primary"]:
            del obs_toks[key]

    for key in list(task_toks.keys()):
        if key not in ["image_primary"]:
            del task_toks[key]

    print("Observation tokenizer keys after pruning:", list(obs_toks.keys()))
    print("Task tokenizer keys after pruning:", list(task_toks.keys()))

    return config


def convert_to_goal_task_batch(batch):
    """
    Converts dataset output into the exact clean model-facing schema.

    observation:
      image_primary
      timestep
      timestep_pad_mask
      task_completed
      pad_mask_dict (image_primary, timestep)

    task:
      image_primary
      pad_mask_dict (image_primary)

    Notes:
    - observation['image_goal'] is removed after conversion
    - no language fields are kept
    """
    batch = dict(batch)
    obs = dict(batch["observation"])

    if "image_goal" not in obs:
        raise KeyError("Expected observation['image_goal'] in batch, but it was missing.")

    goal_img = obs.pop("image_goal")[:, 0]  # [B, H, W, C]

    task = {
        "image_primary": goal_img,
    }

    obs_pad = dict(obs.get("pad_mask_dict", {}))
    task_pad = {}

    if "image_goal" in obs_pad:
        goal_pad = obs_pad.pop("image_goal")[:, 0]  # [B]
    else:
        goal_pad = tf.ones(tf.shape(goal_img)[0], dtype=tf.bool)

    task_pad["image_primary"] = goal_pad
    task["pad_mask_dict"] = task_pad

    obs["pad_mask_dict"] = obs_pad

    batch["observation"] = obs
    batch["task"] = task
    batch.pop("dataset_name", None)

    return batch


def build_dataset(train: bool, dataset_stats):
    """
    TFDS dataset should expose:
      - train
      - val

    with a seeded randomized episode-level split in the RLDS builder.
    """
    dataset_kwargs = dict(
        name=f"{FLAGS.dataset_name}/{FLAGS.dataset_config}",
        data_dir=FLAGS.tfds_data_dir,
        standardize_fn=ModuleSpec.create(kinova_rlds_to_octo),
        image_obs_keys={
            "primary": "image_primary",
            "goal": "goal_image",
        },
        action_normalization_mask=[True, True, True, True, True, True, False],
        dataset_statistics=dataset_stats,
    )

    traj_transform_kwargs = dict(
        window_size=FLAGS.window_size,
        action_horizon=FLAGS.action_horizon,
        subsample_length=None,
    )

    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),
            "goal": (256, 256),
        },
        image_dropout_prob=0.0,
    )

    ds = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        train=train,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
    )

    ds = ds.flatten()

    if train:
        ds = (
            ds.shuffle(FLAGS.shuffle_buffer, reshuffle_each_iteration=True)
            .repeat()
            .batch(FLAGS.batch_size, drop_remainder=True)
            .map(convert_to_goal_task_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        # Keep fixed batch shapes for JIT stability
        ds = (
            ds.batch(FLAGS.batch_size, drop_remainder=True)
            .map(convert_to_goal_task_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )

    return ds


def masked_mean(x, mask, eps=1e-8):
    x = jnp.asarray(x, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), eps)


def binary_stats(pred_bin, gt_bin, mask, eps=1e-8):
    pred_bin = jnp.asarray(pred_bin, dtype=jnp.float32)
    gt_bin = jnp.asarray(gt_bin, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)

    tp = jnp.sum(mask * pred_bin * gt_bin)
    fp = jnp.sum(mask * pred_bin * (1.0 - gt_bin))
    fn = jnp.sum(mask * (1.0 - pred_bin) * gt_bin)

    precision = tp / jnp.maximum(tp + fp, eps)
    recall = tp / jnp.maximum(tp + fn, eps)
    f1 = 2.0 * precision * recall / jnp.maximum(precision + recall, eps)

    return precision, recall, f1


def write_json(path, payload):
    with tf.io.gfile.GFile(path, "w") as f:
        json.dump(payload, f, indent=2)


def extract_readout_tensor(x):
    """
    Extract plain tensor for BCE gripper head from Octo transformer output.

    Expected output shape is either:
      [B, W, D]
    or
      [B, W, R, D]
    where R is number of readout tokens (often 1).
    """
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
            raise TypeError(
                f"Could not extract readout tensor from dict keys={list(x.keys())}"
            )
    else:
        if hasattr(x, "tokens"):
            y = x.tokens
        else:
            raise TypeError(
                f"Could not extract readout tensor from transformer output. "
                f"type={type(x)}, repr={repr(x)[:500]}"
            )

    if len(y.shape) == 4:
        y = y[:, :, 0, :]

    if len(y.shape) != 3:
        raise ValueError(
            f"Expected readout tensor with 3 dims after normalization, got shape={y.shape}"
        )

    return y


# Binary gripper head
class BinaryGripperHead(nn.Module):
    action_horizon: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, emb):
        # emb: [B, W, D]
        x = nn.Dense(self.hidden_dim)(emb)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        logits = nn.Dense(self.action_horizon)(x)  # [B, W, H]
        return logits


class HybridTrainState(train_state.TrainState):
    rng: jax.Array



# Main
def main(_):
    assert FLAGS.tfds_data_dir, "--tfds_data_dir is required"
    assert FLAGS.batch_size % jax.device_count() == 0, (
        "batch_size must be divisible by device_count"
    )

    save_dir = os.path.abspath(FLAGS.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_dir = os.path.join(save_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    initialize_compilation_cache()

    # Use TF only for the input pipeline.
    tf.config.set_visible_devices([], "GPU")

    wandb.init(
        project=FLAGS.wandb_project,
        name=FLAGS.wandb_run_name,
        config={
            "pretrained_path": FLAGS.pretrained_path,
            "dataset_name": FLAGS.dataset_name,
            "dataset_config": FLAGS.dataset_config,
            "dataset_version": FLAGS.dataset_version,
            "batch_size": FLAGS.batch_size,
            "window_size": FLAGS.window_size,
            "action_horizon": FLAGS.action_horizon,
            "steps": FLAGS.steps,
            "warmup_steps": FLAGS.warmup_steps,
            "learning_rate": FLAGS.learning_rate,
            "weight_decay": FLAGS.weight_decay,
            "grad_clip": FLAGS.grad_clip,
            "shuffle_buffer": FLAGS.shuffle_buffer,
            "freeze_transformer": FLAGS.freeze_transformer,
            "eval_interval": FLAGS.eval_interval,
            "val_batches": FLAGS.val_batches,
            "gripper_bce_weight": FLAGS.gripper_bce_weight,
            "gripper_head_hidden_dim": FLAGS.gripper_head_hidden_dim,
            "gripper_label_open_threshold": FLAGS.gripper_label_open_threshold,
            "gripper_eval_threshold": FLAGS.gripper_eval_threshold,
        },
    )

    logging.info("Loading pretrained model...")
    pretrained = OctoModel.load_pretrained(FLAGS.pretrained_path)

    config = remove_unused_tokenizers(pretrained.config)

    dataset_stats = load_tfds_dataset_statistics(
        tfds_data_dir=FLAGS.tfds_data_dir,
        dataset_name=FLAGS.dataset_name,
        builder_config=FLAGS.dataset_config,
        version=FLAGS.dataset_version,
    )

    if dataset_stats is None:
        print("[INFO] No cached dataset statistics JSON found.")
        print("[INFO] Octo will try to load from cache or compute and save statistics automatically.")

    train_ds = build_dataset(train=True, dataset_stats=dataset_stats)
    val_ds = build_dataset(train=False, dataset_stats=dataset_stats)

    train_iter = train_ds.as_numpy_iterator()
    val_iter = val_ds.as_numpy_iterator()

    example_batch = next(train_iter)

    print("obs keys:", list(example_batch["observation"].keys()))
    print("obs pad_mask keys:", list(example_batch["observation"]["pad_mask_dict"].keys()))
    print("action shape:", example_batch["action"].shape)
    print("task keys:", list(example_batch["task"].keys()))
    print("task pad_mask keys:", list(example_batch["task"]["pad_mask_dict"].keys()))
    if "image_primary" in example_batch["task"]:
        print("task image shape:", example_batch["task"]["image_primary"].shape)

    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor=None,
        verbose=False,
        dataset_statistics=dataset_stats,
    )
    model = model.replace(params=pretrained.params)
    del pretrained

    init_rng = jax.random.PRNGKey(0)

    bound = model.module.bind({"params": model.params}, rngs={"dropout": init_rng})
    example_emb = bound.octo_transformer(
        example_batch["observation"],
        example_batch["task"],
        example_batch["observation"]["timestep_pad_mask"],
        train=False,
    )

    print("type(example_emb):", type(example_emb))
    if hasattr(example_emb, "keys"):
        print("example_emb keys:", list(example_emb.keys()))

    example_readout = extract_readout_tensor(example_emb)
    print("example_readout shape:", example_readout.shape)
    print("example arm target shape:", example_batch["action"][..., :6].shape)
    print("example gripper target shape:", example_batch["action"][..., 6].shape)

    gripper_head = BinaryGripperHead(
        action_horizon=FLAGS.action_horizon,
        hidden_dim=FLAGS.gripper_head_hidden_dim,
    )
    gripper_head_params = gripper_head.init(init_rng, example_readout)["params"]

    params = {
        "octo": model.params,
        "gripper_head": gripper_head_params,
    }

    lr_schedule = make_lr_schedule(
        total_steps=FLAGS.steps,
        warmup_steps=FLAGS.warmup_steps,
        peak_lr=FLAGS.learning_rate,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.grad_clip),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=FLAGS.weight_decay,
        ),
    )

    if FLAGS.freeze_transformer:
        print("[WARN] freeze_transformer=True is not implemented in this hybrid script.")
        print("[WARN] Proceeding with full-parameter optimization.")

    state = HybridTrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
        rng=init_rng,
    )

    # Diffusion head trains only arm / ee dims: xyz + rpy/rot6? first 6 dims only.
    cont_dim_mask = jnp.asarray([1, 1, 1, 1, 1, 1, 0], dtype=jnp.float32)

    def loss_fn(params, batch, rng, train=True):
        bound = model.module.bind({"params": params["octo"]}, rngs={"dropout": rng})

        emb = bound.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        readout_emb = extract_readout_tensor(emb)

        
        # Continuous arm loss via original Octo diffusion head, first 6 dims only
        action_pad_mask = jnp.asarray(batch["action_pad_mask"], dtype=jnp.float32)
        cont_mask = action_pad_mask * cont_dim_mask

        cont_loss, cont_metrics = bound.heads["action"].loss(
            emb,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            cont_mask,
            train=train,
        )

        # BCE gripper loss on dim 6 only
        gripper_logits = gripper_head.apply(
            {"params": params["gripper_head"]},
            readout_emb,
        )  # [B, W, H]

        gt_gripper_raw = jnp.asarray(batch["action"][..., 6], dtype=jnp.float32)
        gt_gripper = (gt_gripper_raw >= FLAGS.gripper_label_open_threshold).astype(jnp.float32)

        gripper_mask = jnp.asarray(batch["action_pad_mask"][..., 6], dtype=jnp.float32)

        gripper_bce = optax.sigmoid_binary_cross_entropy(gripper_logits, gt_gripper)
        gripper_bce_loss = masked_mean(gripper_bce, gripper_mask)

        total_loss = cont_loss + FLAGS.gripper_bce_weight * gripper_bce_loss

        pred_gripper_prob = jax.nn.sigmoid(gripper_logits)
        pred_gripper_bin = (pred_gripper_prob >= FLAGS.gripper_eval_threshold).astype(jnp.float32)

        gripper_acc = masked_mean(
            (pred_gripper_bin == gt_gripper).astype(jnp.float32),
            gripper_mask,
        )
        pred_open_rate = masked_mean(pred_gripper_bin, gripper_mask)
        gt_open_rate = masked_mean(gt_gripper, gripper_mask)
        gripper_prob_mean = masked_mean(pred_gripper_prob, gripper_mask)

        open_precision, open_recall, open_f1 = binary_stats(
            pred_gripper_bin,
            gt_gripper,
            gripper_mask,
        )

        metrics = {
            "loss_continuous": cont_loss,
            "loss_gripper_bce": gripper_bce_loss,
            "loss_total": total_loss,
            "gripper_acc": gripper_acc,
            "pred_open_rate": pred_open_rate,
            "gt_open_rate": gt_open_rate,
            "pred_open_prob_mean": gripper_prob_mean,
            "open_precision": open_precision,
            "open_recall": open_recall,
            "open_f1": open_f1,
        }

        flat_cont = flax.traverse_util.flatten_dict({"cont": cont_metrics}, sep="/")
        for k, v in flat_cont.items():
            metrics[k] = v

        return total_loss, metrics

    @jax.jit
    def train_step(state, batch):
        rng, drng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, drng, True
        )
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng=rng)
        return state, loss, info

    @jax.jit
    def eval_step(state, batch):
        loss, info = loss_fn(state.params, batch, state.rng, train=False)
        return loss, info

    def save_hybrid_checkpoint(ckpt_dir, state, step, best_val_loss=None, overwrite=False, keep=3):
        target = {
            "params": jax.device_get(state.params),
            "step": int(step),
            "model_type": "octo_arm_diffusion_gripper_bce",
            "action_horizon": int(FLAGS.action_horizon),
            "gripper_head_hidden_dim": int(FLAGS.gripper_head_hidden_dim),
            "gripper_label_open_threshold": float(FLAGS.gripper_label_open_threshold),
            "gripper_eval_threshold": float(FLAGS.gripper_eval_threshold),
            "pretrained_path": FLAGS.pretrained_path,
        }
        if best_val_loss is not None:
            target["best_val_loss"] = float(best_val_loss)

        checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=target,
            step=step,
            overwrite=overwrite,
            keep=keep,
        )

    # Save metadata once for matching eval / inference scripts
    write_json(
        os.path.join(save_dir, "hybrid_checkpoint_meta.json"),
        {
            "model_type": "octo_arm_diffusion_gripper_bce",
            "pretrained_path": FLAGS.pretrained_path,
            "dataset_name": FLAGS.dataset_name,
            "dataset_config": FLAGS.dataset_config,
            "dataset_version": FLAGS.dataset_version,
            "action_horizon": FLAGS.action_horizon,
            "window_size": FLAGS.window_size,
            "gripper_head_hidden_dim": FLAGS.gripper_head_hidden_dim,
            "gripper_label_open_threshold": FLAGS.gripper_label_open_threshold,
            "gripper_eval_threshold": FLAGS.gripper_eval_threshold,
            "notes": [
                "Use Octo sampled action only for dims 0:6.",
                "Replace gripper dim with BCE head prediction at inference time.",
                "Checkpoint contains params['octo'] and params['gripper_head']."
            ],
        },
    )

    best_val_loss = np.inf
    best_step = -1

    logging.info("Training hybrid run: Octo diffusion arm + BCE gripper...")
    for i in tqdm.tqdm(range(FLAGS.steps), dynamic_ncols=True):
        batch = next(train_iter)
        state, loss, info = train_step(state, batch)

        step = i + 1

        if step % FLAGS.log_interval == 0:
            info = jax.device_get(info)
            loss_val = float(jax.device_get(loss))

            metrics = {f"train/{k}": float(v) for k, v in info.items()}
            metrics["train/loss"] = loss_val
            metrics["train/lr"] = float(lr_schedule(step))

            wandb.log(metrics, step=step)

        if step % FLAGS.eval_interval == 0:
            val_losses = []
            val_infos = []

            for _ in range(FLAGS.val_batches):
                vbatch = next(val_iter)
                vloss, vinfo = eval_step(state, vbatch)
                val_losses.append(float(jax.device_get(vloss)))
                val_infos.append(jax.device_get(vinfo))

            mean_val_loss = float(np.mean(val_losses))

            val_metrics = {}
            if len(val_infos) > 0:
                keys = val_infos[0].keys()
                for k in keys:
                    vals = [float(x[k]) for x in val_infos]
                    val_metrics[f"val/{k}"] = float(np.mean(vals))

            val_metrics["val/loss"] = mean_val_loss
            wandb.log(val_metrics, step=step)

            print(
                f"[eval] step={step} "
                f"val/loss={mean_val_loss:.6f} "
                f"val/gripper_acc={val_metrics.get('val/gripper_acc', float('nan')):.4f} "
                f"val/open_f1={val_metrics.get('val/open_f1', float('nan')):.4f} "
                f"val/pred_open_rate={val_metrics.get('val/pred_open_rate', float('nan')):.4f}"
            )

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_step = step
                print(f"[best] step={step} new best val/loss={best_val_loss:.6f}")

                save_hybrid_checkpoint(
                    ckpt_dir=best_dir,
                    state=state,
                    step=step,
                    best_val_loss=best_val_loss,
                    overwrite=True,
                    keep=1,
                )

        if step % FLAGS.save_interval == 0:
            save_hybrid_checkpoint(
                ckpt_dir=save_dir,
                state=state,
                step=step,
                overwrite=False,
                keep=3,
            )

        del batch

    save_hybrid_checkpoint(
        ckpt_dir=save_dir,
        state=state,
        step=FLAGS.steps,
        overwrite=False,
        keep=3,
    )

    if best_step == -1:
        print("[WARN] No validation checkpoint was selected. Saving final weights to best_dir.")
        save_hybrid_checkpoint(
            ckpt_dir=best_dir,
            state=state,
            step=FLAGS.steps,
            overwrite=True,
            keep=1,
        )
    else:
        print(
            f"[INFO] Best checkpoint was selected at step {best_step} "
            f"with val/loss={best_val_loss:.6f}"
        )


if __name__ == "__main__":
    app.run(main)
