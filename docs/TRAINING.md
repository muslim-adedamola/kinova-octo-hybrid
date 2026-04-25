# Training Guide

This document explains how to finetune the hybrid Octo policy used in this repository.

The training pipeline adapts a pretrained Octo model to a Kinova bottle pick-and-lift task using:

- Octo's original diffusion action head for continuous arm motion;
- a separate binary BCE gripper head for open/close prediction;
- goal-image conditioning;
- TFDS/RLDS Kinova demonstrations.

The overall pipeline is:

```text
Raw Kinova demonstrations
→ TFDS/RLDS dataset
→ Octo standardization
→ Hybrid Octo + BCE gripper finetuning
→ Offline evaluation
→ Real-robot deployment
```

---

## 1. Training Objective

The Kinova action used for training has 7 dimensions:

```text
[dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, gripper_state_bin]
```

In this project, Octo is finetuned using RGB image observations only. No proprioceptive observations are used as model inputs. The hybrid model separates the action prediction problem into two parts.

### Continuous arm action

The first 6 dimensions are modeled with Octo's diffusion action head:

```text
[dx, dy, dz, dtheta_x, dtheta_y, dtheta_z]
```

These correspond to end-effector translation and rotation deltas.

### Binary gripper action

The 7th dimension is modeled with a separate binary classification head:

```text
gripper_state_bin
```

The gripper head is trained using binary cross-entropy.

For the gripper, this project follows the absolute gripper convention used by Octo (as reported in the research paper):

```text
1 = gripper open
0 = gripper closed
```

The training script saves both sets of parameters:

```text
params["octo"]
params["gripper_head"]
```

---

## 2. Dataset Assumptions

Before training, the raw Kinova demonstrations must be converted into TFDS/RLDS format.

See [DATASET.md](DATASET.md).

The default dataset after conversion and as used by the training script is:

```text
dataset_name    = kinova_dataset
dataset_config  = default
dataset_version = 0.1.5
```

The default TFDS directory is passed at runtime using:

```bash
--tfds_data_dir /path/to/tensorflow_datasets
```

The expected built dataset path is usually:

```text
/path/to/tensorflow_datasets/kinova_dataset/default/version_number_of_dataset/
```

---

## 3. Octo Standardization Function

The training script expects the Kinova standardization function to be importable from the Octo package path:

```python
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo
```

Copy the standardization file into your local Octo source tree:

```bash
cp scripts/dataset/kinova_standardize_octo.py /path/to/octo/octo/data/kinova_standardize_octo.py
```

Verify:

```bash
python - <<'PY'
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo
print("Kinova standardization import OK")
PY
```

---

## 4. Goal-Image Conditioning

The dataset standardization function creates goal images using hindsight goal relabeling, following the goal-conditioning setup used as reported in Octo.

During training:

- the current observation image is passed as `observation["image_primary"]`;
- the goal image is passed as `task["image_primary"]`;
- language conditioning is not used.

The training script converts each batch into this model-facing schema:

```text
observation:
  image_primary
  timestep
  timestep_pad_mask
  task_completed
  pad_mask_dict

task:
  image_primary
  pad_mask_dict
```

This keeps the pipeline focused on goal-image-conditioned manipulation.

---

## 5. Main Training Script

The main training script is:

```text
scripts/training/finetune_hybrid_octo_bce.py
```

It performs the following:

- loads a pretrained Octo checkpoint;
- prunes unused tokenizers;
- builds the Kinova TFDS/RLDS dataset;
- converts goal images into Octo task conditioning;
- trains the Octo diffusion action head on the first 6 action dimensions;
- trains a BCE gripper head on the gripper action dimension;
- logs metrics to Weights & Biases;
- evaluates periodically on the validation split;
- saves regular checkpoints and the best validation checkpoint.

---

## 6. Example Training Command

Run from the repository root:

```bash
python scripts/training/finetune_hybrid_octo_bce.py \
  --pretrained_path hf://rail-berkeley/octo-small-1.5 \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --save_dir ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce \
  --dataset_name kinova_dataset \
  --dataset_config default \
  --dataset_version 0.1.5 \
  --batch_size 64 \
  --window_size 2 \
  --action_horizon 4 \
  --steps 50000 \
  --warmup_steps 2000 \
  --learning_rate 3e-5 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --shuffle_buffer 256 \
  --eval_interval 2000 \
  --val_batches 20 \
  --gripper_bce_weight 5.0 \
  --gripper_head_hidden_dim 256
```

Replace `/path/to/tensorflow_datasets` with the TFDS directory used when building the dataset.
Replace dataset version number, that is, 0.1.5 with the TFDS/RLDS version number generated for your dataset.

---

## 7. Key Hyperparameters

Default hyperparameters used for this project:

| Parameter | Value | Meaning |
|---|---:|---|
| `pretrained_path` | `hf://rail-berkeley/octo-small-1.5` | pretrained Octo checkpoint |
| `batch_size` | `64` | global training batch size |
| `window_size` | `2` | observation history length |
| `action_horizon` | `4` | predicted action chunk length |
| `steps` | `50000` | total finetuning steps |
| `warmup_steps` | `2000` | learning-rate warmup |
| `learning_rate` | `3e-5` | peak finetuning learning rate |
| `weight_decay` | `0.1` | AdamW weight decay |
| `grad_clip` | `1.0` | global gradient clipping norm |
| `shuffle_buffer` | `256` | TF dataset shuffle buffer |
| `gripper_bce_weight` | `5.0` | BCE gripper loss weight |
| `gripper_head_hidden_dim` | `256` | hidden dimension of BCE gripper head |

These hyperparameters were selected to closely follow the Octo finetuning recipe (as reported in the paper) where applicable.

Full finetuning is used in this project: the Octo parameters are optimized together with the BCE gripper head. The transformer is not frozen in the released training script.

The script includes a `--freeze_transformer` flag, but parameter freezing is not currently implemented for this hybrid checkpoint format. Support for parameter-freezing modes, such as training only the gripper head or only selected Octo heads, will be added in a future update. Users who want partial finetuning can adapt the optimizer/mask logic from Octo's official finetuning setup, which supports modes such as `head_only`, `head_mlp_only`, and `full`.

---

## 8. Gripper Labeling

The gripper label is created from the 7th action dimension.

Default threshold:

```bash
--gripper_label_open_threshold 0.5
```

This means:

```text
gripper >= 0.5 → OPEN
gripper < 0.5  → CLOSED
```

During validation metrics, predicted probabilities are binarized using:

```bash
--gripper_eval_threshold 0.5
```

You can tune these if your dataset uses a different gripper convention.

---

## 9. Checkpoint Output

The training script saves checkpoints to `--save_dir`.

Example:

```text
checkpoints/
  finetune_ckpts_hybrid_arm_diffusion_gripper_bce/
    checkpoint_*
    hybrid_checkpoint_meta.json
    best/
      checkpoint_*
```

The `best/` directory contains the best validation checkpoint according to validation loss.

The checkpoint stores:

```text
params["octo"]
params["gripper_head"]
```

It also stores metadata such as:

```text
model_type
action_horizon
gripper_head_hidden_dim
gripper_label_open_threshold
gripper_eval_threshold
pretrained_path
```

---

## 10. Validation During Training

The training script periodically runs validation every:

```bash
--eval_interval 2000
```

It averages validation metrics over:

```bash
--val_batches 20
```

Important validation metrics include:

```text
val/loss
val/loss_continuous
val/loss_gripper_bce
val/gripper_acc
val/open_precision
val/open_recall
val/open_f1
val/pred_open_rate
val/gt_open_rate
```

Here, `val/pred_open_rate` is the fraction of gripper predictions classified as open, while `val/gt_open_rate` is the corresponding ground-truth open rate. During training, `val/pred_open_rate` should remain reasonably close to `val/gt_open_rate`; large differences can indicate gripper bias. The best checkpoint is saved when validation loss improves.

---

## 11. Weights & Biases Logging

The script logs to Weights & Biases by default.

Default project:

```text
octo_kinova_goal_image_only
```

Default run name:

```text
finetune_kinova_hybrid_arm_diffusion_gripper_bce
```

You can override them:

```bash
--wandb_project <project_name>
--wandb_run_name <run_name>
```

To run W&B offline:

```bash
export WANDB_MODE=offline
```

---

## 12. GPU and Memory Notes

The training script uses JAX for model computation and TensorFlow for dataset loading.

The script disables TensorFlow GPU visibility:

```python
tf.config.set_visible_devices([], "GPU")
```

This keeps GPU memory available for JAX.

It also uses:

```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

to reduce aggressive JAX memory preallocation.

If you run out of memory, try:

```text
reduce batch_size
reduce val_batches
use Octo-Small instead of Octo-Base
ensure TensorFlow is not using the GPU
```

---

## 13. Common Issues

### `kinova_standardize_octo` import error

Error:

```text
ModuleNotFoundError: No module named 'octo.data.kinova_standardize_octo'
```

Fix:

```bash
cp scripts/dataset/kinova_standardize_octo.py /path/to/octo/octo/data/kinova_standardize_octo.py
```

Then verify:

```bash
python -c "from octo.data.kinova_standardize_octo import kinova_rlds_to_octo; print('OK')"
```

### Dataset not found

Check that the dataset was built successfully:

```text
/path/to/tensorflow_datasets/kinova_dataset/default/0.1.5/
```

Also confirm that the training command uses the same TFDS directory:

```bash
--tfds_data_dir /path/to/tensorflow_datasets
```

### No dataset statistics found

The training script looks for:

```text
dataset_statistics_*.json
```

inside the built TFDS dataset directory.

If it does not find one, it falls back to Octo's automatic dataset statistics handling.

### JAX does not see GPU

Check:

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

If only CPU appears, reinstall the CUDA-compatible JAX/JAXLIB version for your system.

This project was tested with:

```text
jax==0.4.20
jaxlib==0.4.20+cuda11.cudnn86
```

### SciPy `tril` error

Some Octo setups may encounter issues related to newer SciPy versions.

This repository pins:

```text
scipy>=1.6.0,<1.13
```

to avoid compatibility problems that can occur with some Octo/JAX/TensorFlow dependency combinations.

---

## 14. After Training

After training finishes, run offline evaluation:

```bash
python scripts/evaluation/eval_hybrid_octo_bce.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --split val \
  --save_dir ./offline_eval_results_hybrid
```

For gripper-specific diagnostics:

```bash
python scripts/evaluation/eval_gripper_traces.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --split val \
  --save_dir ./offline_eval_gripper_traces \
  --save_gripper_traces \
  --num_trace_episodes 5
```

Then proceed to deployment only after verifying offline behavior and reviewing the model's gripper predictions.

See [DEPLOYMENT.md](DEPLOYMENT.md) for real-robot deployment notes.
