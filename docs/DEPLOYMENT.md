# Deployment Guide

This document explains how to run the finetuned hybrid Octo policy on the Kinova robot.

The deployment script performs real-time goal-image-conditioned inference using:

- an Intel RealSense RGB camera in the scene;
- a finetuned Octo checkpoint;
- a BCE gripper head;
- Kinova Kortex API control;
- latency-aware automatic receding-horizon execution.

The main deployment script is:

```text
scripts/deployment/run_finetuned_hybrid_model.py
```

---

## 1. Safety First

Real-robot deployment can be unsafe if the camera crop, goal image, action scaling, gripper behavior, or robot workspace is misconfigured.

Before running with `--execute_actions`:

- keep the emergency stop within reach;
- clear the robot workspace;
- confirm the robot is in a safe start pose;
- run in shadow mode first;
- confirm the live crop and model input preview;
- use conservative speed limits;
- supervise the robot continuously;
- stop immediately if motion becomes unexpected.

The deployment script can run without sending commands to the robot. This is called shadow mode and should always be tested first.

---

## 2. Deployment Pipeline

The live deployment loop follows this sequence:

```text
RealSense RGB frame
→ optional deployment crop
→ resize to 256×256
→ observation window
→ Octo + BCE gripper inference
→ selected arm/gripper chunk
→ latency-aware receding-horizon command
→ fresh camera frame
→ replan
```

The policy uses:

```text
Octo diffusion head → continuous arm action
BCE gripper head   → gripper open/close probability
```

The final rollout setting used for the demo videos was:

```bash
--control_mode auto_receding_horizon \
--arm_chunk_index 1 \
--gripper_chunk_index 3
```

---

## 3. Required Hardware and Software

Deployment assumes:

```text
Kinova robot reachable over the network
Kinova Kortex Python API installed
Intel RealSense RGB camera connected locally
Octo installed and importable
JAX/Flax environment set up
Finetuned hybrid checkpoint available
Goal image available
```

See [SETUP.md](SETUP.md) for environment setup.

---

## 4. Required Files Before Deployment

Before running live inference, you need:

### 1. Finetuned checkpoint

Example:

```text
checkpoints/
  finetune_ckpts_hybrid_arm_diffusion_gripper_bce/
    best/
      checkpoint_*
```

### 2. Goal image

Example:

```text
goal_image.png
```

The goal image should represent the desired end condition for the task.

### 3. Dataset statistics

The deployment script can load dataset statistics from the TFDS build directory:

```bash
--tfds_data_dir /path/to/tensorflow_datasets
```

Alternatively, provide a statistics JSON directly:

```bash
--stats_json /path/to/dataset_statistics.json
```

### 4. Optional crop JSON files

Deployment crop:

```text
configs/deployment_crop.json
```

Goal-image crop:

```text
configs/goal_crop.json
```

These can be created through the startup crop review interface.

---

## 5. Goal Image

The deployment script loads a goal image from:

```bash
--goal_image_path /path/to/goal_image.png
```

The goal image is processed into:

```python
task["image_primary"]
```

This matches the goal-image-conditioned training setup.

If the goal image was captured from a wide camera view, it should be cropped consistently with the training images:

```bash
--goal_crop_json_path configs/goal_crop.json
```

or manually:

```bash
--goal_crop_box x_min y_min x_max y_max
```

---

## 6. Deployment Cropping

The model was trained on cropped images rather than the full RealSense camera view.

During data collection, the original camera frame captured a wide workspace, including background objects and other robots. Training images were cropped to focus on the active Kinova workspace and target object.

For this reason, the live inference crop should resemble the cropped training images as closely as possible.

Use:

```bash
--startup_crop_review
```

This opens a crop review window before inference starts. The window shows:

- the full camera frame;
- the selected crop box;
- the 256×256 model input preview.

Useful keys:

```text
A / Enter  accept crop
R          reselect or start crop
F          use full frame
Q / Esc    quit
```

Save and reuse a deployment crop:

```bash
--crop_json_path configs/deployment_crop.json
```

See [DATASET.md](DATASET.md) for more details on the cropping workflow.

---

## 7. Shadow Mode

Always run shadow mode first.

In shadow mode, the script performs inference and prints predicted commands, but does not send motion or gripper commands to the robot.

Example:

```bash
python scripts/deployment/run_finetuned_hybrid_model.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --goal_image_path /path/to/goal_image.png \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --show_window \
  --startup_crop_review \
  --control_mode auto_receding_horizon \
  --arm_chunk_index 1 \
  --gripper_chunk_index 3
```

If the crop, goal image, inference logs, and predicted motion look reasonable, then proceed to execution mode.

---

## 8. Execution Mode

To send commands to the Kinova robot, add:

```bash
--execute_actions
```

Example:

```bash
python scripts/deployment/run_finetuned_hybrid_model.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --goal_image_path /path/to/goal_image.png \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --show_window \
  --startup_crop_review \
  --execute_actions \
  --control_mode auto_receding_horizon \
  --arm_chunk_index 1 \
  --gripper_chunk_index 3
```

If your robot IP or login credentials differ from the defaults:

```bash
--ip <robot_ip> -u <username> -p <password>
```

---

## 9. Control Modes

The deployment script supports multiple control modes.

### `auto_receding_horizon`

Recommended mode for final deployment.

```bash
--control_mode auto_receding_horizon
```

This mode waits for a fresh camera frame, runs policy inference, executes selected action chunk(s), stops/settles briefly, waits for a new frame, and replans.

This was used for the real-robot demo rollouts.

### `timed_step`

Debugging mode.

```bash
--control_mode timed_step
```

This mode is useful for controlled timing experiments, but is not the recommended final deployment mode.

### `manual_step`

Debugging mode.

```bash
--control_mode manual_step
```

This mode waits for a keyboard trigger before each step. It is useful for inspecting predictions and robot behavior slowly.

By default, the trigger key is:

```bash
--manual_step_execute_key n
```

---

## 10. Action Chunk Selection

Octo predicts an action chunk with length:

```bash
--action_horizon 4
```

During deployment, the script selects which predicted arm and gripper chunk index to use.

Recommended setting used in the demos:

```bash
--arm_chunk_index 1
--gripper_chunk_index 3
```

### Arm chunk

```bash
--arm_chunk_index 1
```

This selects the arm action from index 1 of the predicted chunk.

### Gripper chunk

```bash
--gripper_chunk_index 3
```

This selects the gripper probability from index 3 of the predicted chunk.

In the final successful rollout settings, using a later gripper chunk helped the gripper close at a better stage of the approach/pick behavior.

---

## 11. Receding-Horizon Burst Execution

The deployment script can execute a short burst of consecutive arm actions before replanning.

Useful flags:

```bash
--arm_burst_len 2
--inter_pulse_gap_sec 0.0
```

`arm_burst_len` controls how many consecutive arm actions are executed from the predicted chunk before stopping and replanning.

Example:

```bash
--arm_chunk_index 1
--arm_burst_len 2
```

This executes action indices:

```text
1, 2
```

before stopping and requesting a fresh camera frame.

A shorter burst can be safer, slow, and more reactive. A longer burst can look smoother and faster, but may slightly increase the risk of executing stale actions. Demo video 1 uses a longer burst, which makes the motion faster than demo videos 2–6, which use a shorter burst.

---

## 12. Timing Parameters

Important timing flags:

```bash
--step_action_duration_sec 0.08
--settle_time_sec 0.01
--fresh_frame_timeout_sec 0.25
--min_loop_period_sec 0.0
```

### `step_action_duration_sec`

Controls how long each twist command is sent.

Smaller values make the robot more step-like and reactive. Larger values can make motion smoother but may increase latency/staleness.

### `settle_time_sec`

Small pause after stopping the arm before replanning.

### `fresh_frame_timeout_sec`

Maximum time to wait for a new camera frame.

### `min_loop_period_sec`

Optional minimum controller loop period.

---

## 13. Speed and Action Scaling

Default speed limits:

```bash
--max_linear_speed 0.08
--max_angular_speed_deg 20.0
```

To enable clamping:

```bash
--clamp_actions
```

Global action scaling:

```bash
--translation_scale 1.0
--rotation_scale 1.0
```

State-dependent scaling is also supported:

```bash
--close_translation_scale 1.0
--close_rotation_scale 1.0
--open_translation_scale 1.0
--open_rotation_scale 1.0
--z_extra_when_closed 0.0
```

These are useful for carefully tuning deployment behavior, but should be changed conservatively.

---

## 14. Gripper Control

The BCE gripper head outputs a probability for each action chunk index.

The deployment script uses the selected gripper probability:

```bash
--gripper_chunk_index 3
```

and converts it to a gripper state using hysteresis.

Default thresholds:

```bash
--close_threshold 0.40
--open_threshold 0.80
```

Interpretation:

```text
if currently open and probability <= close_threshold → close
if currently closed and probability >= open_threshold → open
otherwise → keep previous state
```

This helps reduce gripper chatter.

The final successful settings relied mainly on choosing the correct `gripper_chunk_index`; no hard lock-closed behavior is required for the reported demos.

---

## 15. Live Monitor Window

Use:

```bash
--show_window
```

The live monitor displays:

- full camera view with crop box;
- model input crop;
- predicted arm command;
- gripper probability/state;
- policy inference timing;
- frame age;
- control mode.

Press:

```text
Esc
```

to exit the live monitor.

---

## 16. Recommended Deployment Workflow

A safe deployment workflow is:

1. Connect the RealSense camera.
2. Confirm the Kinova robot is reachable.
3. Move the robot to a safe start pose.
4. Prepare the goal image.
5. Run shadow mode with `--startup_crop_review`.
6. Confirm the crop and model input preview.
7. Confirm inference logs look reasonable.
8. Run with `--execute_actions`.
9. Keep the emergency stop accessible.
10. Stop immediately if behavior becomes unsafe.

---

## 17. Known Deployment Behavior

The current deployed policy can sometimes pause for some seconds after reaching the bottle before closing the gripper. This is visible in some rollouts and remains an open deployment-timing improvement.

Future improvements may include:

- reducing the reach-to-grasp delay;
- improving smoother closed-loop motion;
- adding more long-horizon tasks;
- supporting additional robot wrappers.

---

## 18. Troubleshooting

### RealSense camera not found

Check:

```bash
python -c "import pyrealsense2 as rs; print('RealSense import OK')"
```

Also verify:

```bash
realsense-viewer
```

If the camera is busy, close other programs using it.

### Kortex connection fails

Check robot IP and credentials:

```bash
--ip <robot_ip> -u <username> -p <password>
```

Make sure the robot is reachable:

```bash
ping <robot_ip>
```

### Model input crop looks different from training images

Use:

```bash
--startup_crop_review
```

and adjust the crop until the 256×256 preview resembles the cropped training images.

Save the crop:

```bash
--crop_json_path configs/deployment_crop.json
```

### Gripper closes too early or too late

Try adjusting:

```bash
--gripper_chunk_index
--close_threshold
--open_threshold
```

For the reported demos, the best setting was:

```bash
--gripper_chunk_index 3
```

### Robot motion is too fast

Use action clamping and conservative speed limits:

```bash
--clamp_actions
--max_linear_speed 0.05
--max_angular_speed_deg 10.0
```

You can also reduce:

```bash
--translation_scale
--rotation_scale
```

### Robot motion is too step-like

The latency-aware controller intentionally executes short pulses and replans from fresh frames.

To make motion smoother, carefully experiment with:

```bash
--arm_burst_len
--step_action_duration_sec
--settle_time_sec
```

Increase these slowly and test in a safe workspace.

---

## 19. Example Final Command

Example final command used for real-robot rollout style:

```bash
python scripts/deployment/run_finetuned_hybrid_model.py \
  --checkpoint_path ./checkpoints/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best \
  --goal_image_path /path/to/goal_image.png \
  --tfds_data_dir /path/to/tensorflow_datasets \
  --show_window \
  --startup_crop_review \
  --execute_actions \
  --control_mode auto_receding_horizon \
  --arm_chunk_index 1 \
  --gripper_chunk_index 3 \
  --arm_burst_len 2 \
  --step_action_duration_sec 0.08 \
  --settle_time_sec 0.01
```

Start without `--execute_actions`, verify everything, and only then run the execution command.
