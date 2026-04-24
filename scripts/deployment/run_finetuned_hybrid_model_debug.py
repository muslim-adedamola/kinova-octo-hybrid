# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# import time
# import json
# import glob
# import argparse
# import threading
# from pathlib import Path
# from collections import deque
# from typing import Optional

# import cv2
# import jax
# import jax.numpy as jnp
# import numpy as np
# import pyrealsense2 as rs
# import tensorflow as tf
# tf.config.set_visible_devices([], "GPU")

# import flax.linen as nn
# from flax.training import checkpoints

# from octo.model.octo_model import OctoModel
# from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
# from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
# from kortex_api.autogen.messages import Base_pb2
# from utilities import DeviceConnection


# # -----------------------------------------------------------------------------
# # Hybrid BCE gripper head
# # -----------------------------------------------------------------------------
# class BinaryGripperHead(nn.Module):
#     action_horizon: int
#     hidden_dim: int = 256

#     @nn.compact
#     def __call__(self, emb):
#         # emb: [B, D]
#         x = nn.Dense(self.hidden_dim)(emb)
#         x = nn.gelu(x)
#         x = nn.Dense(self.hidden_dim)(x)
#         x = nn.gelu(x)
#         logits = nn.Dense(self.action_horizon)(x)  # [B, H]
#         return logits


# # -----------------------------------------------------------------------------
# # Live camera reader
# # -----------------------------------------------------------------------------
# class RealSenseLatestRGB:
#     def __init__(self, width=640, height=480, fps=30):
#         self.width = width
#         self.height = height
#         self.fps = fps

#         self.pipeline = rs.pipeline()
#         config = rs.config()
#         config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
#         self.pipeline.start(config)

#         self._lock = threading.Lock()
#         self._latest_img = None
#         self._latest_t = None
#         self._stop = False

#         self._thread = threading.Thread(target=self._run, daemon=True)
#         self._thread.start()

#     def _run(self):
#         while not self._stop:
#             try:
#                 frames = self.pipeline.wait_for_frames()
#                 color_frame = frames.get_color_frame()
#                 if not color_frame:
#                     continue
#                 img = np.asanyarray(color_frame.get_data())
#                 t = time.time()
#                 with self._lock:
#                     self._latest_img = img
#                     self._latest_t = t
#             except Exception:
#                 continue

#     def get_latest(self):
#         with self._lock:
#             if self._latest_img is None:
#                 return None, None
#             return self._latest_img.copy(), self._latest_t

#     def close(self):
#         self._stop = True
#         try:
#             self._thread.join(timeout=1.0)
#         except Exception:
#             pass
#         try:
#             self.pipeline.stop()
#         except Exception:
#             pass


# # -----------------------------------------------------------------------------
# # Crop helpers
# # -----------------------------------------------------------------------------
# def crop_bgr_image(image_bgr, crop_box):
#     x_min, y_min, x_max, y_max = crop_box
#     h, w = image_bgr.shape[:2]

#     x_min = max(0, min(int(x_min), w - 1))
#     x_max = max(x_min + 1, min(int(x_max), w))
#     y_min = max(0, min(int(y_min), h - 1))
#     y_max = max(y_min + 1, min(int(y_max), h))

#     return image_bgr[y_min:y_max, x_min:x_max]


# def draw_crop_box(image_bgr, crop_box, color=(0, 255, 0), thickness=2):
#     vis = image_bgr.copy()
#     if crop_box is not None:
#         x_min, y_min, x_max, y_max = crop_box
#         cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, thickness)
#     return vis


# def save_crop_box_json(path, crop_box):
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     data = {"crop_box": [int(v) for v in crop_box]}
#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)
#     print(f"[INFO] Saved crop box to: {path}")


# def load_crop_box_json(path):
#     with open(path, "r") as f:
#         data = json.load(f)
#     crop_box = data["crop_box"]
#     if len(crop_box) != 4:
#         raise ValueError("crop_box in JSON must have 4 values")
#     return tuple(int(v) for v in crop_box)


# def select_crop_interactively(image_bgr, window_name="Select ROI"):
#     temp = image_bgr.copy()
#     cv2.putText(
#         temp,
#         "Drag ROI, then ENTER/SPACE. Press C to cancel.",
#         (10, 25),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.65,
#         (0, 255, 0),
#         2,
#         cv2.LINE_AA,
#     )
#     roi = cv2.selectROI(window_name, temp, showCrosshair=True, fromCenter=False)
#     cv2.destroyWindow(window_name)

#     x, y, w, h = roi
#     if w <= 0 or h <= 0:
#         return None

#     return (int(x), int(y), int(x + w), int(y + h))


# def resize_for_display(image, max_w=960, max_h=720):
#     h, w = image.shape[:2]
#     scale = min(max_w / w, max_h / h, 1.0)
#     new_w = max(1, int(round(w * scale)))
#     new_h = max(1, int(round(h * scale)))
#     if scale == 1.0:
#         return image
#     return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# def make_preview_panel(full_frame_bgr, crop_box, preview_size=256):
#     full_vis = draw_crop_box(full_frame_bgr, crop_box, color=(0, 255, 0), thickness=2)
#     full_vis = full_vis.copy()

#     instructions = [
#         "A/ENTER: accept crop",
#         "R: reselect crop",
#         "F: use full frame",
#         "Q or ESC: quit",
#     ]
#     y0 = 28
#     for i, line in enumerate(instructions):
#         cv2.putText(
#             full_vis,
#             line,
#             (10, y0 + i * 26),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2,
#             cv2.LINE_AA,
#         )

#     full_vis = resize_for_display(full_vis, max_w=1000, max_h=720)

#     if crop_box is not None:
#         crop_vis = crop_bgr_image(full_frame_bgr, crop_box)
#     else:
#         crop_vis = full_frame_bgr.copy()

#     crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_BGR2RGB)
#     crop_vis = cv2.resize(crop_vis, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
#     crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR)

#     crop_canvas = np.zeros((max(full_vis.shape[0], preview_size + 80), preview_size, 3), dtype=np.uint8)
#     crop_canvas[:preview_size, :, :] = crop_vis
#     cv2.putText(
#         crop_canvas,
#         "Model input preview",
#         (10, preview_size + 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (0, 255, 0),
#         2,
#         cv2.LINE_AA,
#     )

#     H = max(full_vis.shape[0], crop_canvas.shape[0])
#     full_pad = np.zeros((H, full_vis.shape[1], 3), dtype=np.uint8)
#     full_pad[:full_vis.shape[0], :full_vis.shape[1]] = full_vis

#     crop_pad = np.zeros((H, crop_canvas.shape[1], 3), dtype=np.uint8)
#     crop_pad[:crop_canvas.shape[0], :crop_canvas.shape[1]] = crop_canvas

#     spacer = np.zeros((H, 20, 3), dtype=np.uint8)
#     panel = np.concatenate([full_pad, spacer, crop_pad], axis=1)
#     return panel


# def startup_crop_review_loop(camera, initial_crop_box=None, crop_json_path=None):
#     crop_box = initial_crop_box

#     print("[INFO] Entering startup crop review window...")
#     print("[INFO] Keys: A/Enter=accept, R=reselect, F=full frame, Q/Esc=quit")

#     while True:
#         frame, _ = camera.get_latest()
#         if frame is None:
#             time.sleep(0.03)
#             continue

#         panel = make_preview_panel(frame, crop_box, preview_size=256)
#         cv2.imshow("startup_crop_review", panel)
#         key = cv2.waitKey(30) & 0xFF

#         if key in [ord("a"), ord("A"), 13]:
#             if crop_box is not None:
#                 print(f"[INFO] Accepted crop box: {crop_box}")
#                 if crop_json_path is not None:
#                     save_crop_box_json(crop_json_path, crop_box)
#             else:
#                 print("[INFO] Accepted full-frame mode.")
#             cv2.destroyWindow("startup_crop_review")
#             return crop_box

#         elif key in [ord("r"), ord("R")]:
#             sel = select_crop_interactively(frame, window_name="Select deployment crop")
#             if sel is not None:
#                 crop_box = sel
#                 print(f"[INFO] Reselected crop box: {crop_box}")
#             else:
#                 print("[INFO] Crop selection cancelled. Keeping previous crop.")

#         elif key in [ord("f"), ord("F")]:
#             crop_box = None
#             print("[INFO] Using full frame.")

#         elif key in [ord("q"), ord("Q"), 27]:
#             cv2.destroyWindow("startup_crop_review")
#             raise KeyboardInterrupt("User quit during crop review.")


# # -----------------------------------------------------------------------------
# # Image preprocessing
# # -----------------------------------------------------------------------------
# def preprocess_bgr_to_rgb_256(image_bgr, crop_box=None):
#     if crop_box is not None:
#         image_bgr = crop_bgr_image(image_bgr, crop_box)
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)
#     return image_rgb.astype(np.uint8)


# # -----------------------------------------------------------------------------
# # Dataset statistics loading
# # -----------------------------------------------------------------------------
# def load_dataset_statistics(
#     stats_json=None,
#     tfds_data_dir=None,
#     dataset_name="kinova_dataset",
#     dataset_config="default",
#     dataset_version="0.1.5",
# ):
#     if stats_json is not None:
#         with open(stats_json, "r") as f:
#             return json.load(f)

#     if tfds_data_dir is None:
#         return None

#     pattern = os.path.join(
#         tfds_data_dir,
#         dataset_name,
#         dataset_config,
#         dataset_version,
#         "dataset_statistics_*.json",
#     )
#     matches = sorted(glob.glob(pattern))
#     if not matches:
#         return None

#     with open(matches[-1], "r") as f:
#         return json.load(f)


# def tree_lists_to_arrays(x):
#     if isinstance(x, dict):
#         return {k: tree_lists_to_arrays(v) for k, v in x.items()}
#     if isinstance(x, list):
#         return np.asarray(x, dtype=np.float32)
#     return x


# def extract_action_stats(model, dataset_stats=None):
#     candidates = []

#     if getattr(model, "dataset_statistics", None) is not None:
#         candidates.append(model.dataset_statistics)

#     if dataset_stats is not None:
#         candidates.append(dataset_stats)

#     for stats in candidates:
#         if not isinstance(stats, dict):
#             continue
#         if "action" in stats:
#             return stats["action"]
#         for _, value in stats.items():
#             if isinstance(value, dict) and "action" in value:
#                 return value["action"]

#     return None


# def extract_readout_tensor(x):
#     if hasattr(x, "shape") and len(x.shape) >= 3:
#         y = x
#     elif isinstance(x, dict):
#         preferred_keys = [
#             "readout_action",
#             "action_readout",
#             "readout",
#             "readouts",
#             "obs_primary",
#             "obs",
#         ]
#         y = None
#         for key in preferred_keys:
#             if key in x:
#                 candidate = x[key]
#                 if hasattr(candidate, "tokens"):
#                     y = candidate.tokens
#                     break
#                 if hasattr(candidate, "shape") and len(candidate.shape) >= 3:
#                     y = candidate
#                     break
#         if y is None:
#             raise TypeError(f"Could not extract readout tensor from dict keys={list(x.keys())}")
#     else:
#         if hasattr(x, "tokens"):
#             y = x.tokens
#         else:
#             raise TypeError(f"Could not extract readout tensor. type={type(x)}")

#     if len(y.shape) == 4:
#         y = y[:, :, 0, :]

#     if len(y.shape) != 3:
#         raise ValueError(f"Expected readout tensor with 3 dims after normalization, got shape={y.shape}")

#     return y


# # -----------------------------------------------------------------------------
# # Task builder
# # -----------------------------------------------------------------------------
# def build_goal_only_task(goal_image_path, goal_crop_box=None):
#     goal_bgr = cv2.imread(goal_image_path)
#     if goal_bgr is None:
#         raise FileNotFoundError(f"Could not read goal image: {goal_image_path}")

#     goal_rgb = preprocess_bgr_to_rgb_256(goal_bgr, crop_box=goal_crop_box)

#     task = {
#         "image_primary": goal_rgb[None],  # [1, 256, 256, 3]
#         "pad_mask_dict": {
#             "image_primary": np.ones((1,), dtype=bool),
#         },
#     }
#     return task


# # -----------------------------------------------------------------------------
# # Robot command helpers
# # -----------------------------------------------------------------------------
# def clamp(x, lo, hi):
#     return max(lo, min(hi, x))


# def send_gripper_position(base, position):
#     cmd = Base_pb2.GripperCommand()
#     cmd.mode = Base_pb2.GRIPPER_POSITION
#     finger = cmd.gripper.finger.add()
#     finger.finger_identifier = 1
#     finger.value = float(position)
#     base.SendGripperCommand(cmd)


# def open_gripper(base):
#     send_gripper_position(base, 0.0)


# def close_gripper(base, position=1.0):
#     send_gripper_position(base, position)


# def send_twist_for_one_cycle(
#     base,
#     arm_action_6d,
#     dt,
#     translation_scale=1.0,
#     rotation_scale=1.0,
#     max_linear_speed=0.08,
#     max_angular_speed_deg=20.0,
#     execute_actions=False,
#     clamp_actions=False,
#     pulse_duration_sec=None,
#     stop_after_pulse=False,
# ):
#     dx, dy, dz, dtx, dty, dtz = [float(x) for x in arm_action_6d.tolist()]

#     vx = (dx * translation_scale) / dt
#     vy = (dy * translation_scale) / dt
#     vz = (dz * translation_scale) / dt

#     wx_deg = np.rad2deg((dtx * rotation_scale) / dt)
#     wy_deg = np.rad2deg((dty * rotation_scale) / dt)
#     wz_deg = np.rad2deg((dtz * rotation_scale) / dt)

#     if clamp_actions:
#         vx = clamp(vx, -max_linear_speed, max_linear_speed)
#         vy = clamp(vy, -max_linear_speed, max_linear_speed)
#         vz = clamp(vz, -max_linear_speed, max_linear_speed)
#         wx_deg = clamp(wx_deg, -max_angular_speed_deg, max_angular_speed_deg)
#         wy_deg = clamp(wy_deg, -max_angular_speed_deg, max_angular_speed_deg)
#         wz_deg = clamp(wz_deg, -max_angular_speed_deg, max_angular_speed_deg)

#     out = np.array([vx, vy, vz, wx_deg, wy_deg, wz_deg], dtype=np.float32)

#     if not execute_actions:
#         print(
#             "[SHADOW] twist = "
#             f"vx={out[0]:+.4f} m/s, vy={out[1]:+.4f} m/s, vz={out[2]:+.4f} m/s, "
#             f"wx={out[3]:+.2f} deg/s, wy={out[4]:+.2f} deg/s, wz={out[5]:+.2f} deg/s"
#         )
#         return out

#     command = Base_pb2.TwistCommand()
#     command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
#     command.duration = 0
#     command.twist.linear_x = float(out[0])
#     command.twist.linear_y = float(out[1])
#     command.twist.linear_z = float(out[2])
#     command.twist.angular_x = float(out[3])
#     command.twist.angular_y = float(out[4])
#     command.twist.angular_z = float(out[5])

#     base.SendTwistCommand(command)

#     if pulse_duration_sec is not None and pulse_duration_sec > 0:
#         time.sleep(pulse_duration_sec)

#     if stop_after_pulse:
#         stop_robot_motion(base, sleep_sec=0.03)

#     return out


# def stop_robot_motion(base, sleep_sec=0.05):
#     try:
#         base.Stop()
#         if sleep_sec > 0:
#             time.sleep(sleep_sec)
#     except Exception:
#         pass


# def resolve_initial_crop_box(crop_json_path=None, crop_box=None):
#     if crop_json_path is not None and Path(crop_json_path).exists():
#         loaded = load_crop_box_json(crop_json_path)
#         print(f"[INFO] Loaded crop box from JSON: {loaded}")
#         return loaded

#     if crop_box is not None:
#         loaded = tuple(crop_box)
#         print(f"[INFO] Using crop box from CLI: {loaded}")
#         return loaded

#     return None


# # -----------------------------------------------------------------------------
# # Gripper hysteresis / latch
# # -----------------------------------------------------------------------------
# def state_name(s):
#     return "OPEN" if s == 1 else "CLOSE"


# def decide_gripper_state_with_hysteresis(
#     gripper_prob: float,
#     prev_state: Optional[int],
#     close_threshold: float,
#     open_threshold: float,
# ):
#     """
#     State:
#       1 = OPEN
#       0 = CLOSE
#     """
#     if prev_state is None:
#         return (1 if gripper_prob >= 0.5 else 0), "init"

#     if prev_state == 1:
#         if gripper_prob <= close_threshold:
#             return 0, "hysteresis_close"
#         return 1, "hold_open_band"
#     else:
#         if gripper_prob >= open_threshold:
#             return 1, "hysteresis_open"
#         return 0, "hold_close_band"


# def wait_for_manual_step(trigger_key="n"):
#     """
#     Wait until user presses:
#       trigger_key -> continue
#       q or ESC    -> quit
#     Requires an OpenCV window to be alive for key capture.
#     """
#     trigger_key = trigger_key.lower()
#     print(f"[STEP] Waiting for key '{trigger_key}' to run next step. Press 'q' or ESC to quit.")

#     while True:
#         key = cv2.waitKey(50) & 0xFF

#         if key == 255:
#             continue

#         if key in [ord(trigger_key), ord(trigger_key.upper())]:
#             return "continue"

#         if key in [ord("q"), ord("Q"), 27]:
#             return "quit"


# def clamp_chunk_index(idx, horizon):
#     return max(0, min(int(idx), int(horizon) - 1))



# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint_path", type=str, required=True,
#                         help="Hybrid checkpoint dir or checkpoint file.")
#     parser.add_argument("--pretrained_path", type=str, default="hf://rail-berkeley/octo-small-1.5")
#     parser.add_argument("--goal_image_path", type=str, required=True)

#     parser.add_argument("--tfds_data_dir", type=str, default=None)
#     parser.add_argument("--stats_json", type=str, default=None)
#     parser.add_argument("--dataset_name", type=str, default="kinova_dataset")
#     parser.add_argument("--dataset_config", type=str, default="default")
#     parser.add_argument("--dataset_version", type=str, default="0.1.5")

#     parser.add_argument("--window_size", type=int, default=2)
#     parser.add_argument("--action_horizon", type=int, default=4)
#     parser.add_argument("--gripper_head_hidden_dim", type=int, default=256)
#     parser.add_argument("--control_hz", type=float, default=10.0)

#     parser.add_argument("--camera_width", type=int, default=1280)
#     parser.add_argument("--camera_height", type=int, default=720)
#     parser.add_argument("--camera_fps", type=int, default=30)

#     parser.add_argument("--translation_scale", type=float, default=1.0)
#     parser.add_argument("--rotation_scale", type=float, default=1.0)
#     parser.add_argument("--max_linear_speed", type=float, default=0.08)
#     parser.add_argument("--max_angular_speed_deg", type=float, default=20.0)
#     parser.add_argument("--clamp_actions", action="store_true")

#     parser.add_argument("--execute_actions", action="store_true")
#     parser.add_argument("--gripper_sleep", type=float, default=0.05)

#     parser.add_argument("--show_window", action="store_true")
#     parser.add_argument("--debug_schema_once", action="store_true")

#     parser.add_argument("--crop_box", type=int, nargs=4, default=None)
#     parser.add_argument("--crop_json_path", type=str, default=None)
#     parser.add_argument("--startup_crop_review", action="store_true")

#     parser.add_argument("--goal_crop_box", type=int, nargs=4, default=None)
#     parser.add_argument("--goal_crop_json_path", type=str, default=None)

#     parser.add_argument("--close_threshold", type=float, default=0.40)
#     parser.add_argument("--open_threshold", type=float, default=0.80)
#     parser.add_argument("--close_hold_steps", type=int, default=12)
#     parser.add_argument("--reclose_hold_steps", type=int, default=8)

#     parser.add_argument("--chatter_window", type=int, default=20)
#     parser.add_argument("--chatter_transition_limit", type=int, default=3)

#     parser.add_argument(
#         "--lock_closed_after_first_close",
#         action="store_true",
#         help="Once the first close happens, keep the gripper closed until the run ends.",
#     )

#     parser.add_argument(
#         "--control_mode",
#         type=str,
#         default="continuous",
#         choices=["continuous", "timed_step", "manual_step"],
#         help="continuous: run at control_hz, timed_step: one predict/send then wait step_wait_sec, manual_step: wait for key after each step",
#     )
#     parser.add_argument(
#         "--step_wait_sec",
#         type=float,
#         default=3.0,
#         help="Wait time after each sent action in timed_step mode.",
#     )
#     parser.add_argument(
#         "--manual_step_execute_key",
#         type=str,
#         default="n",
#         help="Key to trigger next step in manual_step mode.",
#     )

#     parser.add_argument(
#         "--arm_chunk_index",
#         type=int,
#         default=0,
#         help="Which action chunk index to use for arm motion.",
#     )
#     parser.add_argument(
#         "--gripper_chunk_index",
#         type=int,
#         default=0,
#         help="Which action chunk index to use for gripper decision.",
#     )
#     parser.add_argument(
#     "--step_action_duration_sec",
#     type=float,
#     default=0.15,
#     help="How long to execute each twist command in timed/manual step mode before stopping.",
#     )

#     parser.add_argument("--ip", type=str, default="192.168.1.13")
#     parser.add_argument("-u", "--username", type=str, default="admin")
#     parser.add_argument("-p", "--password", type=str, default="admin")

#     args = parser.parse_args()
#     robot_args = args

#     if args.control_mode == "manual_step" and not args.show_window:
#         print("[INFO] Enabling --show_window automatically for manual_step mode.")
#         args.show_window = True

#     if not (0.0 <= args.close_threshold < args.open_threshold <= 1.0):
#         raise ValueError("Need 0 <= close_threshold < open_threshold <= 1")

#     print("[INFO] Loading hybrid checkpoint...")
#     ckpt = checkpoints.restore_checkpoint(args.checkpoint_path, target=None)
#     params = ckpt["params"]
#     ckpt_step = ckpt.get("step", None)
#     print(f"[INFO] Loaded checkpoint step: {ckpt_step}")

#     print("[INFO] Loading dataset statistics...")
#     dataset_stats = load_dataset_statistics(
#         stats_json=args.stats_json,
#         tfds_data_dir=args.tfds_data_dir,
#         dataset_name=args.dataset_name,
#         dataset_config=args.dataset_config,
#         dataset_version=args.dataset_version,
#     )
#     if dataset_stats is not None:
#         dataset_stats = tree_lists_to_arrays(dataset_stats)

#     # Build minimal example batch for hybrid Octo reconstruction
#     example_batch = {
#         "observation": {
#             "image_primary": np.zeros((1, args.window_size, 256, 256, 3), dtype=np.uint8),
#             "timestep": np.arange(args.window_size, dtype=np.int32)[None],
#             "task_completed": np.zeros((1, args.window_size, 4), dtype=bool),
#             "timestep_pad_mask": np.ones((1, args.window_size), dtype=bool),
#             "pad_mask_dict": {
#                 "image_primary": np.ones((1, args.window_size), dtype=bool),
#                 "timestep": np.ones((1, args.window_size), dtype=bool),
#             },
#         },
#         "task": {
#             "image_primary": np.zeros((1, 256, 256, 3), dtype=np.uint8),
#             "pad_mask_dict": {
#                 "image_primary": np.ones((1,), dtype=bool),
#             },
#         },
#         "action": np.zeros((1, args.window_size, args.action_horizon, 7), dtype=np.float32),
#         "action_pad_mask": np.ones((1, args.window_size, args.action_horizon, 7), dtype=bool),
#     }

#     print("[INFO] Loading Octo model structure...")
#     pretrained = OctoModel.load_pretrained(args.pretrained_path)
#     model = OctoModel.from_config(
#         pretrained.config,
#         example_batch,
#         text_processor=None,
#         verbose=False,
#         dataset_statistics=dataset_stats,
#     )
#     model = model.replace(params=params["octo"])

#     action_stats = extract_action_stats(model, dataset_stats)
#     if action_stats is None:
#         print("[WARN] No action statistics found. Continuous predictions may remain normalized.")
#     else:
#         action_stats = tree_lists_to_arrays(action_stats)

#     gripper_head = BinaryGripperHead(
#         action_horizon=args.action_horizon,
#         hidden_dim=args.gripper_head_hidden_dim,
#     )

#     @jax.jit
#     def predict_step(octo_params, gripper_params, obs, task, rng):
#         bound = model.module.bind({"params": octo_params}, rngs={"dropout": rng})

#         emb = bound.octo_transformer(
#             obs,
#             task,
#             obs["timestep_pad_mask"],
#             train=False,
#         )

#         cont_actions = model.sample_actions(
#             obs,
#             task,
#             unnormalization_statistics=action_stats,
#             rng=rng,
#         )[..., :6]

#         cont_actions = jnp.asarray(cont_actions)
#         if cont_actions.ndim == 4:
#             cont_actions = cont_actions[:, -1, :, :]  # [B, H, 6]
#         elif cont_actions.ndim != 3:
#             raise ValueError(f"Unexpected continuous action shape: {cont_actions.shape}")

#         readout_emb = extract_readout_tensor(emb)   # [B, W, D]
#         readout_emb = readout_emb[:, -1, :]         # [B, D]

#         gripper_logits = gripper_head.apply(
#             {"params": gripper_params},
#             readout_emb,
#         )                                           # [B, H]

#         gripper_prob = jax.nn.sigmoid(gripper_logits)[..., None]  # [B, H, 1]
#         actions = jnp.concatenate([cont_actions, gripper_prob], axis=-1)  # [B, H, 7]
#         return actions, gripper_prob.squeeze(-1)

#     deployment_crop_box = resolve_initial_crop_box(
#         crop_json_path=args.crop_json_path,
#         crop_box=args.crop_box,
#     )

#     goal_crop_box = resolve_initial_crop_box(
#         crop_json_path=args.goal_crop_json_path,
#         crop_box=args.goal_crop_box,
#     )

#     task = build_goal_only_task(
#         goal_image_path=args.goal_image_path,
#         goal_crop_box=goal_crop_box,
#     )

#     image_window = deque(maxlen=args.window_size)
#     recent_gripper_states = deque(maxlen=args.chatter_window)

#     rng = jax.random.PRNGKey(0)

#     print("[INFO] Running one warmup inference...")
#     warm_obs = jax.tree_util.tree_map(jnp.asarray, {
#         "image_primary": np.zeros((1, args.window_size, 256, 256, 3), dtype=np.uint8),
#         "timestep": np.arange(args.window_size, dtype=np.int32)[None],
#         "task_completed": np.zeros((1, args.window_size, 4), dtype=bool),
#         "timestep_pad_mask": np.ones((1, args.window_size), dtype=bool),
#         "pad_mask_dict": {
#             "image_primary": np.ones((1, args.window_size), dtype=bool),
#             "timestep": np.ones((1, args.window_size), dtype=bool),
#         },
#     })
#     warm_task = jax.tree_util.tree_map(jnp.asarray, task)
#     rng, warm_rng = jax.random.split(rng)
#     _acts, _grip = predict_step(
#         params["octo"],
#         params["gripper_head"],
#         warm_obs,
#         warm_task,
#         warm_rng,
#     )
#     _acts = jax.device_get(_acts)
#     _grip = jax.device_get(_grip)
#     print("[INFO] Warmup done.")

#     camera = RealSenseLatestRGB(
#         width=args.camera_width,
#         height=args.camera_height,
#         fps=args.camera_fps,
#     )

#     print("[INFO] Warming up camera...")
#     time.sleep(0.5)
#     for _ in range(60):
#         img, _ = camera.get_latest()
#         if img is not None:
#             break
#         time.sleep(0.03)

#     if args.startup_crop_review:
#         deployment_crop_box = startup_crop_review_loop(
#             camera,
#             initial_crop_box=deployment_crop_box,
#             crop_json_path=args.crop_json_path,
#         )
#     else:
#         if deployment_crop_box is None:
#             print("[INFO] No deployment crop box provided. Using full frame.")
#         else:
#             print(f"[INFO] Final deployment crop box: {deployment_crop_box}")

#     with DeviceConnection.createTcpConnection(robot_args) as router_tcp, DeviceConnection.createUdpConnection(robot_args) as router_udp:
#         base = BaseClient(router_tcp)
#         base_cyclic = BaseCyclicClient(router_udp)

#         print("[INFO] Starting live inference loop. Press Ctrl+C to stop.")

#         last_executed_gripper_state = None
#         total_transitions = 0
#         schema_printed = False
#         dt_target = 1.0 / args.control_hz

#         hold_close_counter = 0
#         ever_closed_once = False

#         try:
#             while True:
#                 t0 = time.time()

#                 image_bgr, _ = camera.get_latest()
#                 if image_bgr is None:
#                     continue

#                 image_rgb = preprocess_bgr_to_rgb_256(image_bgr, crop_box=deployment_crop_box)
#                 image_window.append(image_rgb)

#                 while len(image_window) < args.window_size:
#                     image_window.appendleft(image_window[0])

#                 obs = {
#                     "image_primary": np.stack(image_window, axis=0)[None],
#                     "timestep": np.arange(args.window_size, dtype=np.int32)[None],
#                     "task_completed": np.zeros((1, args.window_size, 4), dtype=bool),
#                     "timestep_pad_mask": np.ones((1, args.window_size), dtype=bool),
#                     "pad_mask_dict": {
#                         "image_primary": np.ones((1, args.window_size), dtype=bool),
#                         "timestep": np.ones((1, args.window_size), dtype=bool),
#                     },
#                 }

#                 if args.debug_schema_once and not schema_printed:
#                     print("OBS KEYS:", list(obs.keys()))
#                     print("OBS PAD MASK KEYS:", list(obs["pad_mask_dict"].keys()))
#                     print("OBS image_primary shape:", obs["image_primary"].shape)
#                     print("TASK KEYS:", list(task.keys()))
#                     print("TASK image_primary shape:", task["image_primary"].shape)
#                     schema_printed = True

#                 obs_jax = jax.tree_util.tree_map(jnp.asarray, obs)
#                 task_jax = jax.tree_util.tree_map(jnp.asarray, task)

#                 rng, sample_rng = jax.random.split(rng)

#                 start_pred = time.time()
#                 action_chunk, gripper_chunk_probs = predict_step(
#                     params["octo"],
#                     params["gripper_head"],
#                     obs_jax,
#                     task_jax,
#                     sample_rng,
#                 )

#                 action_chunk = jax.device_get(action_chunk)                 # [1, H, 7]
#                 gripper_chunk_probs = jax.device_get(gripper_chunk_probs)  # [1, H]
#                 pred_time = time.time() - start_pred

#                 action_chunk = np.asarray(action_chunk)[0]
#                 gripper_chunk_probs = np.asarray(gripper_chunk_probs)[0] 

#                 print(f"[TIMING] policy_step={pred_time:.4f}s")

#                 horizon = action_chunk.shape[0]
#                 arm_chunk_idx = clamp_chunk_index(args.arm_chunk_index, horizon)
#                 gripper_chunk_idx = clamp_chunk_index(args.gripper_chunk_index, horizon)

#                 arm_action_6d = action_chunk[arm_chunk_idx, :6].copy()
#                 gripper_prob = float(gripper_chunk_probs[gripper_chunk_idx])
                
#                 print("[GRIPPER_CHUNK]", np.round(gripper_chunk_probs, 4).tolist())
#                 print(
#                     f"[ACTION] arm_chunk_idx={arm_chunk_idx} "
#                     f"gripper_chunk_idx={gripper_chunk_idx} "
#                     f"arm={arm_action_6d.tolist()} "
#                     f"gprob={gripper_prob:.3f}"
#                 )

#                 raw_state, raw_reason = decide_gripper_state_with_hysteresis(
#                     gripper_prob=gripper_prob,
#                     prev_state=last_executed_gripper_state,
#                     close_threshold=args.close_threshold,
#                     open_threshold=args.open_threshold,
#                 )

#                 final_state = raw_state
#                 final_reason = raw_reason

#                 if hold_close_counter > 0:
#                     final_state = 0
#                     final_reason = f"latched_close({hold_close_counter})"
#                     hold_close_counter -= 1
#                 elif last_executed_gripper_state is not None and last_executed_gripper_state == 1 and raw_state == 0:
#                     if not ever_closed_once:
#                         hold_close_counter = max(args.close_hold_steps - 1, 0)
#                         ever_closed_once = True
#                         final_reason = f"{raw_reason}+start_initial_close_latch"
#                     else:
#                         hold_close_counter = max(args.reclose_hold_steps - 1, 0)
#                         final_reason = f"{raw_reason}+start_reclose_latch"

#                 if final_state == 0:
#                     ever_closed_once = True

#                 if args.lock_closed_after_first_close and ever_closed_once:
#                     final_state = 0
#                     final_reason = "task_lock_closed_after_first_close"

#                 if args.control_mode == "continuous":
#                     command_dt = dt_target
#                     pulse_duration_sec = None
#                     stop_after_pulse = False
#                 else:
#                     command_dt = args.step_action_duration_sec
#                     pulse_duration_sec =args.step_action_duration_sec
#                     stop_after_pulse = True

#                 scaled_arm = send_twist_for_one_cycle(
#                     base,
#                     arm_action_6d,
#                     dt=command_dt,
#                     translation_scale=args.translation_scale,
#                     rotation_scale=args.rotation_scale,
#                     max_linear_speed=args.max_linear_speed,
#                     max_angular_speed_deg=args.max_angular_speed_deg,
#                     execute_actions=args.execute_actions,
#                     clamp_actions=args.clamp_actions,
#                     pulse_duration_sec=pulse_duration_sec,
#                     stop_after_pulse=stop_after_pulse,
#                 )

#                 recent_gripper_states.append(final_state)

#                 transitioned = (
#                     last_executed_gripper_state is not None
#                     and final_state != last_executed_gripper_state
#                 )

#                 if transitioned:
#                     total_transitions += 1

#                 if args.execute_actions:
#                     if last_executed_gripper_state is None or final_state != last_executed_gripper_state:
#                         if final_state == 1:
#                             open_gripper(base)
#                         else:
#                             close_gripper(base, 1.0)

#                         if args.gripper_sleep > 0:
#                             time.sleep(args.gripper_sleep)

#                 if len(recent_gripper_states) >= 2:
#                     recent_transitions = int(np.sum(
#                         np.asarray(list(recent_gripper_states))[1:] !=
#                         np.asarray(list(recent_gripper_states))[:-1]
#                     ))
#                 else:
#                     recent_transitions = 0

#                 if recent_transitions > args.chatter_transition_limit:
#                     print(
#                         f"[WARN] Possible gripper chatter: "
#                         f"{recent_transitions} flips in last {len(recent_gripper_states)} steps"
#                     )

#                 prev_name = "None" if last_executed_gripper_state is None else state_name(last_executed_gripper_state)
#                 final_name = state_name(final_state)
#                 raw_name = state_name(raw_state)

#                 print(
#                     f"[LIVE] "
#                     f"prob={gripper_prob:.3f} "
#                     f"raw={raw_name}({raw_reason}) "
#                     f"final={final_name}({final_reason}) "
#                     f"prev={prev_name} "
#                     f"transitioned={transitioned} "
#                     f"hold_close_counter={hold_close_counter} "
#                     f"total_transitions={total_transitions} "
#                     f"recent_transitions={recent_transitions}"
#                 )

#                 last_executed_gripper_state = final_state

#                 if args.show_window:
#                     full_vis = draw_crop_box(image_bgr, deployment_crop_box, color=(0, 255, 0), thickness=2)
#                     crop_vis = crop_bgr_image(image_bgr, deployment_crop_box) if deployment_crop_box is not None else image_bgr.copy()
#                     crop_vis = cv2.resize(crop_vis, (256, 256), interpolation=cv2.INTER_AREA)

#                     txt1 = (
#                         f"vx={scaled_arm[0]:+.4f} vy={scaled_arm[1]:+.4f} vz={scaled_arm[2]:+.4f} "
#                         f"wx={scaled_arm[3]:+.2f} wy={scaled_arm[4]:+.2f} wz={scaled_arm[5]:+.2f}"
#                     )
#                     txt2 = (
#                         f"g_prob={gripper_prob:.3f} raw={raw_name} final={final_name} "
#                         f"a_idx={arm_chunk_idx} g_idx={gripper_chunk_idx} "
#                         f"hold={hold_close_counter} trans={total_transitions}"
#                     )

#                     cv2.putText(full_vis, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
#                     cv2.putText(full_vis, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
#                     cv2.putText(crop_vis, "model input", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

#                     full_vis = resize_for_display(full_vis, max_w=1000, max_h=720)

#                     H = max(full_vis.shape[0], crop_vis.shape[0])
#                     full_pad = np.zeros((H, full_vis.shape[1], 3), dtype=np.uint8)
#                     crop_pad = np.zeros((H, crop_vis.shape[1], 3), dtype=np.uint8)
#                     full_pad[:full_vis.shape[0], :full_vis.shape[1]] = full_vis
#                     crop_pad[:crop_vis.shape[0], :crop_vis.shape[1]] = crop_vis
#                     spacer = np.zeros((H, 20, 3), dtype=np.uint8)
#                     panel = np.concatenate([full_pad, spacer, crop_pad], axis=1)

#                     cv2.imshow("octo_hybrid_live_monitor", panel)
#                     key = cv2.waitKey(1) & 0xFF
#                     if key == 27:
#                         break

#                 if args.control_mode == "continuous":
#                     elapsed = time.time() - t0
#                     sleep_t = dt_target - elapsed
#                     if sleep_t > 0:
#                         time.sleep(sleep_t)

#                 elif args.control_mode == "timed_step":
#                     print(f"[STEP] Waiting {args.step_wait_sec:.2f}s before next perception step...")
#                     time.sleep(args.step_wait_sec)
                
#                 elif args.control_mode == "manual_step":
#                     decision = wait_for_manual_step(args.manual_step_execute_key)
#                     if decision == "quit":
#                         break

#         finally:
#             try:
#                 if args.execute_actions:
#                     base.Stop()
#                     time.sleep(0.2)
#             except Exception:
#                 pass

#             camera.close()
#             cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()



# #python run_finetuned_hybrid_model_debug.py --checkpoint_path /home/aifors/run_finetuned_model_octo/finetune_ckpts_hybrid_arm_diffusion_gripper_bce/best/ --goal_image_path goal_image_f.png --tfds_data_dir /home/aifors/tensorflow_datasets/ --show_window --startup_crop_review --execute_actions --control_mode timed_step --arm_chunk_index 1 --gripper_chunk_index 3 --step_wait_sec 0.02




import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import json
import glob
import argparse
import threading
from pathlib import Path
from collections import deque
from typing import Optional

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pyrealsense2 as rs
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import flax.linen as nn
from flax.training import checkpoints

from octo.model.octo_model import OctoModel
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from utilities import DeviceConnection


# -----------------------------------------------------------------------------
# Hybrid BCE gripper head
# -----------------------------------------------------------------------------
class BinaryGripperHead(nn.Module):
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


# -----------------------------------------------------------------------------
# Live camera reader
# -----------------------------------------------------------------------------
class RealSenseLatestRGB:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

        self._lock = threading.Lock()
        self._latest_img = None
        self._latest_t = None
        self._frame_counter = 0
        self._stop = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                img = np.asanyarray(color_frame.get_data())
                t = time.time()
                with self._lock:
                    self._latest_img = img
                    self._latest_t = t
                    self._frame_counter += 1
            except Exception:
                continue

    def get_latest(self):
        with self._lock:
            if self._latest_img is None:
                return None, None, None
            return self._latest_img.copy(), self._latest_t, self._frame_counter

    def wait_for_new_frame(self, last_frame_counter=None, timeout_sec=1.0, poll_sec=0.005):
        start = time.time()
        while True:
            img, t, fc = self.get_latest()
            if img is not None:
                if last_frame_counter is None or fc != last_frame_counter:
                    return img, t, fc, True
            if time.time() - start > timeout_sec:
                return img, t, fc, False
            time.sleep(poll_sec)

    def close(self):
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Crop helpers
# -----------------------------------------------------------------------------
def crop_bgr_image(image_bgr, crop_box):
    x_min, y_min, x_max, y_max = crop_box
    h, w = image_bgr.shape[:2]

    x_min = max(0, min(int(x_min), w - 1))
    x_max = max(x_min + 1, min(int(x_max), w))
    y_min = max(0, min(int(y_min), h - 1))
    y_max = max(y_min + 1, min(int(y_max), h))

    return image_bgr[y_min:y_max, x_min:x_max]


def draw_crop_box(image_bgr, crop_box, color=(0, 255, 0), thickness=2):
    vis = image_bgr.copy()
    if crop_box is not None:
        x_min, y_min, x_max, y_max = crop_box
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, thickness)
    return vis


def save_crop_box_json(path, crop_box):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"crop_box": [int(v) for v in crop_box]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved crop box to: {path}")


def load_crop_box_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    crop_box = data["crop_box"]
    if len(crop_box) != 4:
        raise ValueError("crop_box in JSON must have 4 values")
    return tuple(int(v) for v in crop_box)


def select_crop_interactively(image_bgr, window_name="Select ROI"):
    temp = image_bgr.copy()
    cv2.putText(
        temp,
        "Drag ROI, then ENTER/SPACE. Press C to cancel.",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    roi = cv2.selectROI(window_name, temp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None

    return (int(x), int(y), int(x + w), int(y + h))


def resize_for_display(image, max_w=960, max_h=720):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if scale == 1.0:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def make_preview_panel(full_frame_bgr, crop_box, preview_size=256):
    full_vis = draw_crop_box(full_frame_bgr, crop_box, color=(0, 255, 0), thickness=2)
    full_vis = full_vis.copy()

    instructions = [
        "A/ENTER: accept crop",
        "R: reselect crop",
        "F: use full frame",
        "Q or ESC: quit",
    ]
    y0 = 28
    for i, line in enumerate(instructions):
        cv2.putText(
            full_vis,
            line,
            (10, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    full_vis = resize_for_display(full_vis, max_w=1000, max_h=720)

    if crop_box is not None:
        crop_vis = crop_bgr_image(full_frame_bgr, crop_box)
    else:
        crop_vis = full_frame_bgr.copy()

    crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_BGR2RGB)
    crop_vis = cv2.resize(crop_vis, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
    crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR)

    crop_canvas = np.zeros((max(full_vis.shape[0], preview_size + 80), preview_size, 3), dtype=np.uint8)
    crop_canvas[:preview_size, :, :] = crop_vis
    cv2.putText(
        crop_canvas,
        "Model input preview",
        (10, preview_size + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    H = max(full_vis.shape[0], crop_canvas.shape[0])
    full_pad = np.zeros((H, full_vis.shape[1], 3), dtype=np.uint8)
    full_pad[:full_vis.shape[0], :full_vis.shape[1]] = full_vis

    crop_pad = np.zeros((H, crop_canvas.shape[1], 3), dtype=np.uint8)
    crop_pad[:crop_canvas.shape[0], :crop_canvas.shape[1]] = crop_canvas

    spacer = np.zeros((H, 20, 3), dtype=np.uint8)
    panel = np.concatenate([full_pad, spacer, crop_pad], axis=1)
    return panel


def startup_crop_review_loop(camera, initial_crop_box=None, crop_json_path=None):
    crop_box = initial_crop_box

    print("[INFO] Entering startup crop review window...")
    print("[INFO] Keys: A/Enter=accept, R=reselect, F=full frame, Q/Esc=quit")

    while True:
        frame, _, _ = camera.get_latest()
        if frame is None:
            time.sleep(0.03)
            continue

        panel = make_preview_panel(frame, crop_box, preview_size=256)
        cv2.imshow("startup_crop_review", panel)
        key = cv2.waitKey(30) & 0xFF

        if key in [ord("a"), ord("A"), 13]:
            if crop_box is not None:
                print(f"[INFO] Accepted crop box: {crop_box}")
                if crop_json_path is not None:
                    save_crop_box_json(crop_json_path, crop_box)
            else:
                print("[INFO] Accepted full-frame mode.")
            cv2.destroyWindow("startup_crop_review")
            return crop_box

        elif key in [ord("r"), ord("R")]:
            sel = select_crop_interactively(frame, window_name="Select deployment crop")
            if sel is not None:
                crop_box = sel
                print(f"[INFO] Reselected crop box: {crop_box}")
            else:
                print("[INFO] Crop selection cancelled. Keeping previous crop.")

        elif key in [ord("f"), ord("F")]:
            crop_box = None
            print("[INFO] Using full frame.")

        elif key in [ord("q"), ord("Q"), 27]:
            cv2.destroyWindow("startup_crop_review")
            raise KeyboardInterrupt("User quit during crop review.")


# -----------------------------------------------------------------------------
# Image preprocessing
# -----------------------------------------------------------------------------
def preprocess_bgr_to_rgb_256(image_bgr, crop_box=None):
    if crop_box is not None:
        image_bgr = crop_bgr_image(image_bgr, crop_box)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    return image_rgb.astype(np.uint8)


# -----------------------------------------------------------------------------
# Dataset statistics loading
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
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None

    with open(matches[-1], "r") as f:
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


# -----------------------------------------------------------------------------
# Task builder
# -----------------------------------------------------------------------------
def build_goal_only_task(goal_image_path, goal_crop_box=None):
    goal_bgr = cv2.imread(goal_image_path)
    if goal_bgr is None:
        raise FileNotFoundError(f"Could not read goal image: {goal_image_path}")

    goal_rgb = preprocess_bgr_to_rgb_256(goal_bgr, crop_box=goal_crop_box)

    task = {
        "image_primary": goal_rgb[None],
        "pad_mask_dict": {
            "image_primary": np.ones((1,), dtype=bool),
        },
    }
    return task


# -----------------------------------------------------------------------------
# Robot command helpers
# -----------------------------------------------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def send_gripper_position(base, position):
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    finger = cmd.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = float(position)
    base.SendGripperCommand(cmd)


def open_gripper(base):
    send_gripper_position(base, 0.0)


def close_gripper(base, position=1.0):
    send_gripper_position(base, position)


def stop_robot_motion(base, sleep_sec=0.05):
    try:
        base.Stop()
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    except Exception:
        pass


def scale_arm_action_for_state(
    arm_action_6d,
    gripper_state,
    close_translation_scale=1.0,
    close_rotation_scale=1.0,
    open_translation_scale=1.0,
    open_rotation_scale=1.0,
    z_extra_when_closed=0.0,
):
    arm = np.array(arm_action_6d, dtype=np.float32).copy()

    if gripper_state == 0:
        arm[:3] *= float(close_translation_scale)
        arm[3:] *= float(close_rotation_scale)
        arm[2] += float(z_extra_when_closed)
    else:
        arm[:3] *= float(open_translation_scale)
        arm[3:] *= float(open_rotation_scale)

    return arm


def send_twist_for_one_cycle(
    base,
    arm_action_6d,
    dt,
    translation_scale=1.0,
    rotation_scale=1.0,
    max_linear_speed=0.08,
    max_angular_speed_deg=20.0,
    execute_actions=False,
    clamp_actions=False,
    pulse_duration_sec=None,
    stop_after_pulse=False,
):
    dx, dy, dz, dtx, dty, dtz = [float(x) for x in arm_action_6d.tolist()]

    vx = (dx * translation_scale) / dt
    vy = (dy * translation_scale) / dt
    vz = (dz * translation_scale) / dt

    wx_deg = np.rad2deg((dtx * rotation_scale) / dt)
    wy_deg = np.rad2deg((dty * rotation_scale) / dt)
    wz_deg = np.rad2deg((dtz * rotation_scale) / dt)

    if clamp_actions:
        vx = clamp(vx, -max_linear_speed, max_linear_speed)
        vy = clamp(vy, -max_linear_speed, max_linear_speed)
        vz = clamp(vz, -max_linear_speed, max_linear_speed)
        wx_deg = clamp(wx_deg, -max_angular_speed_deg, max_angular_speed_deg)
        wy_deg = clamp(wy_deg, -max_angular_speed_deg, max_angular_speed_deg)
        wz_deg = clamp(wz_deg, -max_angular_speed_deg, max_angular_speed_deg)

    out = np.array([vx, vy, vz, wx_deg, wy_deg, wz_deg], dtype=np.float32)

    if not execute_actions:
        print(
            "[SHADOW] twist = "
            f"vx={out[0]:+.4f} m/s, vy={out[1]:+.4f} m/s, vz={out[2]:+.4f} m/s, "
            f"wx={out[3]:+.2f} deg/s, wy={out[4]:+.2f} deg/s, wz={out[5]:+.2f} deg/s"
        )
        return out

    command = Base_pb2.TwistCommand()
    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    command.duration = 0
    command.twist.linear_x = float(out[0])
    command.twist.linear_y = float(out[1])
    command.twist.linear_z = float(out[2])
    command.twist.angular_x = float(out[3])
    command.twist.angular_y = float(out[4])
    command.twist.angular_z = float(out[5])

    base.SendTwistCommand(command)

    if pulse_duration_sec is not None and pulse_duration_sec > 0:
        time.sleep(pulse_duration_sec)

    if stop_after_pulse:
        stop_robot_motion(base, sleep_sec=0.03)

    return out


def resolve_initial_crop_box(crop_json_path=None, crop_box=None):
    if crop_json_path is not None and Path(crop_json_path).exists():
        loaded = load_crop_box_json(crop_json_path)
        print(f"[INFO] Loaded crop box from JSON: {loaded}")
        return loaded

    if crop_box is not None:
        loaded = tuple(crop_box)
        print(f"[INFO] Using crop box from CLI: {loaded}")
        return loaded

    return None


# -----------------------------------------------------------------------------
# Gripper hysteresis / latch
# -----------------------------------------------------------------------------
def state_name(s):
    return "OPEN" if s == 1 else "CLOSE"


def decide_gripper_state_with_hysteresis(
    gripper_prob: float,
    prev_state: Optional[int],
    close_threshold: float,
    open_threshold: float,
):
    """
    State:
      1 = OPEN
      0 = CLOSE
    """
    if prev_state is None:
        return (1 if gripper_prob >= 0.5 else 0), "init"

    if prev_state == 1:
        if gripper_prob <= close_threshold:
            return 0, "hysteresis_close"
        return 1, "hold_open_band"
    else:
        if gripper_prob >= open_threshold:
            return 1, "hysteresis_open"
        return 0, "hold_close_band"


def wait_for_manual_step(trigger_key="n"):
    trigger_key = trigger_key.lower()
    print(f"[STEP] Waiting for key '{trigger_key}' to run next step. Press 'q' or ESC to quit.")
    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == 255:
            continue
        if key in [ord(trigger_key), ord(trigger_key.upper())]:
            return "continue"
        if key in [ord("q"), ord("Q"), 27]:
            return "quit"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Hybrid checkpoint dir or checkpoint file.")
    parser.add_argument("--pretrained_path", type=str, default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--goal_image_path", type=str, required=True)

    parser.add_argument("--tfds_data_dir", type=str, default=None)
    parser.add_argument("--stats_json", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="kinova_dataset")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--dataset_version", type=str, default="0.1.5")

    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--gripper_head_hidden_dim", type=int, default=256)

    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--camera_fps", type=int, default=30)

    parser.add_argument("--translation_scale", type=float, default=1.0)
    parser.add_argument("--rotation_scale", type=float, default=1.0)
    parser.add_argument("--max_linear_speed", type=float, default=0.08)
    parser.add_argument("--max_angular_speed_deg", type=float, default=20.0)
    parser.add_argument("--clamp_actions", action="store_true")

    parser.add_argument("--execute_actions", action="store_true")
    parser.add_argument("--gripper_sleep", type=float, default=0.05)

    parser.add_argument("--show_window", action="store_true")
    parser.add_argument("--debug_schema_once", action="store_true")

    parser.add_argument("--crop_box", type=int, nargs=4, default=None)
    parser.add_argument("--crop_json_path", type=str, default=None)
    parser.add_argument("--startup_crop_review", action="store_true")

    parser.add_argument("--goal_crop_box", type=int, nargs=4, default=None)
    parser.add_argument("--goal_crop_json_path", type=str, default=None)

    parser.add_argument("--close_threshold", type=float, default=0.40)
    parser.add_argument("--open_threshold", type=float, default=0.80)
    parser.add_argument("--close_hold_steps", type=int, default=12)
    parser.add_argument("--reclose_hold_steps", type=int, default=8)

    parser.add_argument("--chatter_window", type=int, default=20)
    parser.add_argument("--chatter_transition_limit", type=int, default=3)

    parser.add_argument(
        "--lock_closed_after_first_close",
        action="store_true",
        help="Once the first close happens, keep the gripper closed until the run ends.",
    )

    parser.add_argument(
        "--control_mode",
        type=str,
        default="auto_receding_horizon",
        choices=["auto_receding_horizon", "timed_step", "manual_step"],
        help="auto_receding_horizon: automatic pulse->stop->settle->fresh-frame->replan. "
             "timed_step/manual_step retained for debugging.",
    )

    parser.add_argument(
        "--manual_step_execute_key",
        type=str,
        default="n",
        help="Key to trigger next step in manual_step mode.",
    )

    parser.add_argument(
        "--arm_chunk_index",
        type=int,
        default=1,
        help="Which arm action inside the predicted chunk to execute.",
    )
    parser.add_argument(
        "--gripper_chunk_index",
        type=int,
        default=3,
        help="Which gripper action inside the predicted chunk to use.",
    )

    parser.add_argument(
        "--step_action_duration_sec",
        type=float,
        default=0.12,
        help="How long to execute each twist pulse before stopping.",
    )
    parser.add_argument(
        "--settle_time_sec",
        type=float,
        default=0.03,
        help="Additional time after stopping the arm before replanning.",
    )
    parser.add_argument(
        "--fresh_frame_timeout_sec",
        type=float,
        default=0.25,
        help="Max time to wait for a fresh camera frame before replanning with latest available frame.",
    )
    parser.add_argument(
        "--min_loop_period_sec",
        type=float,
        default=0.0,
        help="Optional minimum overall controller loop period.",
    )
    parser.add_argument(
        "--skip_if_no_fresh_frame",
        action="store_true",
        help="If no fresh frame arrives in time, skip that control cycle instead of reusing latest frame.",
    )

    parser.add_argument(
        "--close_translation_scale",
        type=float,
        default=1.0,
        help="Extra multiplier on xyz deltas when gripper final state is CLOSE.",
    )
    parser.add_argument(
        "--close_rotation_scale",
        type=float,
        default=1.0,
        help="Extra multiplier on rot deltas when gripper final state is CLOSE.",
    )
    parser.add_argument(
        "--open_translation_scale",
        type=float,
        default=1.0,
        help="Extra multiplier on xyz deltas when gripper final state is OPEN.",
    )
    parser.add_argument(
        "--open_rotation_scale",
        type=float,
        default=1.0,
        help="Extra multiplier on rot deltas when gripper final state is OPEN.",
    )
    parser.add_argument(
        "--z_extra_when_closed",
        type=float,
        default=0.0,
        help="Additive z delta after scaling when gripper final state is CLOSE.",
    )

    parser.add_argument("--ip", type=str, default="192.168.1.13")
    parser.add_argument("-u", "--username", type=str, default="admin")
    parser.add_argument("-p", "--password", type=str, default="admin")

    args = parser.parse_args()
    robot_args = args

    if args.control_mode == "manual_step" and not args.show_window:
        print("[INFO] Enabling --show_window automatically for manual_step mode.")
        args.show_window = True

    if not (0.0 <= args.close_threshold < args.open_threshold <= 1.0):
        raise ValueError("Need 0 <= close_threshold < open_threshold <= 1")

    print("[INFO] Loading hybrid checkpoint...")
    ckpt = checkpoints.restore_checkpoint(args.checkpoint_path, target=None)
    params = ckpt["params"]
    ckpt_step = ckpt.get("step", None)
    print(f"[INFO] Loaded checkpoint step: {ckpt_step}")

    print("[INFO] Loading dataset statistics...")
    dataset_stats = load_dataset_statistics(
        stats_json=args.stats_json,
        tfds_data_dir=args.tfds_data_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_version=args.dataset_version,
    )
    if dataset_stats is not None:
        dataset_stats = tree_lists_to_arrays(dataset_stats)

    example_batch = {
        "observation": {
            "image_primary": np.zeros((1, args.window_size, 256, 256, 3), dtype=np.uint8),
            "timestep": np.arange(args.window_size, dtype=np.int32)[None],
            "task_completed": np.zeros((1, args.window_size, 4), dtype=bool),
            "timestep_pad_mask": np.ones((1, args.window_size), dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.ones((1, args.window_size), dtype=bool),
                "timestep": np.ones((1, args.window_size), dtype=bool),
            },
        },
        "task": {
            "image_primary": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "pad_mask_dict": {
                "image_primary": np.ones((1,), dtype=bool),
            },
        },
        "action": np.zeros((1, args.window_size, args.action_horizon, 7), dtype=np.float32),
        "action_pad_mask": np.ones((1, args.window_size, args.action_horizon, 7), dtype=bool),
    }

    print("[INFO] Loading Octo model structure...")
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
        )[..., :6]

        cont_actions = jnp.asarray(cont_actions)
        if cont_actions.ndim == 4:
            cont_actions = cont_actions[:, -1, :, :]
        elif cont_actions.ndim != 3:
            raise ValueError(f"Unexpected continuous action shape: {cont_actions.shape}")

        readout_emb = extract_readout_tensor(emb)
        readout_emb = readout_emb[:, -1, :]

        gripper_logits = gripper_head.apply(
            {"params": gripper_params},
            readout_emb,
        )

        gripper_prob = jax.nn.sigmoid(gripper_logits)[..., None]
        actions = jnp.concatenate([cont_actions, gripper_prob], axis=-1)
        return actions, gripper_prob.squeeze(-1)

    deployment_crop_box = resolve_initial_crop_box(
        crop_json_path=args.crop_json_path,
        crop_box=args.crop_box,
    )

    goal_crop_box = resolve_initial_crop_box(
        crop_json_path=args.goal_crop_json_path,
        crop_box=args.goal_crop_box,
    )

    task = build_goal_only_task(
        goal_image_path=args.goal_image_path,
        goal_crop_box=goal_crop_box,
    )

    image_window = deque(maxlen=args.window_size)
    recent_gripper_states = deque(maxlen=args.chatter_window)

    rng = jax.random.PRNGKey(0)

    print("[INFO] Running one warmup inference...")
    warm_obs = jax.tree_util.tree_map(jnp.asarray, {
        "image_primary": np.zeros((1, args.window_size, 256, 256, 3), dtype=np.uint8),
        "timestep": np.arange(args.window_size, dtype=np.int32)[None],
        "task_completed": np.zeros((1, args.window_size, 4), dtype=bool),
        "timestep_pad_mask": np.ones((1, args.window_size), dtype=bool),
        "pad_mask_dict": {
            "image_primary": np.ones((1, args.window_size), dtype=bool),
            "timestep": np.ones((1, args.window_size), dtype=bool),
        },
    })
    warm_task = jax.tree_util.tree_map(jnp.asarray, task)
    rng, warm_rng = jax.random.split(rng)
    _acts, _grip = predict_step(
        params["octo"],
        params["gripper_head"],
        warm_obs,
        warm_task,
        warm_rng,
    )
    _acts = jax.device_get(_acts)
    _grip = jax.device_get(_grip)
    print("[INFO] Warmup done.")

    camera = RealSenseLatestRGB(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )

    print("[INFO] Warming up camera...")
    time.sleep(0.5)
    last_frame_counter = None
    for _ in range(60):
        img, _, fc = camera.get_latest()
        if img is not None:
            last_frame_counter = fc
            break
        time.sleep(0.03)

    if args.startup_crop_review:
        deployment_crop_box = startup_crop_review_loop(
            camera,
            initial_crop_box=deployment_crop_box,
            crop_json_path=args.crop_json_path,
        )
    else:
        if deployment_crop_box is None:
            print("[INFO] No deployment crop box provided. Using full frame.")
        else:
            print(f"[INFO] Final deployment crop box: {deployment_crop_box}")

    with DeviceConnection.createTcpConnection(robot_args) as router_tcp, DeviceConnection.createUdpConnection(robot_args) as router_udp:
        base = BaseClient(router_tcp)
        base_cyclic = BaseCyclicClient(router_udp)

        print("[INFO] Starting live inference loop. Press Ctrl+C to stop.")

        last_executed_gripper_state = None
        total_transitions = 0
        schema_printed = False

        hold_close_counter = 0
        ever_closed_once = False

        last_camera_timestamp_used = None
        controller_step = 0

        try:
            while True:
                loop_start = time.time()

                # -------------------------------------------------------------
                # Fresh-frame gating
                # -------------------------------------------------------------
                if args.control_mode == "auto_receding_horizon":
                    image_bgr, image_t, frame_counter, got_fresh = camera.wait_for_new_frame(
                        last_frame_counter=last_frame_counter,
                        timeout_sec=args.fresh_frame_timeout_sec,
                        poll_sec=0.005,
                    )
                    if image_bgr is None:
                        continue

                    if (not got_fresh) and args.skip_if_no_fresh_frame:
                        print("[WARN] No fresh frame arrived in time; skipping cycle.")
                        continue

                    last_frame_counter = frame_counter

                else:
                    image_bgr, image_t, frame_counter = camera.get_latest()
                    if image_bgr is None:
                        continue
                    last_frame_counter = frame_counter

                frame_age_before_policy = max(0.0, time.time() - image_t) if image_t is not None else float("nan")

                image_rgb = preprocess_bgr_to_rgb_256(image_bgr, crop_box=deployment_crop_box)
                image_window.append(image_rgb)
                while len(image_window) < args.window_size:
                    image_window.appendleft(image_window[0])

                obs = {
                    "image_primary": np.stack(image_window, axis=0)[None],
                    "timestep": np.arange(args.window_size, dtype=np.int32)[None],
                    "task_completed": np.zeros((1, args.window_size, 4), dtype=bool),
                    "timestep_pad_mask": np.ones((1, args.window_size), dtype=bool),
                    "pad_mask_dict": {
                        "image_primary": np.ones((1, args.window_size), dtype=bool),
                        "timestep": np.ones((1, args.window_size), dtype=bool),
                    },
                }

                if args.debug_schema_once and not schema_printed:
                    print("OBS KEYS:", list(obs.keys()))
                    print("OBS PAD MASK KEYS:", list(obs["pad_mask_dict"].keys()))
                    print("OBS image_primary shape:", obs["image_primary"].shape)
                    print("TASK KEYS:", list(task.keys()))
                    print("TASK image_primary shape:", task["image_primary"].shape)
                    schema_printed = True

                obs_jax = jax.tree_util.tree_map(jnp.asarray, obs)
                task_jax = jax.tree_util.tree_map(jnp.asarray, task)

                rng, sample_rng = jax.random.split(rng)

                start_pred = time.time()
                action_chunk, gripper_chunk_probs = predict_step(
                    params["octo"],
                    params["gripper_head"],
                    obs_jax,
                    task_jax,
                    sample_rng,
                )
                pred_time = time.time() - start_pred

                action_chunk = np.asarray(jax.device_get(action_chunk))[0]
                gripper_chunk_probs = np.asarray(jax.device_get(gripper_chunk_probs))[0]

                arm_chunk_idx = max(0, min(args.arm_chunk_index, action_chunk.shape[0] - 1))
                gripper_chunk_idx = max(0, min(args.gripper_chunk_index, action_chunk.shape[0] - 1))

                arm_action_6d = action_chunk[arm_chunk_idx][:6].copy()
                gripper_prob = float(gripper_chunk_probs[gripper_chunk_idx])

                print(f"[TIMING] policy_step={pred_time:.4f}s frame_age_before_policy={frame_age_before_policy:.4f}s")
                print("[GRIPPER_CHUNK]", np.round(gripper_chunk_probs, 4).tolist())
                print(
                    f"[ACTION] arm_chunk_idx={arm_chunk_idx} gripper_chunk_idx={gripper_chunk_idx} "
                    f"arm={arm_action_6d.tolist()} gprob={gripper_prob:.3f}"
                )

                raw_state, raw_reason = decide_gripper_state_with_hysteresis(
                    gripper_prob=gripper_prob,
                    prev_state=last_executed_gripper_state,
                    close_threshold=args.close_threshold,
                    open_threshold=args.open_threshold,
                )

                final_state = raw_state
                final_reason = raw_reason

                if hold_close_counter > 0:
                    final_state = 0
                    final_reason = f"latched_close({hold_close_counter})"
                    hold_close_counter -= 1
                elif last_executed_gripper_state is not None and last_executed_gripper_state == 1 and raw_state == 0:
                    if not ever_closed_once:
                        hold_close_counter = max(args.close_hold_steps - 1, 0)
                        ever_closed_once = True
                        final_reason = f"{raw_reason}+start_initial_close_latch"
                    else:
                        hold_close_counter = max(args.reclose_hold_steps - 1, 0)
                        final_reason = f"{raw_reason}+start_reclose_latch"

                if final_state == 0:
                    ever_closed_once = True

                if args.lock_closed_after_first_close and ever_closed_once:
                    final_state = 0
                    final_reason = "task_lock_closed_after_first_close"

                arm_action_6d = scale_arm_action_for_state(
                    arm_action_6d,
                    gripper_state=final_state,
                    close_translation_scale=args.close_translation_scale,
                    close_rotation_scale=args.close_rotation_scale,
                    open_translation_scale=args.open_translation_scale,
                    open_rotation_scale=args.open_rotation_scale,
                    z_extra_when_closed=args.z_extra_when_closed,
                )

                # -------------------------------------------------------------
                # Execute as pulse -> stop -> settle
                # -------------------------------------------------------------
                scaled_arm = send_twist_for_one_cycle(
                    base,
                    arm_action_6d,
                    dt=args.step_action_duration_sec,
                    translation_scale=args.translation_scale,
                    rotation_scale=args.rotation_scale,
                    max_linear_speed=args.max_linear_speed,
                    max_angular_speed_deg=args.max_angular_speed_deg,
                    execute_actions=args.execute_actions,
                    clamp_actions=args.clamp_actions,
                    pulse_duration_sec=args.step_action_duration_sec,
                    stop_after_pulse=True,
                )

                if args.settle_time_sec > 0:
                    time.sleep(args.settle_time_sec)

                recent_gripper_states.append(final_state)

                transitioned = (
                    last_executed_gripper_state is not None
                    and final_state != last_executed_gripper_state
                )
                if transitioned:
                    total_transitions += 1

                if args.execute_actions:
                    if last_executed_gripper_state is None or final_state != last_executed_gripper_state:
                        if final_state == 1:
                            open_gripper(base)
                        else:
                            close_gripper(base, 1.0)

                        if args.gripper_sleep > 0:
                            time.sleep(args.gripper_sleep)

                if len(recent_gripper_states) >= 2:
                    recent_transitions = int(np.sum(
                        np.asarray(list(recent_gripper_states))[1:] !=
                        np.asarray(list(recent_gripper_states))[:-1]
                    ))
                else:
                    recent_transitions = 0

                if recent_transitions > args.chatter_transition_limit:
                    print(
                        f"[WARN] Possible gripper chatter: "
                        f"{recent_transitions} flips in last {len(recent_gripper_states)} steps"
                    )

                prev_name = "None" if last_executed_gripper_state is None else state_name(last_executed_gripper_state)
                final_name = state_name(final_state)
                raw_name = state_name(raw_state)

                print(
                    f"[LIVE] "
                    f"step={controller_step} "
                    f"prob={gripper_prob:.3f} "
                    f"raw={raw_name}({raw_reason}) "
                    f"final={final_name}({final_reason}) "
                    f"prev={prev_name} "
                    f"transitioned={transitioned} "
                    f"hold_close_counter={hold_close_counter} "
                    f"total_transitions={total_transitions} "
                    f"recent_transitions={recent_transitions}"
                )

                last_executed_gripper_state = final_state
                last_camera_timestamp_used = image_t
                controller_step += 1

                if args.show_window:
                    full_vis = draw_crop_box(image_bgr, deployment_crop_box, color=(0, 255, 0), thickness=2)
                    crop_vis = crop_bgr_image(image_bgr, deployment_crop_box) if deployment_crop_box is not None else image_bgr.copy()
                    crop_vis = cv2.resize(crop_vis, (256, 256), interpolation=cv2.INTER_AREA)

                    txt1 = (
                        f"vx={scaled_arm[0]:+.4f} vy={scaled_arm[1]:+.4f} vz={scaled_arm[2]:+.4f} "
                        f"wx={scaled_arm[3]:+.2f} wy={scaled_arm[4]:+.2f} wz={scaled_arm[5]:+.2f}"
                    )
                    txt2 = (
                        f"g_prob={gripper_prob:.3f} raw={raw_name} final={final_name} "
                        f"hold={hold_close_counter} trans={total_transitions}"
                    )
                    txt3 = (
                        f"pred={pred_time:.3f}s frame_age={frame_age_before_policy:.3f}s "
                        f"mode={args.control_mode}"
                    )

                    cv2.putText(full_vis, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(full_vis, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(full_vis, txt3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(crop_vis, "model input", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

                    full_vis = resize_for_display(full_vis, max_w=1000, max_h=720)

                    H = max(full_vis.shape[0], crop_vis.shape[0])
                    full_pad = np.zeros((H, full_vis.shape[1], 3), dtype=np.uint8)
                    crop_pad = np.zeros((H, crop_vis.shape[1], 3), dtype=np.uint8)
                    full_pad[:full_vis.shape[0], :full_vis.shape[1]] = full_vis
                    crop_pad[:crop_vis.shape[0], :crop_vis.shape[1]] = crop_vis
                    spacer = np.zeros((H, 20, 3), dtype=np.uint8)
                    panel = np.concatenate([full_pad, spacer, crop_pad], axis=1)

                    cv2.imshow("octo_hybrid_live_monitor", panel)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

                if args.control_mode == "timed_step":
                    # Legacy timed-step behavior for debugging
                    pass

                elif args.control_mode == "manual_step":
                    decision = wait_for_manual_step(args.manual_step_execute_key)
                    if decision == "quit":
                        break

                if args.min_loop_period_sec > 0:
                    elapsed = time.time() - loop_start
                    sleep_t = args.min_loop_period_sec - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

        finally:
            try:
                if args.execute_actions:
                    base.Stop()
                    time.sleep(0.2)
            except Exception:
                pass

            camera.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()