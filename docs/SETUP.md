# Setup Guide

This document describes the recommended setup for running the Kinova-Octo Hybrid pipeline.

The project assumes that you already have, or can create, a working Octo environment. This repository adds the Kinova-specific dataset builder, hybrid Octo+BCE finetuning scripts, offline evaluation scripts, and real-robot deployment wrapper.

---

## 1. Recommended System

This project was developed and tested in a Linux environment with:

```text
Ubuntu/Linux
Python 3.10
CUDA-enabled NVIDIA GPU for training
Intel RealSense RGB camera
Kinova manipulator with Kortex API access
```

The live deployment script assumes:

```text
Kinova robot reachable over the network
Intel RealSense camera connected locally
Kinova Kortex Python API installed
Octo installed and importable
```

---

## 2. Create a Conda Environment

Create a fresh environment:

```bash
conda create -n octo_kinova python=3.10 -y
conda activate octo_kinova
```

Upgrade basic packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

---

## 3. Install Octo

First, install Octo following the official Octo repository instructions.

Example workflow:

```bash
git clone https://github.com/octo-models/octo.git
cd octo
pip install -e .
pip install -r requirements.txt
```

After installation, check that Octo imports correctly:

```bash
python -c "from octo.model.octo_model import OctoModel; print('Octo import OK')"
```

This repository builds on Octo and expects the `octo` Python package to be importable from the active environment.

---

## 4. Install This Repository's Extra Dependencies

Clone this repository:

```bash
git clone https://github.com/muslim-adedamola/kinova-octo-hybrid.git
cd kinova-octo-hybrid
```

Install the additional dependencies:

```bash
pip install -r requirements.txt
```

The important extra dependencies for this project include:

```text
opencv-python
pillow
pandas
matplotlib
apache-beam
pyrealsense2
kortex-api
wandb
```
[Link to install Kortex API](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/tree/master/api_python/examples#install-kinova-kortex-python-api-and-required-dependencies)

If you get an error message related to protobuf, just install protobuf version 3.20.3

```bash
pip install protobuf==3.20.3
```

---

## 5. JAX / CUDA Notes

For training on GPU, install a CUDA-enabled JAX build that matches your system.

This project was tested with:

```text
jax==0.4.20
jaxlib==0.4.20+cuda11.cudnn86
```

Your exact `jaxlib` installation command may depend on your CUDA/cuDNN versions.

Check your JAX devices:

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

Expected output should include a GPU device for training.

If only CPU appears, reinstall the CUDA-compatible `jaxlib` version for your machine.

---

## 6. TensorFlow GPU Visibility

The training and deployment scripts use JAX for model computation and TensorFlow mainly for data loading.

Several scripts explicitly disable TensorFlow GPU visibility:

```python
tf.config.set_visible_devices([], "GPU")
```

This prevents TensorFlow from occupying GPU memory that should be used by JAX.

The scripts also set:

```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

to reduce aggressive JAX GPU memory preallocation.

---

## 7. Install Kinova Kortex API

The live deployment script requires the Kinova Kortex Python API.

If available through pip in your environment:

```bash
pip install kortex-api==2.6.0.post3
```

Verify the import:

```bash
python - <<'PY'
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
print("Kortex API import OK")
PY
```

The deployment script uses:

```text
BaseClient
BaseCyclicClient
Base_pb2
TCPTransport
UDPTransport
RouterClient
SessionManager
```

Make sure your robot is reachable over the network before deployment.

Default connection parameters in the scripts are:

```text
IP: 192.168.1.10 or 192.168.1.13
username: admin
password: admin
```

These should be changed using command-line flags if your robot uses different credentials:

```bash
--ip <robot_ip> -u <username> -p <password>
```

---

## 8. Install Intel RealSense Support

The live deployment script uses `pyrealsense2`.

Install:

```bash
pip install pyrealsense2
```

Verify the camera can be accessed:

```bash
python - <<'PY'
import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
frames = pipeline.wait_for_frames()
color = frames.get_color_frame()
print("RealSense color frame OK:", bool(color))
pipeline.stop()
PY
```

If this fails, check:

- the camera is connected;
- RealSense permissions/udev rules are configured;
- the camera is not already being used by another process;
- `realsense-viewer` can see the camera.

---

## 9. Add the Kinova Standardization Function to Octo

The training and evaluation scripts use:

```python
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo
```

Copy the standardization file into your local Octo source tree:

```bash
cp scripts/dataset/kinova_standardize_octo.py /path/to/octo/octo/data/kinova_standardize_octo.py
```

For example:

```bash
cp scripts/dataset/kinova_standardize_octo.py ~/octo/octo/data/kinova_standardize_octo.py
```

Then verify:

```bash
python - <<'PY'
from octo.data.kinova_standardize_octo import kinova_rlds_to_octo
print("Kinova standardization import OK")
PY
```

---

## 10. Prepare TFDS Data Directory

The raw Kinova episodes should be placed under the TensorFlow Datasets manual download directory.

By default, this repository's dataset builder looks for:

```text
<data_dir>/downloads/manual/kinova_dataset_2/episodes/
```

Example:

```text
/home/<user>/tensorflow_datasets/downloads/manual/kinova_dataset_2/episodes/
```

The name `kinova_dataset_2` is not a TensorFlow Datasets requirement. It is simply the folder name used for the dataset version in this project. If you want to use a different folder name, update the `root_dir` line inside:

```text
scripts/dataset/kinova_dataset/kinova_dataset_dataset_builder.py
```

Specifically, change:

```python
root_dir = os.path.join(manual_dir, "kinova_dataset_2", "episodes")
```

to match your own folder name.

See `docs/DATASET.md` for the expected raw episode structure.

Build the dataset:

```bash
tfds build scripts/dataset/kinova_dataset \
  --overwrite \
  --data_dir /path/to/tensorflow_datasets \
  --beam_pipeline_options="direct_running_mode=multi_threading,direct_num_workers=10"
```

Validate one episode:

```bash
python scripts/dataset/validate_one_episode.py \
  --data_dir /path/to/tensorflow_datasets \
  --config default \
  --split train
```

---

## 11. Weights & Biases

The training script logs to Weights & Biases by default.

Login:

```bash
wandb login
```

The default project name is:

```text
octo_kinova_goal_image_only
```

You can change it during training with:

```bash
--wandb_project <project_name>
--wandb_run_name <run_name>
```

If you do not want to sync logs online, set:

```bash
export WANDB_MODE=offline
```

---

## 12. Check the Repository Layout

After setup, the repository should contain:

```text
kinova-octo-hybrid/
├── scripts/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   └── deployment/
├── docs/
├── configs/
├── assets/
├── checkpoints/
└── data/
```

---

## 13. Smoke Tests

### Octo import

```bash
python -c "from octo.model.octo_model import OctoModel; print('Octo OK')"
```

### Kinova standardization import

```bash
python -c "from octo.data.kinova_standardize_octo import kinova_rlds_to_octo; print('Standardization OK')"
```

### Kortex import

```bash
python -c "from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient; print('Kortex OK')"
```

### RealSense import

```bash
python -c "import pyrealsense2 as rs; print('RealSense OK')"
```

### OpenCV import

```bash
python -c "import cv2; print('OpenCV OK')"
```

---

## 14. Additional Octo Setup and Multi-Node Notes

For advanced Octo setup notes, troubleshooting, and multi-node finetuning, the following external guide may be useful:

- [Octo Fine-tuning on Multiple Nodes — TRAIL, University of Tokyo](https://trail.t.u-tokyo.ac.jp/blog/24-12-19-octo-multinode/)

This guide discusses:

- installing Octo in a Conda environment;
- installing CUDA-enabled JAX;
- fixing the `scipy.linalg.tril` error by using `scipy<1.13`;
- resolving a JAX/cuDNN mismatch by installing the cuDNN version expected by `jaxlib`;
- initializing JAX distributed training for multi-node finetuning;
- handling WandB issues when multiple distributed processes start logging.

This repository was developed for single-node finetuning, but the TRAIL guide is a useful reference if you want to adapt the training script for multi-node JAX execution.

---

## 15. Next Steps

After setup:

1. Build the TFDS/RLDS dataset using `docs/DATASET.md`.
2. Train the hybrid model using `docs/TRAINING.md`.
3. Evaluate the trained checkpoint using the offline evaluation scripts.
4. Run live deployment first in shadow mode.
5. Use `--execute_actions` only after confirming crop, goal image, robot connection, and safety conditions.
