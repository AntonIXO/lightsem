#!/usr/bin/env bash
set -euo pipefail

# RTMPose uses pacman Python, ONNX Runtime, CUDA, cuDNN, OpenCV and PyAV.
# Preload is necessary because this distro ORT provider omits cuDNN DT_NEEDED.
# Never mix pip-provided CUDA/cuDNN libraries into this process.
export PYTHONPATH="/mnt/lightsem-data/system_rtmpose_site:/root/lightsem"
export LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/lib"
export LD_PRELOAD="/usr/lib/libcudnn.so.9:/usr/lib/libcudnn_cnn.so.9:/usr/lib/libcudnn_ops.so.9:/usr/lib/libcudnn_adv.so.9:/usr/lib/libcudnn_graph.so.9:/usr/lib/libcudnn_heuristic.so.9:/usr/lib/libcudnn_engines_precompiled.so.9:/usr/lib/libcudnn_engines_runtime_compiled.so.9"
export TORCH_HOME=/mnt/lightsem-data/torch-home
export RTMPOSE_DET_MODEL=/mnt/lightsem-data/torch-home/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx
export RTMPOSE_POSE_PROFILE="${RTMPOSE_POSE_PROFILE:-bf16}"
if [[ "$RTMPOSE_POSE_PROFILE" == "fp32" ]]; then
  export RTMPOSE_POSE_MODEL="${RTMPOSE_POSE_MODEL:-/mnt/lightsem-data/torch-home/hub/checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx}"
  exec /usr/bin/python3 "$@"
fi

# BF16 pose uses the isolated Python 3.12 TensorRT runtime. The detector still
# runs through ORT CUDA; RTMPOSE_POSE_PROFILE=fp32 is the explicit legacy path.
export PYTHONPATH="/root/lightsem/.bf16_pydeps:/mnt/lightsem-data/tensorrt_env:/mnt/lightsem-data/onnxruntime_gpu_site:/mnt/lightsem-data/system_rtmpose_site:/root/lightsem"
export LD_LIBRARY_PATH="/mnt/lightsem-data/onnxruntime_gpu_site/onnxruntime/capi:/mnt/lightsem-data/tensorrt_env/tensorrt_libs:/opt/cuda/lib64:/usr/lib"
export RTMPOSE_BF16_ENGINE="${RTMPOSE_BF16_ENGINE:-/root/lightsem/experiments/pose_bf16_20261009/rtmpose_m_autocast_bf16.plan}"
exec /root/lightsem/.venv/bin/python "$@"
