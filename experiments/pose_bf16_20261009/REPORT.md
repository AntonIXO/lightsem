# BF16 RTMPose deployment branch — 2026-10-09

Status: **real BF16 TensorRT pose engine executed; full production pipeline not promoted**.

## What ran

1. TensorRT 11.4.0.106 was installed with `uv` into an isolated target.
2. NVIDIA ModelOpt AutoCast converted the RTMPose-m ONNX graph to mixed BF16/FP32. It converted 251/254 pose nodes (98.82%) to BF16 and kept sensitive nodes in FP32.
3. TensorRT built `/root/lightsem/experiments/pose_bf16_20261009/rtmpose_m_autocast_bf16.plan` successfully.
4. The engine executed real RTMPose crops from four source recordings, using PTS-preserved frames and the existing FP32 detector boxes. Every output was finite.
5. A single-engine comparison measured FP32 TensorRT at a median 596 pose crops/s and the BF16 AutoCast TensorRT engine at a median 1114 pose crops/s: **1.87× faster for the pose stage**. This excludes detector and decode. BF16-vs-FP32 TensorRT pose differences were measured on the same crops:

| recording | source-frame rate | BF16 pose calls/s | finite fraction | mean keypoint difference |
|---|---:|---:|---:|---:|
| C'mon | 135.5 | 408.7 | 1.0 | 1.75 px |
| Jump In The Line | 204.2 | 418.7 | 1.0 | 2.22 px |
| Last Friday Night | 45.3 | 419.8 | 1.0 | 2.56 px |
| Silent control | 365.1 | 413.8 | 1.0 | 2.26 px |

The pose engine is therefore a real BF16 deployment result, not the earlier trivial Identity probe.

## Current limitation

The detector is not yet BF16 TensorRT-qualified. The YOLOX graph's TopK exceeds TensorRT's maximum for this static export, and the ONNX Runtime TensorRT EP binary requires `libnvinfer.so.10` while the installed TensorRT package provides `libnvinfer.so.11`. I did not symlink incompatible ABI versions. A full BF16 end-to-end arm therefore remains pending detector-engine repair; the current result is BF16 RTMPose pose-stage execution with FP32 detector boxes.

A direct all-initializer BF16 conversion produced NaN outputs and was rejected. The successful AutoCast graph is the valid branch because it preserves numerically sensitive nodes in FP32.

Artifacts:

- BF16 engine: `rtmpose_m_autocast_bf16.plan`
- Converted graph: `rtmpose_m_autocast_bf16.onnx`
- FP32 comparison engine: `rtmpose_m_fp32.plan`
- Source comparison: `compare_autocast_pose.py`
- Initial invalid conversion: `rtmpose_m_bf16.plan` (diagnostic only; NaN outputs)

Do not replace the production RTMPose cache or call this a qualified full-pipeline baseline until detector BF16/TensorRT compatibility and the normal masked admission audit pass.
