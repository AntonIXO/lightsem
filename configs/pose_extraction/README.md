# Pose extraction profiles

The default profile is `bf16`: TensorRT AutoCast BF16 RTMPose-m pose stage with
the FP32 CUDA YOLOX-m HumanArt detector. The existing FP32 ORT path remains an
explicit override:

```bash
RTMPOSE_POSE_PROFILE=fp32 bash scripts/data/run_rtmpose_system.sh \
  scripts/data/extract_rtmpose_native.py ...
```

The BF16 engine is external to the repository and must match the SHA256 in
`baseline.json`. BF16 is not silently replaced by FP32 when the engine or its
TensorRT runtime is unavailable; the command fails with the missing artifact.

The current BF16 receipt is pose-stage evidence. It does not automatically
qualify the full detector/tracker/masked-admission pipeline.
