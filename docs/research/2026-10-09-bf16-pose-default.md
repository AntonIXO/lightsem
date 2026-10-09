# BF16 RTMPose default and FP32 override

The default Just Dance pose profile is now `bf16`. It uses the authenticated
TensorRT AutoCast engine `rtmpose_m_autocast_bf16.plan` for the RTMPose-m pose
stage and keeps the HumanArt YOLOX-m detector on CUDA ONNX. The default engine
is external to git and is authenticated by
`configs/pose_extraction/baseline.json`.

The explicit legacy/reference override is:

```bash
RTMPOSE_POSE_PROFILE=fp32 bash scripts/data/run_rtmpose_system.sh \
  scripts/data/extract_rtmpose_native.py ...
```

The default BF16 wrapper was smoke-tested on a real source frame: four
detections, `(4, 17, 2)` finite keypoints, and finite scores. The BF16 pose
engine was compared with TensorRT FP32 on identical crops: median pose-stage
throughput was 1,114 versus 596 crop calls/s, with 1.75–2.56 px mean keypoint
drift. This is pose-stage evidence; full-corpus admission and detector/identity
qualification remain separate.
