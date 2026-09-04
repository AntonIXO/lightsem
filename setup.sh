#!/bin/bash
set -e
cd /root/lightsem
uv venv --python 3.12 .venv 2>&1 | tail -2
uv pip install --python .venv/bin/python -q \
  yt-dlp librosa mediapipe opencv-python-headless numpy scipy scikit-learn matplotlib pandas 2>&1 | tail -5
.venv/bin/python - <<'PYEOF'
import librosa, mediapipe, cv2, numpy, sklearn
print("librosa", librosa.__version__)
print("mediapipe", mediapipe.__version__)
print("cv2", cv2.__version__)
print("sklearn", sklearn.__version__)
PYEOF
.venv/bin/yt-dlp --version
echo SETUP_OK
