#!/usr/bin/env bash
# Probe: what dance-video sources are reachable & legal to pull
set -u
echo "=== AIST Dance DB (research-licensed, the legal Just Dance equivalent) ==="
for u in \
  "https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/gBR_sBM_c01_d04_mBR0_ch01.mp4" \
  "https://aistdancedb.ongaaccel.jp/v1.0.0/video/60M/gBR_sBM_c01_d04_mBR0_ch01.mp4" \
  "https://aistdancedb.ongaaccel.jp/data/video/refined_10M_all_video_url.csv" \
  ; do
  code=$(curl -sL -o /dev/null -w '%{http_code} %{size_download}' --max-time 25 -r 0-1024 "$u" 2>/dev/null)
  echo "  [$code]  $u"
done
echo
echo "=== local AIST++ already on disk ==="
du -sh /root/lightsem/data/kp3d /root/lightsem/data/wav 2>/dev/null
ls /root/lightsem/data/wav/wav/ 2>/dev/null | head -3
ls /root/lightsem/data/kp3d/keypoints3d/ 2>/dev/null | head -3
