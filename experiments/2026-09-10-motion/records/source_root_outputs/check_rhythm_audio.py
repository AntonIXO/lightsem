import subprocess,json,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate,resample_poly
out=Path('/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z');source,sr=sf.read('/root/shpongle.flac',dtype='float32',always_2d=True)
assert sr==44100
video=out/'full_candidate_1080p_av1.mp4'
raw=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(video),'-map','0:a:0','-vn','-ar',str(sr),'-ac','2','-f','f32le','-'])
decoded=np.frombuffer(raw,np.float32).reshape(-1,2)
rows=[]
for start in (0,70,190,265):
 lo=int(start*sr);hi=min(int((start+10)*sr),len(decoded),len(source))
 a=source[lo:hi].mean(1);b=decoded[lo:hi].mean(1)
 # Explicit +/-100ms candidate lag, no post-hoc waveform retiming.
 x=resample_poly(a,1,10);y=resample_poly(b,1,10)
 c=correlate(y-y.mean(),x-x.mean(),mode='full',method='fft');center=len(x)-1;radius=int(.1*sr/10)
 lag=(np.argmax(c[center-radius:center+radius+1])-radius)*10/sr
 centered_a=a-a.mean();centered_b=b-b.mean()
 similarity=float(np.dot(centered_a,centered_b)/max(np.linalg.norm(centered_a)*np.linalg.norm(centered_b),1e-12))
 mse=float(np.mean((a-b)**2));zero=float(np.mean(a*a));gain=float(np.dot(a,b)/max(np.dot(a,a),1e-12))
 rows.append({'start_s':start,'best_lag_s_within_100ms':float(lag),'normalized_waveform_similarity_at_zero_lag':similarity,
  'source_rms':float(np.sqrt(zero)),'decoded_rms':float(np.sqrt(np.mean(b*b))),'decoded_source_rmse':float(np.sqrt(mse)),
  'silence_baseline_rmse':float(np.sqrt(zero)),'linear_gain':gain})
report={'samples_source':len(source),'samples_decoded':len(decoded),'expected_video_samples':round(277.3*sr),'sample_rate':sr,
 'audio_pcm_sha256':hashlib.sha256(raw).hexdigest(),'windows':rows,
 'scope':'AAC lossy audio; source waveform alignment on four fixed windows, not listening or continuous musical acceptance. Full decode and streamPTS are checked separately.'}
(out/'source_audio_alignment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
