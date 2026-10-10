"""Render source frames with raw RTMPose detections and admission masks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import av, cv2, numpy as np

EDGES=((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
       (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def frames_at_pts(frames, requested, tolerance=1e-6):
    """Select by PTS, not decoded-frame number (the cache may use a stride)."""
    result = {}
    for frame in frames:
        if frame.time is None:
            raise ValueError('Missing source PTS in evidence renderer')
        for index, stamp in requested.items():
            if abs(float(frame.time)-stamp) <= tolerance:
                result[index] = (frame.to_ndarray(format='bgr24'),float(frame.time))
        if len(result) == len(requested):
            return result
    raise ValueError('Source frames do not match the requested cache PTS')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists(): raise FileExistsError(a.output)
    z=np.load(a.cache,allow_pickle=False); meta=json.loads(str(z['meta'])); source=Path(meta['source_path'])
    frame_index=z['frame_index'].astype(int); candidates=z['candidate'].astype(bool); ids=z['track_id'].astype(int); boxes=z['box']; keypoints=z['K']; times=z['t']
    rows_by={int(f):np.flatnonzero(frame_index==f) for f in range(len(times))}
    accepted=np.flatnonzero(ids>=0); accepted_frame=int(frame_index[accepted[len(accepted)//2]]) if len(accepted) else 0
    rejected_frames=[f for f in range(len(times)) if not np.any(candidates[rows_by[f]])]
    rejected_frame=int(rejected_frames[len(rejected_frames)//2]) if rejected_frames else accepted_frame
    wanted=[accepted_frame,rejected_frame]; got={}
    with av.open(str(source)) as container:
        stream=container.streams.video[0]
        got=frames_at_pts(container.decode(stream),{index:float(times[index]) for index in wanted})
    a.output.parent.mkdir(parents=True,exist_ok=True); panels=[]
    for f in wanted:
        image=got[f][0].copy(); rows=rows_by[f]
        for row in rows:
            b=boxes[row].astype(int); selected=bool(candidates[row]); tracked=ids[row]>=0
            color=(40,220,40) if tracked else ((0,190,220) if selected else (120,120,120))
            cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),color,2)
            k=keypoints[row]
            for i,j in EDGES:
                if np.isfinite(k[[i,j]]).all(): cv2.line(image,tuple(k[i].astype(int)),tuple(k[j].astype(int)),color,2)
        label=f'frame={f} PTS={times[f]:.3f}s '+('tracked candidate; admission pending' if f==accepted_frame else 'no candidate; missing observation')
        cv2.rectangle(image,(0,0),(image.shape[1],48),(0,0,0),-1); cv2.putText(image,label,(12,31),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        panels.append(image)
    canvas=np.concatenate(panels,axis=0); cv2.imwrite(str(a.output),canvas)
    report={'source_path':str(source),'source_sha256':meta['source_sha256'],'cache':str(a.cache),'cache_sha256':__import__('hashlib').file_digest(a.cache.open('rb'),'sha256').hexdigest(),'frames':{'tracked':accepted_frame,'missing':rejected_frame},'decoded_pts':{str(f):got[f][1] for f in wanted},'output':str(a.output),'output_sha256':__import__('hashlib').file_digest(a.output.open('rb'),'sha256').hexdigest()}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
