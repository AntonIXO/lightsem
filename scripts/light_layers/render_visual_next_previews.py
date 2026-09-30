#!/usr/bin/env python3
"""Render and authenticate the two frozen C palette-layout previews."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,time
from pathlib import Path
from lightsem.model.light_layers.visual import sha256,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path,required=True)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    plan=json.loads(Path(freeze['plan']).read_text());root=Path(plan['output_root']);manifest=copy.deepcopy(plan)
    def publish():
        path=root/'MANIFEST.json';temp=root/'MANIFEST.json.tmp'
        temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(path)
    for mapping in ('input_hashes','source_hashes'):
        for p,d in plan[mapping].items():assert sha256(p)==d,p
    started=time.monotonic();publish()
    env=dict(os.environ,TORCH_ENV='/root/lightsem/.venv/torch-access',OPENBLAS_NUM_THREADS='2')
    for row in manifest['variants']:
        if row['status']=='verified_existing':
            assert sha256(row['path'])==row['sha256'];continue
        assert sha256(row['streams'])==row['streams_sha256']
        output=Path(row['path']);output.parent.mkdir(parents=True,exist_ok=False)
        for mode in ('render','audit'):
            command=['bash','run_torch.sh','-u','-m','scripts.rendering.render_light_layers_review',mode,
                '--streams',row['streams'],'--output',str(output),'--label',row['caption'],
                '--audio',plan['source_audio'],'--max-bytes','50000000']
            with (output.parent/(mode+'.log')).open('xb') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=720,check=True)
        audit_path=output.with_name(output.stem+'_audit.json');audit=json.loads(audit_path.read_text())
        render=json.loads(output.with_name(output.stem+'_render.json').read_text())
        assert audit['decoded_frames']==8319 and audit['pts_uniform'] and audit['full_software_AV_decode']
        assert audit['pcm_sha256']==plan['expected_pcm_sha256']
        assert render['video_sha256']==sha256(output) and output.stat().st_size<=50000000
        row.update(status='rendered_and_verified',sha256=sha256(output),bytes=output.stat().st_size,audit=str(audit_path))
        publish();print('PREVIEW_VERIFIED',row['letter'],flush=True)
        if time.monotonic()-started>freeze['max_seconds']:raise RuntimeError('Preview delivery allowance')
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    manifest['status']='all_previews_rendered_and_verified';manifest['elapsed_s']=time.monotonic()-started;publish()
    write_json(root/'DELIVERY.json',{'status':manifest['status'],'elapsed_s':manifest['elapsed_s'],
        'manifest_sha256':sha256(root/'MANIFEST.json'),'freeze_sha256':sha256(args.freeze),
        'new_videos':2,'baseline_C_reused':True,'same_audio_verified':True,'new_training_claim':False})


if __name__=='__main__':main()
