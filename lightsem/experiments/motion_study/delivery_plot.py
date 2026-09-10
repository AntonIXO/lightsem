"""Standalone scientific figures from saved full-track observations."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .delivery import OUT,ROOT
from ..stage11_common import read_authenticated,authenticated_json,digest
from ...conductor.rig import canonical_rig


def main():
    p=read_authenticated(OUT/'protocol.json');r=read_authenticated(OUT/'lighting/observations.json')
    videos=read_authenticated(OUT/'lighting/videos.json');v={x['name']:x for x in videos['rows']}
    with np.load(ROOT/'data/shpongle/loudness.npz') as z:loud={k:z[k] for k in z.files}
    palette={'incumbent':'#768492','hybrid':'#136f91','brightness_off':'#c66e39'}
    names={'incumbent':'Incumbent','hybrid':'Shared body + audio dynamics','brightness_off':'Same motion, brightness control off'}
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,layout='constrained')
    axes[0].plot(loud['time'],np.maximum(loud['momentary'],-60),color='#8692a0',lw=.8,label='Momentary (400 ms)')
    axes[0].plot(loud['time'],np.maximum(loud['short_term'],-60),color='#263645',lw=1.2,label='Short term (3 s)')
    axes[0].set_ylabel('Audio loudness\nLUFS');axes[0].set_ylim(-30,-3)
    for name in palette:
        with np.load(OUT/'lighting'/f'{name}.npz') as z:dmx=z['dmx']
        dim=[];angles=[]
        for f in canonical_rig().by_kind('moving_head'):
            a=f.address-1;dim.append(dmx[:,a+4]/255.)
            angles.append(np.column_stack(((dmx[:,a].astype(float)*256+dmx[:,a+1])*f.pan_range_deg/65535,
                                           (dmx[:,a+2].astype(float)*256+dmx[:,a+3])*f.tilt_range_deg/65535)))
        mean=np.mean(dim,axis=0);times=np.arange(len(mean))/40
        axes[1].plot(times,mean,color=palette[name],lw=1.,label=names[name])
        if name!='brightness_off':
            speed=np.linalg.norm(np.diff(np.stack(angles,1),axis=0)*40,axis=-1)
            n=len(speed)//40;med=np.median(speed[:n*40].reshape(n,40,6),axis=(1,2))
            axes[2].plot(np.arange(n)+.5,med,color=palette[name],lw=1.,label=names[name])
        obs=v[name]['observations']
        axes[3].plot([x['time'] for x in obs],[x['bulb_luma'] for x in obs],color=palette[name],lw=1.,label=names[name])
    axes[1].set_ylabel('Mean head dimmer\nemitted DMX / 255');axes[1].set_ylim(0,1)
    axes[2].set_ylabel('Median head speed\ndegrees / second')
    axes[3].set_ylabel('Bulb-region luminance\ndisplay RGB weighted mean');axes[3].set_xlabel('Track time (seconds)')
    for ax in axes:
        for k,(lo,hi) in p['segments'].items():ax.axvspan(lo,hi,color='#849d91',alpha=.10)
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',ncol=3,fontsize=8)
    axes[0].set_xlim(0,p['entries'][0]['duration'])
    fig.suptitle('Shpongle: separate audio, emitted movement and rendered brightness\nDiagnostic native model; music qualification failed',fontsize=14)
    files=[]
    for ext in ('png','pdf'):
        path=OUT/f'shpongle_dynamics.{ext}';fig.savefig(path,dpi=160);files.append(path)
    plt.close(fig)
    # All comparisons are retained in a machine-readable table as well.
    path=OUT/'lighting_summary.csv'
    import csv
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['variant','segment','speed_median_deg_s','speed_p95_deg_s','dimmer_median','bulb_luma','frame_luma','border_luma'])
        for variant,row in r['rows'].items():
            for seg,x in row['segments'].items():writer.writerow([variant,seg,x['speed_median'],x['speed_p95'],x['dimmer_median'],x['screen']['bulb_luma'],x['screen']['frame_luma'],x['screen']['border_luma']])
    files.append(path)
    authenticated_json(OUT/'figures.json',{'files':{str(p):digest(p) for p in files},'observations_sha256':digest(OUT/'lighting/observations.json'),'plot_source_sha256':digest(__file__)})


def comparison():
    """Labelled two-up movie; original renders and measurements stay intact."""
    import subprocess
    path=OUT/'shpongle_comparison.mp4'
    receipt=OUT/'shpongle_comparison.json'
    inputs=[OUT/'lighting'/f'{name}.mp4' for name in ('incumbent','hybrid')]
    if receipt.exists():
        r=read_authenticated(receipt)
        if digest(path)!=r['sha256'] or r['inputs']!={str(p):digest(p) for p in inputs}:raise ValueError('Comparison drift')
        return
    font='/usr/share/fonts/TTF/DejaVuSans.ttf'
    graph=(f'[0:v]scale=960:540,pad=960:580:0:40:black,drawtext=fontfile={font}:text=Incumbent:x=15:y=10:fontsize=21:fontcolor=white[a];'
           f'[1:v]scale=960:540,pad=960:580:0:40:black,drawtext=fontfile={font}:text=Experimental shared body + audio dynamics:x=15:y=10:fontsize=21:fontcolor=white[b];'
           '[a][b]hstack=inputs=2[v]')
    command=['ffmpeg','-v','error','-nostdin','-n','-i',str(inputs[0]),'-i',str(inputs[1]),'-filter_complex',graph,
             '-map','[v]','-map','1:a:0','-c:v','av1_nvenc','-preset','p5','-cq','25','-b:v','0','-pix_fmt','yuv420p','-c:a','copy','-shortest',str(path)]
    subprocess.run(command,check=True)
    authenticated_json(receipt,{'path':str(path),'sha256':digest(path),'inputs':{str(p):digest(p) for p in inputs},'command':command,
        'scope':'Labelled display comparison only; scores are measured on original full-resolution renders.'})


def correct_source_labels():
    """Preserve original reviews and correct the duplicated source-view heading."""
    import subprocess
    p=read_authenticated(OUT/'protocol.json');rows=[]
    for e in p['entries']:
        if 'source_video' not in e:continue
        original=OUT/'skeletons'/f'{e["key"]}.mp4'
        output=OUT/'skeletons/label_corrected'/original.name
        receipt=output.with_suffix('.json');output.parent.mkdir(parents=True,exist_ok=True)
        if receipt.exists():
            r=read_authenticated(receipt)
            if digest(output)!=r['sha256'] or digest(original)!=r['original_sha256']:raise ValueError('Label correction drift')
            rows.append(r);continue
        filters=[]
        for y in (0,480):
            filters += [f'drawbox=x=0:y={y}:w=640:h=58:color=white:t=fill',
                        f'drawtext=fontfile=/usr/share/fonts/TTF/DejaVuSans.ttf:text=Source recording - same frontal view:x=10:y={y+9}:fontsize=19:fontcolor=black']
        command=['ffmpeg','-v','error','-nostdin','-n','-i',str(original),'-vf',','.join(filters),'-map','0:v:0','-map','0:a:0',
                 '-c:v','av1_nvenc','-preset','p5','-cq','25','-b:v','0','-pix_fmt','yuv420p','-c:a','copy',str(output)]
        subprocess.run(command,check=True)
        r={'path':str(output),'sha256':digest(output),'original':str(original),'original_sha256':digest(original),'command':command,
           'correction':'Only the first-column headings are replaced. Source video is the same frontal recording in both rows; native panels show front and oblique projections. Original scientific arrays and reviews are retained.'}
        authenticated_json(receipt,r);rows.append(r)
    authenticated_json(OUT/'skeletons/label_corrections.json',{'rows':rows})


if __name__=='__main__':
    import sys
    if '--comparison' in sys.argv:comparison()
    elif '--correct-labels' in sys.argv:correct_source_labels()
    else:main()
