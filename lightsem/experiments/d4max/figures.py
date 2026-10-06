"""Static scientific figures from saved bodies and actual emitted DMX."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ..dynamic_dance_combo.campaign import read,save,sha
from ..dynamic_dance import diagnostics as d,transfer_audit as ta
from ..dynamic_dance.contracts import DanceResult,DancePlan,MusicTimeline
from lightsem.conductor.body_transfer import decode

LABELS={'native_180':'Native flow','D7_full':'D7 full','G78':'D7+D8','C3_combo':'D4+D7+D8','C4_combo':'D4+D6+D7+D8','D4_standard':'D4 on same D7','D4_max':'D4-max','D4_max_D6':'D4-max+D6',**{f'D{i}_180':f'D{i} (original)' for i in range(1,9)}}
COLORS={'D7_reference':'#818891','C3':'#1479b8','C4':'#d65e26'}
NAMES={'D7_reference':'D7 source','C3':'D4-max','C4':'D4-max + D6'}


def run(root):
 root=Path(root);plan=read(root/'PLAN.json');ranking=read(root/'reports/RANKING.json');out=root/'reports/figures_v2';out.mkdir(parents=True,exist_ok=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
 ids=ranking['all_metric_order'];records={r['id']:r for r in ranking['rows']};ys=np.arange(len(ids))
 fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
 for ax,track in zip(axes,plan['tracks']):
  song=track['id'];means=[records[i]['songs'][song]['mean_rank'] for i in ids];colors=['#1479b8' if records[i]['eligible'] else '#d6c5bf' for i in ids]
  ax.barh(ys,means,color=colors)
  for y,i,value in zip(ys,ids,means):ax.text(value+.12,y,f'{value:.2f}'+(' *' if not records[i]['songs'][song]['physical_eligible'] else ''),va='center',fontsize=9)
  ax.set_title(track['title']);ax.set_xlabel('Mean rank across 3 emitted-motion proxies (lower is better)');ax.set_xlim(0,len(ids)+1);ax.grid(axis='x',alpha=.2)
 axes[0].set_yticks(ys,[LABELS[i] for i in ids]);axes[0].invert_yaxis()
 winner=ranking['winner'];fig.suptitle('Observed output ranking · 60–180 s · winner with physical checks: '+(LABELS[winner] if winner else 'none'),fontsize=14)
 fig.text(.15,.015,'Blue: eligible on both songs. *: physical gate failed on this song. Automatic acoustic proxies; not an artistic verdict.\nOrder minimizes the worse-song rank, then overall mean. Older 180 s and full-song cohorts use different generated sources.',fontsize=9)
 fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(out/'ranking.png',dpi=150);fig.savefig(out/'ranking.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(18,8),sharey=True)
 for column,(ax,title) in enumerate(zip(axes,['Accent F1 − shifted null95','Recurrence Spearman − shifted null95','Release contrast − shifted null95'])):
  for si,(song,color) in enumerate([('shpongle','#1479b8'),('cure_for_me','#d65e26')]):
   values=[records[i]['songs'][song]['metrics'][column] for i in ids]
   ax.barh(ys+(-.18 if si==0 else .18),values,height=.32,color=color,label=song)
  ax.axvline(0,color='black',lw=.8);ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Higher is better')
 axes[0].set_yticks(ys,[LABELS[i] for i in ids]);axes[0].invert_yaxis();axes[-1].legend(loc='lower right');fig.suptitle('All saved variants · actual emitted light motion · matched 60–180 s windows');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'all_metrics.png',dpi=150);plt.close(fig)
 for track in plan['tracks']:
  song=track['id'];music=MusicTimeline.load(track['music_timeline']);traces={};bodydata={};metas={}
  for method in ['D7_reference','C3','C4']:
   body=DanceResult.load(root/'bodies'/method/song/'body.npz');bodydata[method]=body;metas[method]=body.metadata
   z=np.load(root/'timelines'/method/song/'timeline.npz');patch=read(root/'timelines'/method/song/'patch.json');c=decode(z['universes'],patch);mov=np.array([x['kind']!='bulb' for x in patch]);v=z['targets'][:,mov]-np.array([p['position'] for p in patch])[mov];length=np.linalg.norm(v,axis=-1)
   req=np.stack([np.rad2deg(np.arctan2(v[...,0],-v[...,1])),np.rad2deg(np.arcsin(np.clip(v[...,2]/length,-1,1)))],-1)
   traces[method]={'body':d.dynamics_series(body.world,body.time),'time':body.time,'ltime':z['time'].copy(),'requested':ta.motor_speed(req,z['time']).mean(1),'emitted':ta.motor_speed(c['angles_deg'][:,mov],z['time']).mean(1),'brightness':c['dimmer'][:,[p['kind']=='beam' for p in patch]].mean(1),'rate':z['playback_rate'].copy() if 'playback_rate' in z else np.ones(len(z['time']))}
   z.close()
  rows=[('root','Root\nm/s')]+[(name,name.replace('_',' ').title()+'\nm/s') for name in d.REGIONS]
  fig,axes=plt.subplots(len(rows)+2,1,figsize=(17,13),sharex=True)
  for ax,(key,label) in zip(axes,rows):
   for method,tr in traces.items():
    values=tr['body']['root_speed_m_s'] if key=='root' else tr['body']['regional_speed_m_s'][key]
    ax.plot(tr['time'],gaussian_filter1d(values,4.5),color=COLORS[method],lw=1.,alpha=.85,label=NAMES[method])
   ax.set_ylabel(label,fontsize=9);ax.grid(alpha=.2)
  for method,tr in traces.items():
   axes[-2].plot(tr['ltime'],gaussian_filter1d(tr['emitted'],6),color=COLORS[method],lw=1,label=NAMES[method]);axes[-1].plot(tr['ltime'],tr['brightness'],color=COLORS[method],lw=1)
  axes[-2].set_ylabel('Emitted motor\nspeed · deg/s');axes[-1].set_ylabel('Mean beam\ndimmer · 0–1');axes[-1].set_xlabel('Original song time · seconds')
  axes[0].legend(ncol=3,loc='upper right');axes[0].set_title(track['title']+' · D7 / D4-max / D4-max+D6 · limb speeds relative to pelvis; root speed in world')
  for ax in axes:
   for section in music.sections:ax.axvline(section['start'],color='#b2bdc6',lw=.6,alpha=.4)
   ax.set_xlim(0,track['duration_s'])
  fig.tight_layout();fig.savefig(out/f'{song}_movement.png',dpi=145);plt.close(fig)
  fig,axes=plt.subplots(4,1,figsize=(17,9),sharex=True)
  fragments=metas['C3']['fragment_provenance'];cmap=plt.get_cmap('tab10')
  for f in fragments:
   a,b=f['target_seconds'];axes[0].broken_barh([(a,b-a)],(f['motif_family']-.38,.76),facecolors=cmap(f['motif_family']))
   axes[1].plot([a,b],f['source_seconds'],color=cmap(f['motif_family']),lw=2)
  axes[0].set_ylabel('Musical family');axes[0].set_title(track['title']+' · recurring motif selection and actual playback rate')
  axes[1].plot([0,track['duration_s']],[0,track['duration_s']],ls='--',color='#999',label='Chronological source');axes[1].set_ylabel('D7 library time · s');axes[1].legend()
  for method in ['C3','C4']:
   tr=traces[method];rate=tr['rate'];rate=rate.mean(1) if rate.ndim==2 else rate
   axes[2].plot(tr['ltime'],rate,color=COLORS[method],label=NAMES[method],lw=1)
  axes[2].set_ylabel('Local playback rate');axes[2].set_ylim(-.05,1.6);axes[2].legend(ncol=2)
  body=bodydata['C4'];axes[3].plot(body.time,body.source_time-body.time,color=COLORS['C4'])
  anchors=np.array(metas['C3']['motif_anchor_frames'])/30
  axes[3].scatter(anchors,np.zeros(len(anchors)),s=9,color='black',zorder=3,label='Exact shared motif anchors')
  axes[3].set_ylabel('D6 clock offset\ninside D4-max · s');axes[3].set_xlabel('Original song time · seconds');axes[3].legend()
  for ax in axes:ax.set_xlim(0,track['duration_s']);ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(out/f'{song}_motifs.png',dpi=145);plt.close(fig)
  # Matched summary explicitly separates intended motion and real actuated output.
  fig,axes=plt.subplots(2,1,figsize=(17,7),sharex=True)
  for method,tr in traces.items():
   axes[0].plot(tr['ltime'],gaussian_filter1d(tr['requested'],6),label=NAMES[method],color=COLORS[method],lw=1)
   axes[1].plot(tr['ltime'],gaussian_filter1d(tr['emitted'],6),label=NAMES[method],color=COLORS[method],lw=1)
  axes[0].set_title(track['title']+' · requested versus actual constrained DMX motion');axes[0].set_ylabel('Requested · deg/s');axes[1].set_ylabel('Emitted · deg/s');axes[1].set_xlabel('Song time · s');axes[0].legend(ncol=3)
  for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,track['duration_s'])
  fig.tight_layout();fig.savefig(out/f'{song}_transfer.png',dpi=145);plt.close(fig)
 save(root/'reports/FIGURES_v2.json',{'files':{str(p):sha(p) for p in out.iterdir()},'smoothing':'Only plotted speed traces use Gaussian150ms; numeric metrics and source arrays unsmoothed.','inputs':{'ranking':sha(root/'reports/RANKING.json'),'plan':sha(root/'PLAN.json')}})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
