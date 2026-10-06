"""Static, exportable plots from saved native bodies and observed validation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from .data import ROOT,read,write,sha
from .showcase import ALIASES
from ..dynamic_dance.contracts import DanceResult
from ..dynamic_dance.diagnostics import dynamics_series

COLORS={'continued':'#8497aa','dynamics':'#1eafc4','joint':'#df6b2f'}
LABELS={'continued':'A · Continued training','dynamics':'B · Dynamics losses','joint':'C · Joint planner'}

def build(root=ROOT):
    root=Path(root);out=root/'reports/plots';out.mkdir(parents=True,exist_ok=False)
    plan=read(root/'PLAN.json');validation=read(root/'reports/VALIDATION.json');files={}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':160})
    for track in plan['tracks']:
        series={}
        for arm,alias in ALIASES.items():
            body=DanceResult.load(root/'bodies'/alias/track['id']/'body.npz');series[arm]=dynamics_series(body.world,body.time)
        regional=list(series['continued']['regional_speed_m_s']);fig,axes=plt.subplots(2+len(regional),1,figsize=(14,11),sharex=True,constrained_layout=True)
        keys=['root','whole']+regional
        for ax,key in zip(axes,keys):
            for arm,s in series.items():
                y=s['root_speed_m_s'] if key=='root' else s['world_speed_m_s'] if key=='whole' else s['regional_speed_m_s'][key]
                ax.plot(s['time'],gaussian_filter1d(y,7.5),color=COLORS[arm],label=LABELS[arm],lw=1.1)
            ax.set_ylabel(key.replace('_',' ')+'\nm/s');ax.grid(alpha=.18);ax.set_ylim(bottom=0)
        axes[0].legend(ncol=3,loc='upper right');axes[-1].set_xlabel('Original song time (seconds); no time warp')
        fig.suptitle(track['title']+' — actual generated movement\nRoot/world movement separate from root-relative regions; 250ms display smoothing only')
        p=out/(track['id']+'_regional_speed.png');fig.savefig(p);plt.close(fig);files[str(p)]=sha(p)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for seed,ax in enumerate(axes):
        names=[];scores=[]
        for arm in ALIASES:
            groups=validation['group_means'][f'{arm}_seed{seed}'];scores.append(np.mean([r['profile_error'] for r in groups.values()]));names.append(LABELS[arm])
        bars=ax.bar(range(3),scores,color=list(COLORS.values()));ax.set_xticks(range(3),['A continued','B dynamics','C joint']);ax.set_title(f'Training seed {seed}');ax.set_ylabel('Observed-cell speed-profile error (lower is better)');ax.bar_label(bars,fmt='%.4f');ax.set_ylim(0,max(scores)*1.2)
    fig.suptitle('Just Dance development recordings — descriptive paired comparison\n38/72 windows lack some region/4s observations; full registered primary is unqualified')
    p=out/'validation_error.png';fig.savefig(p);plt.close(fig);files[str(p)]=sha(p)
    # Deterministically first frozen JD validation song, middle window. No
    # inspection-based choice of an attractive interval or successful model.
    study=read(root/'STUDY_PLAN.json');w=next(w for w in study['evaluation']['windows'] if w['domain']=='jd' and w['window']==1)
    fig,axes=plt.subplots(6,1,figsize=(12,10),sharex=True,constrained_layout=True)
    for arm in ALIASES:
        with np.load(root/'evaluation_v5'/arm/'seed_0'/f"trace_{w['key']}.npz") as z:
            t=z['time'];values=z['profiles'][:,:,1];observed=z['observed_profiles'][:,:,1];mask=z['profile_valid'][:,:,1]
            for j,ax in enumerate(axes):
                if arm=='continued':ax.plot(t,np.where(mask[:,j],observed[:,j],np.nan),color='#28252b',lw=1.8,label='Recorded coach (2D observation)')
                ax.plot(t,np.where(mask[:,j],values[:,j],np.nan),color=COLORS[arm],lw=1.1,label=LABELS[arm]);ax.set_ylabel(['Lower body','Torso','Head','Left arm','Right arm','Whole articulation'][j]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('Original capture time (s)');fig.suptitle(f'Observed vs generated 1-second speed profiles · {w["key"]}\nTorso-lengths/s, canonical frontal projection; missing observations remain gaps')
    p=out/'recorded_vs_generated.png';fig.savefig(p);plt.close(fig);files[str(p)]=sha(p)
    write(root/'reports/PLOTS.json',{'files':files,'display_seed':0,'validation_example':w,'example_selection':'first frozen JD group, middle window; no result selection','smooth_plot_only':True})
    return files
