import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightsem.experiments.motion_study.spatial_observer import project
r1=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_01');r2=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_02')
paths={'Rejected hybrid':r1/'old_hybrid_spatial.npz','Incumbent':r1/'old_incumbent_spatial.npz','Surface + old smoothing':r1/'articulated_surface_spatial.npz','Surface + recovered bandwidth':r2/'articulated_surface_spatial.npz'}
data={}
for name,path in paths.items():
 with np.load(path) as z:data[name]={k:z[k] for k in z.files}
fig,axs=plt.subplots(4,1,figsize=(15,13),sharex=True)
for name,d in data.items():
 t=np.arange(len(d['hit']))/40
 axs[0].plot(t[::8],d['hit_x_span'][::8],lw=.65,label=name)
 axs[1].plot(d['grid_time'],d['overlap'],lw=.7,label=name)
 axs[2].plot(d['grid_time'],d['coverage'],lw=.7,label=name)
 axs[3].plot(t[::8],d['central'][::8].astype(int),lw=.6,label=name)
for ax,title in zip(axs,['First-surface x span (m)','Soft-cone support overlap fraction (5Hz grid)','Surface support area (m2; NOT luminance)','4+ axes concentrated at central stage (40Hz, display5Hz)']):
 ax.set_ylabel(title);ax.grid(alpha=.2)
 for lo,hi in [(0,20),(60,76),(70,90),(108,118),(188,200),(250,277)]:ax.axvspan(lo,hi,color='gray',alpha=.06)
axs[0].axhline(4,color='black',ls='--',lw=.5);axs[0].legend(ncol=2);axs[-1].set_xlabel('Track time (s)')
fig.tight_layout();fig.savefig(r2/'spatial_timeline.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(2,4,figsize=(16,8))
for col,(name,d) in enumerate(data.items()):
 for row,(lo,hi) in enumerate([(70,90),(188,200)]):
  h=d['hit'][lo*40:hi*40];ax=axs[row,col]
  for i in range(6):
   ax.plot(h[::4,i,0],h[::4,i,2],lw=.8,label=f'head{i+1}');ax.scatter(h[0,i,0],h[0,i,2],s=15)
  ax.set_xlim(-8,8);ax.set_ylim(-6,2);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_title(name+f'\n{lo}-{hi}s');ax.set_xlabel('x (m)');ax.set_ylabel('z (m)')
fig.tight_layout();fig.savefig(r2/'surface_trajectories.png',dpi=140)
