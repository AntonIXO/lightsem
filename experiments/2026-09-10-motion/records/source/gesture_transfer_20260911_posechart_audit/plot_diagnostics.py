import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
OUT=Path(__file__).resolve().parent
d=np.load(OUT/'synthetic_probes.npz')
colors=['#1965b0','#4eb265','#ca8eae','#dc050c','#f7cb45','#882e72']
labels=['R hand','R leg','R torso','L torso','L leg','L hand']
fig,ax=plt.subplots(3,2,figsize=(12,12),sharex=True,sharey=True)
for row,name in enumerate(['hands_fold_cross','legs_cross','torso_signed_bend']):
 for col,mapper in enumerate(['existing','candidate']):
  a=ax[row,col];v=d[name+'_'+mapper+'_chart']
  for k in range(6):
   a.plot(v[:,k,0],v[:,k,1],color=colors[k],alpha=.7,lw=1)
   a.scatter(v[0,k,0],v[0,k,1],color=colors[k],s=25,marker='o')
   a.scatter(v[-1,k,0],v[-1,k,1],color=colors[k],s=55,marker='x',label=labels[k])
  a.axhline(7,color='k',ls='--',lw=.6);a.set_title(name+' / '+mapper);a.grid(alpha=.2)
  a.set_xlim(-7.5,7.5);a.set_ylim(0,12.5);a.set_xlabel('surface chart x (m)');a.set_ylabel('unfolded u (m)')
ax[0,1].legend(fontsize=8,ncol=2)
fig.suptitle('Synthetic FK raw targets only: start circles, end crosses. Not rendered beam spread.')
fig.tight_layout();fig.savefig(OUT/'chart_trajectories.png',dpi=150)
fig,ax=plt.subplots(1,2,figsize=(10,5))
from lightsem.experiments.motion_study.geometry import PARENTS
for col,key in enumerate(['nullspace_pose_a','nullspace_pose_b']):
 b=d[key][0]
 for j in range(1,22):
  ids=[PARENTS[j],j];ax[col].plot(b[ids,2],b[ids,1],'k-',lw=1)
 ax[col].scatter(b[[18,20],2],b[[18,20],1],c=['orange','red'])
 ax[col].set_aspect('equal');ax[col].set_xlim(-.8,.8);ax[col].set_ylim(-.1,2);ax[col].set_title(key);ax[col].set_xlabel('world z (m)');ax[col].set_ylabel('world y (m)')
fig.suptitle('Different FK arm poses; exact same six targets in both mappers')
fig.tight_layout();fig.savefig(OUT/'fk_nullspace_counterexample.png',dpi=150)
