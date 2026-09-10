"""Scientific figures and their inspectable tabular data; CPU plotting only."""
from pathlib import Path
import argparse,csv,itertools,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest
from lightsem.experiments.night_common import ARMS,SEEDS
OUT=ROOT/'delivery/research'
BLUE='#285875';GRAY='#929292';DARK='#242424'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':170,'savefig.facecolor':'white'})
TABLES={}
FILE_OUTPUTS=[]

def savefig(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'));fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
    FILE_OUTPUTS.extend([OUT/(name+'.png'),OUT/(name+'.pdf')])

def table(name,rows):
    if not rows:raise ValueError('Empty table '+name)
    TABLES[name]=rows
    with (OUT/(name+'.csv')).open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    FILE_OUTPUTS.append(OUT/(name+'.csv'))

def diagnostics():
    OUT.mkdir(parents=True,exist_ok=True)
    pose=read_authenticated(ROOT/'diagnostics/supplementary_pose.json');rows=[]
    probes=['descriptor128','descriptor256','pose_code24','audio_base','audio_fourier','audio_muq']
    labels=['Descriptors\n128','Descriptors\n256','Pose code\n24','Audio\nBase','Audio\nFourier','Audio\nMuQ']
    for row in pose['rows']:
        for mode,error in row['primary_errors'].items():rows.append({'probe':row['name'],'seed':row['seed'],'canonical_id':row['canonical_id'],'mode':mode,
            **{k:error[k] for k in ('joint_error_tl','velocity_error_tl_s','root_error_tl','bone_length_error_tl','valid_joints','prediction_std','truth_std')}})
    table('pose_errors',rows)
    fig,ax=plt.subplots(figsize=(10,3.7));upper=1.
    for seed,col,dx in [(0,DARK,-.13),(1,BLUE,.13)]:
        stats=[pose['summary'][f'{p}/seed_{seed}']['full']['joint_error_tl'] for p in probes]
        means=np.array([s['mean'] for s in stats]);bounds=np.array([s['ci95'] for s in stats]).T
        upper=max(upper,float(bounds[1].max())*1.12)
        ax.errorbar(np.arange(6)+dx,means,yerr=np.maximum(np.array([means-bounds[0],bounds[1]-means]),0),fmt='o',color=col,capsize=4,label=f'Seed {seed}')
    mean=pose['summary']['descriptor256/seed_0']['train_mean']['joint_error_tl']['mean']
    ax.axhline(mean,ls=':',lw=1,color=GRAY,label='TRAIN mean pose')
    ax.set(xticks=range(6),xticklabels=labels,ylabel='Mean joint error (torso lengths)',title='Pose reconstruction and audio-only prediction, 40 VAL songs',ylim=(0,max(upper,mean*1.1)))
    ax.legend(frameon=False,ncol=3,loc='upper left');savefig(fig,'pose_reconstruction')
    time_rows=[]
    for r in pose['rows']:
        for mode,value in r['extra'].items():
            for seconds,error in value['smoothed_joint_error_tl'].items():time_rows.append({'probe':r['name'],'seed':r['seed'],'canonical_id':r['canonical_id'],'mode':mode,'seconds':float(seconds),'joint_error_tl':error})
    table('pose_timescales',time_rows)
    fig,ax=plt.subplots(figsize=(10,3.2));times=[.1,.5,2,8]
    for probe,label,color in [('descriptor256','Descriptors 256',DARK),('pose_code24','Pose-conditioned code 24',BLUE),('audio_muq','Audio MuQ',GRAY)]:
        values=[np.mean([r['joint_error_tl'] for r in time_rows if r['probe']==probe and r['mode']=='full' and r['seconds']==s]) for s in times]
        ax.plot(times,values,'o-',label=label,color=color)
    ax.set(xscale='log',yscale='log',xticks=times,xticklabels=[str(t) for t in times],xlabel='Moving-average duration (seconds)',ylabel='Joint error (torso lengths)',title='Supplementary temporal averaging; both seeds averaged')
    ax.legend(frameon=False);savefig(fig,'pose_timescales')
    segment_rows=[];persistence=[]
    for r in pose['rows']:
        for mode,v in r['extra'].items():
            for name,s in v['segments'].items():segment_rows.append({'probe':r['name'],'seed':r['seed'],'canonical_id':r['canonical_id'],'mode':mode,'segment':name,**s})
        for seconds,error in r['oracle_persistence'].items():persistence.append({'canonical_id':r['canonical_id'],'lag_s':float(seconds),'joint_error_tl':error})
    table('pose_event_context',segment_rows);table('oracle_persistence',persistence)
    table('pose_donor_controls',[{'probe':r['name'],'seed':r['seed'],'canonical_id':r['canonical_id'],**r['donor_control']} for r in pose['rows']])
    emb=read_authenticated(ROOT/'diagnostics/supplementary_embeddings.json');rows=[]
    for family,value in emb['results'].items():
        for r in value['rows']:
            for j,target in enumerate(emb['targets']):rows.append({'family':family,'canonical_id':r['canonical_id'],'fold':r['fold'],'target':target,
                'standardized_mse':r['standardized_mse_by_channel'][j],'train_mean_standardized_mse':r['train_mean_standardized_mse_by_channel'][j],
                'within_song_variance':r['within_song_variance_by_channel'][j],'centered_standardized_mse':r['centered_standardized_mse_by_channel'][j]})
    table('embedding_probes',rows)
    families=['base','fourier','muq','mert','complex_phase','width_control'];fig,ax=plt.subplots(figsize=(10,3.2))
    stats=[emb['results'][f]['summary']['standardized_mse'] for f in families];means=np.array([s['mean'] for s in stats]);bounds=np.array([s['ci95'] for s in stats]).T
    ax.errorbar(range(6),means,yerr=np.maximum(np.array([means-bounds[0],bounds[1]-means]),0),fmt='o',color=BLUE,capsize=4)
    ax.axhline(emb['results']['base']['summary']['train_mean_standardized_mse']['mean'],color=GRAY,ls='--',label='Training-fold constant mean')
    ax.set(xticks=range(6),xticklabels=['Base','Fourier','MuQ','MERT','Complex phase','Width control'],ylabel='Training-fold standardized MSE',title='80-song TRAIN development probes; lower is better');ax.legend(frameon=False)
    savefig(fig,'embedding_controls')
    rhythm=read_authenticated(ROOT/'rhythm/report.json');rows=[]
    for r in rhythm['rows']:
        for method,score in r['scores'].items():rows.append({'key':r['key'],'lead_sheet':r['lead_sheet'],'player':r['player'],'method':method,'target':'beats',**{m:score[m] for m in ('f1','cmlt','amlt')}})
    reference_path=ROOT/'rhythm/beat_this_reference/report.json'
    if reference_path.exists():
        reference=read_authenticated(reference_path)
        for r in reference['rows']:
            for target,score in r['scores'].items():rows.append({'key':r['key'],'lead_sheet':r['lead_sheet'],'player':r['player'],'method':'Beat This! reference','target':target,**{m:score[m] for m in ('f1','cmlt','amlt')}})
    table('rhythm_results',rows)
    methods=['constant_grid','plp']+(['Beat This! reference'] if reference_path.exists() else [])
    fig,axes=plt.subplots(1,2,figsize=(10,3.2))
    for ax,metric in zip(axes,['f1','cmlt']):
        values=[np.mean([r[metric] for r in rows if r['method']==method and r['target']=='beats']) for method in methods]
        ax.bar(range(len(methods)),values,color=[GRAY,BLUE,DARK][:len(methods)],width=.55)
        ax.set(xticks=range(len(methods)),xticklabels=['Constant grid','PLP','Beat This! ref.'][:len(methods)],ylim=(0,1.03),ylabel=metric.upper(),title='GuitarSet: 358 annotated microphone excerpts')
    savefig(fig,'rhythm_benchmark')
    synthetic=read_authenticated(ROOT/'diagnostics/rhythm_stress/report.json');rows=[]
    for r in synthetic['synthetic']:
        for method,score in r['scores'].items():rows.append({'case':r['name'],'method':method,**score})
    table('rhythm_synthetic',rows)
    fig,ax=plt.subplots(figsize=(10,3.3))
    for j,(method,label,color) in enumerate([('constant_grid','Constant grid',GRAY),('local_pulse','PLP',BLUE)]):
        ax.bar(np.arange(7)+(j-.5)*.32,[r['scores'][method]['f1'] for r in synthetic['synthetic']],width=.30,label=label,color=color)
    ax.set(xticks=range(7),xticklabels=['Steady','Drift','Tempo step','Syncopation','Swing','Silence','Meter change'],ylabel='Beat F1',ylim=(0,1.16),title='Synthetic stress tests; meter recognition is not scored by beat F1')
    ax.legend(frameon=False,ncol=2);savefig(fig,'rhythm_stress')
    coaches=read_authenticated(ROOT/'diagnostics/supplementary_coaches.json')
    table('simultaneous_coaches',[{k:v for k,v in r.items() if k!='keys'} for r in coaches['rows']])
    periodicity=read_authenticated(ROOT/'diagnostics/periodicity/report.json')
    table('periodicity_summary',[{k:r[k] for k in ('key','canonical_id','start_frame','dominant_fourier_hz_median','dominant_autocorrelation_hz_median','input_constant_beat_hz')} for r in periodicity['rows']])

def final():
    report=read_authenticated(ROOT/'evaluation/report.json');cells=sorted(report['cells'],key=lambda c:(list(ARMS).index(c['arm']),c['seed']))
    summaries=[];song_rows=[];losses=[]
    for c in cells:
        arm,seed=c['arm'],c['seed'];run=ROOT/'runs'/arm/f'seed_{seed}'
        r=read_authenticated(run/'report.json')
        summaries.append({'arm':arm,'seed':seed,'parameters':r['parameters'],'training_seconds':r['training_seconds'],
            'energy_score':c['mean_energy_score'],'controlled_song_fraction':c['controlled_song_fraction'],'alignment_advantage':c['alignment']['mean'],
            'local_alignment_advantage':c['local_alignment']['mean'],'event_ap_advantage':c['event_ap_advantage']['mean'],'physical_clip_fraction':c['physical_clip_fraction'],
            'brightness_clip_fraction':c['brightness_clip_fraction'],'all_gates_pass':c['all_gates_pass'],**{'gate_'+k:v for k,v in c['gates'].items()}})
        for row in read_authenticated(ROOT/'evaluation'/arm/f'seed_{seed}'/'inference_report.json')['rows']:
            v=row['metrics']
            for mode,es in v['energy_score'].items():song_rows.append({'arm':arm,'seed':seed,'canonical_id':row['canonical_id'],'key':row['key'],'mode':mode,
                'energy_score':es,'alignment_error':v['alignment_errors'][mode],'controlled':v['controlled'],'audio_response_deg':v['audio_response_deg'],
                'noise_response_deg':v['noise_response_deg'],'alignment_advantage':v['alignment_advantage'],'local_alignment_advantage':v['local_alignment_advantage'],'event_ap_advantage':v['event_ap_advantage']})
        losses.extend(json.loads(line) for line in (run/'training.jsonl').read_text().splitlines())
    table('main_cells',summaries);table('main_song_metrics',song_rows);table('training_losses',losses)
    fig,axes=plt.subplots(1,2,figsize=(8,3.7))
    for ax,metric,title in zip(axes,['energy_score','controlled_song_fraction'],['Native score (lower is better)','Sufficient audio response']):
        for seed,col,dx in [(0,DARK,-.12),(1,BLUE,.12)]:
            values=[next(r[metric] for r in summaries if r['arm']==arm and r['seed']==seed) for arm in ARMS]
            ax.scatter(np.arange(6)+dx,values,color=col,label=f'Seed {seed}')
        ax.set(xticks=range(6),xticklabels=[a.replace('_','\n') for a in ARMS],title=title)
        if metric=='controlled_song_fraction':ax.axhline(.75,ls='--',lw=1,color=GRAY);ax.set_ylim(-.03,1.03)
        if metric=='alignment_advantage':ax.axhline(.05,ls='--',lw=1,color=GRAY)
    axes[0].legend(frameon=False);savefig(fig,'main_comparison')
    keys=list(cells[0]['gates']);fig,ax=plt.subplots(figsize=(8,5))
    matrix=np.array([[c['gates'][k] for k in keys] for c in cells],int)
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix,cmap=ListedColormap(['#ededed',BLUE]),vmin=0,vmax=1,aspect='auto')
    for i,j in itertools.product(range(12),range(len(keys))):ax.text(j,i,'PASS' if matrix[i,j] else 'FAIL',ha='center',va='center',fontsize=8,color='white' if matrix[i,j] else DARK)
    gate_labels={'conditioning':'Audio\nresponse','native_alignment':'A donor','local_alignment':'A local','event_selectivity':'B AP','visible_motion':'Visible',
        'brightness':'Dimmer','no_fast_jitter':'Jitter','native_temporal_variation':'Time σ','native_draw_variation':'Draw σ','group_independence':'Groups','mirror_geometry':'Mirror','integrity':'Identity'}
    ax.set(yticks=range(12),yticklabels=[f"{c['arm']} / {c['seed']}" for c in cells],xticks=range(len(keys)),xticklabels=[gate_labels[k] for k in keys],title='Every mandatory gate, every training seed')
    ax.tick_params(axis='x',labelsize=9);ax.tick_params(axis='y',labelsize=10);savefig(fig,'gate_matrix')
    fig,axes=plt.subplots(2,3,figsize=(11,5.4),sharex=True)
    for ax,arm in zip(axes.ravel(),ARMS):
        for seed,col in [(0,DARK),(1,BLUE)]:
            rs=[r for r in losses if r['arm']==arm and r['seed']==seed];ax.plot([r['step'] for r in rs],[r['total'] for r in rs],color=col,lw=.8,label=f'Seed {seed}')
        ax.set(title=arm,ylabel='Total loss',xlabel='Optimizer updates')
    axes[0,0].legend(frameon=False);savefig(fig,'training_curves')
    before_after=read_authenticated(ROOT/'evaluation/incumbent_adapter/report.json');rows=[]
    for r in before_after['rows']:
        for mode,v in r['modes'].items():rows.append({'key':r['key'],'mode':mode,'span_deg':v['beam']['span_2s_median_deg'],'lit_motion_fraction':v['beam']['lit_motion_fraction'],
            'visible_motion_fraction':v['screen']['visible_motion_frame_fraction'],'heads_with_dimmer_span':v['composition']['heads_with_span_ge_point4'],
            'heads_with_150ms_off':v['off_intervals']['heads_with_off_at_least_150ms'],'unrelated_velocity_correlation':v['composition']['unrelated_velocity_abs_correlation']})
    table('adapter_comparison',rows)
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    for ax,metric,title in zip(axes,['heads_with_dimmer_span','heads_with_150ms_off','visible_motion_fraction'],['Heads spanning at least 0.4 dimmer','Heads dark for at least 150 ms','Head-only visible motion fraction']):
        for i,key in enumerate(sorted({r['key'] for r in rows})):
            values=[next(r[metric] for r in rows if r['key']==key and r['mode']==m) for m in ('before','after')]
            ax.plot([0,1],values,'o-',alpha=.65,lw=1,color=BLUE)
        ax.set(xticks=[0,1],xticklabels=['Previous adapter','Expression v4'],title=title,xlim=(-.2,1.2))
    savefig(fig,'adapter_comparison')
    local=read_authenticated(ROOT/'diagnostics/muq_local_controls/report.json');rows=[]
    for cell in local['cells']:
        for r in cell['rows']:
            for mode,values in r['results'].items():rows.append({'arm':cell['arm'],'seed':cell['seed'],'canonical_id':r['canonical_id'],'mode':mode,
                'energy_score':values['energy_score'],'alignment_error':values['alignment_error'],**{f'event_ap_{j}':v for j,v in enumerate(values['event_ap'])}})
    table('muq_local_controls',rows)
    with (ROOT/'delivery/shpongle/whole_track_luma.csv').open() as file:
        rows=[{k:(v if k=='mode' else float(v)) for k,v in r.items()} for r in csv.DictReader(file)]
    table('whole_track_luminance',rows)
    ledger=read_authenticated(ROOT/'delivery/compute_ledger.json')
    table('compute_ledger',ledger['gpu_stage_rows'])

def workbook():
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    from openpyxl.comments import Comment
    from openpyxl.utils import get_column_letter
    wb=Workbook();wb.remove(wb.active)
    for name,rows in TABLES.items():
        ws=wb.create_sheet(name[:31]);headers=list(rows[0]);ws.append(headers)
        for r in rows:ws.append([r[k] for k in headers])
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='343434');cell.alignment=Alignment(wrap_text=True)
        for j,h in enumerate(headers,1):ws.column_dimensions[get_column_letter(j)].width=min(36,max(13,len(h)+2))
        source_map={'pose_errors':'diagnostics/supplementary_pose.json','pose_timescales':'diagnostics/supplementary_pose.json','pose_event_context':'diagnostics/supplementary_pose.json',
            'oracle_persistence':'diagnostics/supplementary_pose.json','pose_donor_controls':'diagnostics/supplementary_pose.json','embedding_probes':'diagnostics/supplementary_embeddings.json',
            'rhythm_results':'rhythm/report.json and rhythm/beat_this_reference/report.json','rhythm_synthetic':'diagnostics/rhythm_stress/report.json','simultaneous_coaches':'diagnostics/supplementary_coaches.json',
            'periodicity_summary':'diagnostics/periodicity/report.json','main_cells':'evaluation/report.json','main_song_metrics':'evaluation/{arm}/seed_{seed}/inference_report.json',
            'training_losses':'runs/{arm}/seed_{seed}/training.jsonl','adapter_comparison':'evaluation/incumbent_adapter/report.json','muq_local_controls':'diagnostics/muq_local_controls/report.json',
            'whole_track_luminance':'delivery/shpongle/whole_track_luma.csv','compute_ledger':'delivery/compute_ledger.json'}
        ws['A1'].comment=Comment('Source relative to '+str(ROOT)+': '+source_map[name]+'. Canonical songs are the resampling unit where confidence intervals are shown.','LightSem')
    ws=wb.create_sheet('Sources');ws.append(['Source type','Title or artifact','Location','SHA256'])
    paths=list((ROOT/'diagnostics').glob('supplementary_*.json'))+[ROOT/'protocol.json',ROOT/'rhythm/report.json',ROOT/'evaluation/report.json',ROOT/'diagnostics/periodicity/report.json',ROOT/'diagnostics/muq_local_controls/report.json',ROOT/'delivery/compute_ledger.json']
    for path in sorted(paths):
        if path.exists():ws.append(['Local artifact',path.name,str(path),digest(path)])
    for row in read_authenticated(ROOT/'research/source_ledger.json')['sources']:ws.append(['Primary source '+str(row['number']),row['authors']+'. '+row['title']+'. '+row['publication'],row['url'],''])
    for col,width in [('A',24),('B',70),('C',90),('D',68)]:ws.column_dimensions[col].width=width
    ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    wb.save(OUT/'chart_data.xlsx')
    FILE_OUTPUTS.append(OUT/'chart_data.xlsx')
    authenticated_json(OUT/'figure_manifest.json',{'artifacts':{p.name:digest(p) for p in sorted(set(FILE_OUTPUTS))},'source_sha256':digest(__file__),'tables':{k:len(v) for k,v in TABLES.items()}})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');a=ap.parse_args()
    diagnostics()
    if a.final:final()
    workbook()
