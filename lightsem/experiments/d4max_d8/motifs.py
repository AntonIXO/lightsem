"""D4-max: a small music-indexed motif bank, composed through feasible joins.

Only the supplied song's body fragments produce motion. Learned audio features
select motifs; no temporal oscillator, independent dance signal or fitted pose
model exists here. Six audio families and at most three takes per family are
fixed before outputs; the path lexicographically minimizes non-bank fallback.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.ndimage import gaussian_filter1d
from ..dynamic_dance.contracts import DanceResult,DancePlan,MusicTimeline
from ..dynamic_dance.retiming import (resample_body,contacts_from_body,_blend_bodies,
                                     FEET,REGIONS)

@dataclass(frozen=True)
class Config:
    bars_per_motif:int=2
    audio_families:int=6
    takes_per_family:int=3
    descriptor_phases:int=4
    beam_width:int=64
    pose_gap_limit_m:float=.25
    minimum_rate:float=.5
    maximum_rate:float=1.5
    transition_s:float=.5
    muq_weight:float=.7
    source_weight:float=.3


def _unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def _clock(world,fps):
    w=np.asarray(world,np.float64)
    if w.ndim!=3 or w.shape[1:]!=(22,3) or len(w)<3 or not np.isfinite(w).all():raise ValueError('Finite own-song22-joint source required')
    if fps!=30:raise ValueError('Native30Hz required')
    return w,np.arange(len(w))/fps


def phrase_nodes(music,frames,cfg=Config(),fps=30):
    last=(frames-1)/fps
    beats=np.asarray(music.downbeats,float)
    beats=beats[(beats>0)&(beats<last)]
    # AIO boundaries are estimated musical anchors, never a verse/chorus speed rule.
    structural=[float(s['start']) for s in music.sections if 0<float(s['start'])<last]
    boundaries=[0.]
    for beat in beats[::cfg.bars_per_motif]:
        if beat-boundaries[-1]>=1.:boundaries.append(float(beat))
    boundaries+=structural+[last]
    points=np.unique(np.rint(np.sort(boundaries)*fps).astype(int))
    # Merge tiny boundary slivers without inventing a beat grid.
    keep=[0]
    for p in points[1:-1]:
        if p-keep[-1]>=30 and frames-1-p>=30:keep.append(int(p))
    keep.append(frames-1)
    return [{'start':a,'end':b,'start_s':a/fps,'end_s':b/fps} for a,b in zip(keep[:-1],keep[1:])]


def acoustic_features(music,nodes,cfg=Config(),reference=None):
    muq=np.asarray(music.muq,np.float64)
    if reference is None:
        center=np.mean(muq,axis=0)
        source_scale=np.maximum(np.quantile(music.sources,.9,axis=0),1e-8)
        reference={'muq_center':center,'source_scale':source_scale}
    else:
        center=np.asarray(reference['muq_center']);source_scale=np.asarray(reference['source_scale'])
    vectors=[]
    for node in nodes:
        a,b=node['start_s'],node['end_s'];phases=np.linspace(a,b,cfg.descriptor_phases+1)
        means=[];stems=[]
        for left,right in zip(phases[:-1],phases[1:]):
            mt=np.asarray(music.muq_time);mask=(mt>=left)&(mt<right)
            if mask.any():part=muq[mask].mean(0)
            else:part=np.array([np.interp((left+right)/2,mt,muq[:,i]) for i in range(muq.shape[1])])
            means.append(part-center)
            mask=(music.time>=left)&(music.time<right)
            values=music.sources[mask].mean(0) if mask.any() else music.sources[np.argmin(abs(music.time-(left+right)/2))]
            stems.append(np.log1p(values/source_scale))
        m=_unit(np.asarray(means).reshape(1,-1))[0];s=_unit(np.asarray(stems).reshape(1,-1))[0]
        vectors.append(np.r_[np.sqrt(cfg.muq_weight)*m,np.sqrt(cfg.source_weight)*s])
    x=_unit(np.asarray(vectors))
    if not np.isfinite(x).all():raise ValueError('Invalid audio descriptors')
    return x,reference


def kmedoids(features,k):
    x=np.asarray(features,float);norm=np.sum(x*x,axis=1);distance=np.clip(.5*(norm[:,None]+norm[None,:])-x@x.T,0,2);n=len(x)
    k=min(k,n);medoids=[int(np.argmin(distance.sum(1)))]
    while len(medoids)<k:
        separation=distance[:,medoids].min(1);separation[medoids]=-1
        if separation.max()<=1e-12:break
        medoids.append(int(np.argmax(separation)))
    for _ in range(20):
        labels=np.argmin(distance[:,medoids],axis=1);updated=[]
        for label,medoid in enumerate(medoids):
            group=np.flatnonzero(labels==label)
            updated.append(int(group[np.argmin(distance[np.ix_(group,group)].sum(1))]) if len(group) else medoid)
        if updated==medoids:break
        medoids=updated
    return np.asarray(medoids,int),np.argmin(distance[:,medoids],axis=1),distance


def build_bank(world,music,cfg=Config()):
    w,time=_clock(world,30);nodes=phrase_nodes(music,len(w),cfg)
    x,reference=acoustic_features(music,nodes,cfg);medoids,labels,distance=kmedoids(x,cfg.audio_families)
    contact=contacts_from_body(w,30);takes={}
    for family,medoid in enumerate(medoids):
        group=np.flatnonzero(labels==family)
        # Interior clips have continuation for the physical crossfade.
        pool=[int(i) for i in group if i not in (0,len(nodes)-1)]
        pool.sort(key=lambda i:(distance[i,medoid],i));chosen=[];signatures=set()
        for i in pool:
            node=nodes[i];signature=tuple(contact[node['start']].astype(int))+tuple(contact[node['end']].astype(int))
            if signature not in signatures:
                chosen.append(i);signatures.add(signature)
            if len(chosen)==cfg.takes_per_family:break
        for i in pool:
            if len(chosen)>=cfg.takes_per_family:break
            if i not in chosen:chosen.append(i)
        takes[int(family)]=chosen
    return {'nodes':nodes,'features':x,'feature_reference':reference,'medoids':medoids,'labels':labels,'takes':takes,'config':asdict(cfg),'audio_only_family_definition':True}


def _edge(w,contact,previous,current,gap):
    a,b=previous['end'],current['start']
    if not np.array_equal(contact[a],contact[b]):return None
    q=(w[a]-w[a,0])-(w[b]-w[b,0]);cost=float(np.sqrt(np.mean(q*q)))
    return cost if cost<=gap else None


def prune_end_states(paths):
    best={}
    for path in paths:
        key=path[3]
        if key not in best or path[:3]<best[key][:3]:best[key]=path
    return sorted(best.values(),key=lambda x:x[:3])


def select_path(world,bank,target_music,cfg=Config(),control=None):
    w,time=_clock(world,30);target_nodes=phrase_nodes(target_music,len(w),cfg)
    # Identity fallback nodes always preserve the target-time source trajectory.
    source_nodes=bank['nodes'];sources=list(source_nodes)+[dict(n) for n in target_nodes]
    q,_=acoustic_features(target_music,target_nodes,cfg,bank['feature_reference'])
    centers=bank['features'][bank['medoids']];families=np.argmax(q@centers.T,axis=1)
    if control=='shuffle':families=np.roll(families,max(1,len(families)//2))
    elif control not in (None,'chronological'):raise ValueError('Unknown motif control')
    contact=contacts_from_body(w,30);beam=[(0,0.,[],None)];edge_cache={}
    for ti,node in enumerate(target_nodes):
        identity=len(source_nodes)+ti;family=int(families[ti]);possible=[]
        for donor in bank['takes'].get(family,[]):
            d=sources[donor];rate=(d['end']-d['start'])/(node['end']-node['start'])
            if cfg.minimum_rate<=rate<=cfg.maximum_rate:possible.append(donor)
        if ti in (0,len(target_nodes)-1) or control=='chronological':possible=[]
        possible=list(dict.fromkeys(possible+[identity]));extended=[]
        for fallback,cost,path,prev in beam:
            for donor in possible:
                if prev is None:seam=0.
                else:
                    key=(prev,donor)
                    if key not in edge_cache:edge_cache[key]=_edge(w,contact,sources[prev],sources[donor],cfg.pose_gap_limit_m)
                    seam=edge_cache[key]
                    if seam is None:continue
                is_fallback=donor==identity
                audio_cost=0. if is_fallback else float(1-np.dot(q[ti],bank['features'][donor]))
                extended.append((fallback+int(is_fallback),cost+audio_cost+.2*seam,path+[donor],donor))
        # Whole chronological path is an exact continuous candidate, retained
        # independently of beam width to prevent a later dead end.
        chronological=[len(source_nodes)+j for j in range(ti+1)]
        item=(ti+1,0.,chronological,identity)
        if not any(x[2]==chronological for x in extended):extended.append(item)
        # Future transitions depend only on the last donor at this target layer.
        # Preserve the best path for EACH end state; repeated paths to one donor
        # must not consume the beam and evict another feasible donor state.
        beam=prune_end_states(extended)

    best=min(beam,key=lambda x:(x[0],x[1],x[2]))
    return sources,target_nodes,best[2],families,{'nonbank_fallbacks':best[0],'path_secondary_cost':best[1],'edge_pairs_checked':len(edge_cache),'rejected_edges':sum(v is None for v in edge_cache.values())}


def compose(world,music,cfg=Config(),*,bank=None,target_music=None,control=None):
    w,time=_clock(world,30);bank=build_bank(w,music,cfg) if bank is None else bank;target_music=music if target_music is None else target_music
    meta={'method':'D4_max','source_order_preserved':False,'independent_motion_source':False,'motion_origin':'audio-indexed recomposition of own-song D7 body','source_library_label':'D7','configuration':asdict(cfg),'selection':'Exact dynamic programming over last-donor states: lexicographically minimize nonbank fallback then acoustic/seam cost','time_map_classification':'recomposition; source jumps are not playback rates','control':control,'fragment_provenance':[]}
    if np.array_equal(w,np.broadcast_to(w[:1],w.shape)):
        return DanceResult(time,w.astype(np.float32),contacts_from_body(w),time.copy(),metadata={**meta,'diagnostic_reason':'frozen source, no independent motion'}),DancePlan(time,time.copy(),np.ones((len(w),1)),metadata=meta),bank
    sources,nodes,path,families,search=select_path(w,bank,target_music,cfg,control)
    output=w.copy();clock=time.copy();rates=np.ones((len(w),1));translations=[];source_contacts=contacts_from_body(w)
    for index,(node,donor_id) in enumerate(zip(nodes,path)):
        donor=sources[donor_id];a,b=node['start'],node['end'];ds,de=donor['start'],donor['end']
        source_clock=np.linspace(ds/30,de/30,b-a+1);segment=resample_body(w,source_clock,30.)
        shift=np.zeros(3)
        if index:
            previous=sources[path[index-1]];common=source_contacts[ds].astype(bool)
            if common.any():shift=(output[a,np.asarray(FEET)[common]].mean(0)-segment[0,np.asarray(FEET)[common]].mean(0))
            else:shift=output[a,0]-segment[0,0]
            shift[1]=0.;segment+=shift
            fade=min(round(cfg.transition_s*30),max(0,(b-a)//4))
            if fade:
                prev_rate=(previous['end']-previous['start'])/(nodes[index-1]['end']-nodes[index-1]['start'])
                cont=np.minimum(previous['end']/30+np.arange(fade+1)/30*prev_rate,time[-1])
                prior=resample_body(w,cont,30.)+translations[-1]
                u=np.linspace(0,1,fade+1);weight=6*u**5-15*u**4+10*u**3
                segment[:fade+1]=_blend_bodies(prior,segment[:fade+1],weight)
        output[a:b+1]=segment;clock[a:b+1]=source_clock;rate=(de-ds)/(b-a);rates[a:b+1,0]=rate;translations.append(shift)
        meta['fragment_provenance'].append({'target_fragment':index,'motif_family':int(families[index]),'source_fragment':int(donor_id),'target_seconds':[a/30,b/30],'source_seconds':[ds/30,de/30],'local_rate':rate,'canonical_bank_take':bool(donor_id<len(bank['nodes'])),'root_translation_m':shift.tolist(),'transition_s':min(cfg.transition_s,(b-a)/30/4) if index else 0.})
    meta.update(search,bank_fragment_count=sum(x['canonical_bank_take'] for x in meta['fragment_provenance']),recomposed_fragment_count=sum(x['target_seconds']!=x['source_seconds'] for x in meta['fragment_provenance']),graph_node_count=len(nodes),bank_families=len(bank['medoids']),bank_take_count=sum(map(len,bank['takes'].values())),bank_source_nodes={str(k):v for k,v in bank['takes'].items()})
    meta['motif_anchor_frames']=[nodes[0]['start']]+[n['end'] for n in nodes]
    meta['selected_body_fraction_from_bank']=sum((n['end']-n['start']) for n,di in zip(nodes,path) if di<len(bank['nodes']))/(len(w)-1)
    if meta['recomposed_fragment_count']==0:output=w.copy();clock=time.copy();rates[:]=1.;meta['diagnostic_reason']='no feasible canonical motif; identity preserved'
    if not np.isfinite(output).all():raise ValueError('Nonfinite composed body')
    result=DanceResult(time=time,world=output.astype(np.float32),contacts=contacts_from_body(output),source_time=clock,metadata=meta)
    plan=DancePlan(time=time,source_time=clock,effort=rates.astype(np.float32),metadata=meta)
    result.validate();plan.validate();return result,plan,bank
