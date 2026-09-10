"""Emitted-DMX illumination diagnostics, not a music/aesthetic qualification.

Axis intersections use finite floor/back wall from shaders.py; truss occlusion
is not modelled. Cone support uses the shader's soft edge at half strength.
Surface-grid and camera-area metrics are geometric support, NOT pixel luminance.
All temporal geometry uses every DMX frame, regardless of brightness. Visible
support is reported separately so flashing cannot masquerade as choreography.
"""
import numpy as np
from ..quality_observer import beam_series


def first_hits(directions, positions):
    v, p = np.asarray(directions), np.asarray(positions)
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = -p[..., 1]/v[..., 1]
        tw = (-6-p[..., 2])/v[..., 2]
        floor = p + tf[..., None]*v
        wall = p + tw[..., None]*v
    vf = (tf>0)&(abs(floor[...,0])<11)&(floor[...,2]>-7)&(floor[...,2]<9)
    vw = (tw>0)&(abs(wall[...,0])<9)&(wall[...,1]>=0)&(wall[...,1]<9)
    tf = np.where(vf,tf,np.inf); tw = np.where(vw,tw,np.inf)
    distance = np.minimum(tf,tw)
    valid = np.isfinite(distance)
    hit = p + np.where(valid,distance,0)[...,None]*v
    hit = np.where(valid[...,None],hit,np.nan)
    surface = np.where(valid,np.where(tf<=tw,0,1),-1)
    return hit, surface, floor


def project(points):
    camera=np.array([0.,2.6,11.]); forward=np.array([0.,3.2,-3.])-camera
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    delta=points-camera;depth=delta@forward
    return np.stack(((delta@right)/depth/.85/(16/9),(delta@up)/depth/.85),-1)


def surface_grid(step=.25):
    # Fixed stage ROI, not an adaptive box that flatters a collapsed pattern.
    x,z=np.meshgrid(np.arange(-8+step/2,8,step),np.arange(-6+step/2,3,step))
    floor=np.stack((x,x*0,z),-1).reshape(-1,3)
    x,y=np.meshgrid(np.arange(-8+step/2,8,step),np.arange(step/2,8,step))
    wall=np.stack((x,y,x*0-6),-1).reshape(-1,3)
    grid=np.concatenate((floor,wall))
    # Small-cell projected area via four corners. Overlap at wall/floor seam
    # has zero area; truss and grazing camera occlusion are not represented.
    corners=[]
    for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        off=np.concatenate((np.tile([a,0,b],(len(floor),1)),np.tile([a,b,0],(len(wall),1))))*step/2
        corners.append(project(grid+off))
    uv=np.stack(corners,1)
    area=abs(np.sum(uv[:,:,0]*np.roll(uv[:,:,1],-1,axis=1)-uv[:,:,1]*np.roll(uv[:,:,0],-1,axis=1),axis=1))/2/4
    return grid,area


def longest_run(mask,fps):
    padded=np.r_[False,mask,False].astype(int);changes=np.diff(padded)
    return float(np.max(np.flatnonzero(changes==-1)-np.flatnonzero(changes==1),initial=0)/fps)


def spatial_series(frames,rig,fps=40.,grid_hz=5.):
    d=np.asarray(frames);heads=rig.by_kind('moving_head')
    positions=np.array([f.position for f in heads]);v,w=beam_series(d,rig)
    hit,surface,floor=first_hits(v,positions)
    # An invalid ray is a gate failure, not silently dropped.
    valid=(surface>=0).all(1)
    center=np.nanmedian(hit,axis=1)
    distances=np.linalg.norm(hit-center[:,None],axis=-1)
    focus=(np.sum(distances<1.5,axis=1)>=4)&valid
    central=(np.sum((abs(hit[:,:,0])<1.5)&(np.linalg.norm(hit[:,:,[1,2]]-[0,-4.5],axis=-1)<2),axis=1)>=4)&valid
    grid,area=surface_grid();delta=grid[None]-positions[:,None]
    norm=np.linalg.norm(delta,axis=-1);unit=delta/norm[...,None]
    idx=np.arange(0,len(d),max(1,round(fps/grid_hz)))
    cover=[];overlap=[];screen=[];litcover=[]
    for t in idx:
        half=np.deg2rad(np.array([f.beam_min_deg+d[t,f.address-1+9]/255*(f.beam_max_deg-f.beam_min_deg) for f in heads])/2)
        # smoothstep(edge-.018,edge+.0072,cosA)==.5 at edge-.0054
        support=np.einsum('hgc,hc->hg',unit,v[t]) >= (np.cos(half)-.0054)[:,None]
        counts=support.sum(0);union=counts>0
        cover.append(float(union.sum()*.25**2))
        overlap.append(float(np.sum(counts>=2)/max(union.sum(),1)))
        screen.append(float(area[union].sum()))
        litcover.append(float(np.any(support&(w[t]>.03)[:,None],axis=0).sum()*.25**2))
    return {'hit':hit,'surface':surface,'floor':floor,'valid':valid,'focus':focus,'central':central,
            'floor_x_span':np.ptp(floor[:,:,0],axis=1),'hit_x_span':np.ptp(hit[:,:,0],axis=1),
            'hit_z_span':np.ptp(hit[:,:,2],axis=1),'projected_axis_x_span':np.ptp(project(hit)[:,:,0],axis=1)/2,
            'grid_time':idx/fps,'coverage':np.array(cover),'overlap':np.array(overlap),
            'projected_coverage':np.array(screen),'lit_coverage':np.array(litcover),'level':w}


def summary(series,lo,hi,fps=40.):
    sl=slice(round(lo*fps),round(hi*fps));hit=series['hit'][sl]
    out={}
    for k in ('floor_x_span','hit_x_span','hit_z_span','projected_axis_x_span'):
        a=series[k][sl];out[k]={q:float(np.nanquantile(a,p)) for q,p in [('p10',.1),('median',.5),('p90',.9)]}
    gi=(series['grid_time']>=lo)&(series['grid_time']<hi)
    for k in ('coverage','overlap','projected_coverage','lit_coverage'):
        out[k+'_median']=float(np.median(series[k][gi]))
    out.update(valid_fraction=float(series['valid'][sl].mean()),focus_fraction=float(series['focus'][sl].mean()),
               focus_longest_s=longest_run(series['focus'][sl],fps),central_fraction=float(series['central'][sl].mean()))
    extents=[];relative=[];shape=[]
    for start in range(0,len(hit)-round(4*fps)+1,round(fps)):
        chunk=hit[start:start+round(4*fps)]
        centered=chunk-np.mean(chunk,axis=1,keepdims=True)
        extents.extend(np.linalg.norm(np.nanquantile(chunk,.9,axis=0)-np.nanquantile(chunk,.1,axis=0),axis=-1))
        relative.extend(np.linalg.norm(np.nanquantile(centered,.9,axis=0)-np.nanquantile(centered,.1,axis=0),axis=-1))
        dist=np.linalg.norm(chunk[:,:,None]-chunk[:,None,:],axis=-1)[:,np.triu_indices(6,1)[0],np.triu_indices(6,1)[1]]
        shape.append(float(np.nanmedian(np.nanquantile(dist,.9,axis=0)-np.nanquantile(dist,.1,axis=0))))
    out['trajectory_4s_median_m']=float(np.nanmedian(extents)) if extents else 0.
    out['relative_trajectory_4s_median_m']=float(np.nanmedian(relative)) if relative else 0.
    out['pair_distance_change_4s_median_m']=float(np.nanmedian(shape)) if shape else 0.
    return out


# Frozen on engineering controls and the rejected DMX BEFORE candidate mapping.
THRESHOLDS={'span_p10_min_m':4.,'coverage_min_m2':8.,'focus_fraction_max':.2,
            'focus_longest_max_s':3.,'valid_fraction_min':.99,'overlap_max':.45,
            'trajectory_min_m':.8,'relative_trajectory_min_m':.5,'pair_change_min_m':.35}


def gates(s,thresholds=THRESHOLDS,active=True):
    t=thresholds
    result={'spatial_spread':s['hit_x_span']['p10']>=t['span_p10_min_m'],
            'surface_coverage':s['coverage_median']>=t['coverage_min_m2'],
            'not_concentrated':s['focus_fraction']<=t['focus_fraction_max'] and s['focus_longest_s']<=t['focus_longest_max_s'],
            'surface_valid':s['valid_fraction']>=t['valid_fraction_min'],
            'cone_separation':s['overlap_median']<=t['overlap_max']}
    if active:
        result.update(trajectory=s['trajectory_4s_median_m']>=t['trajectory_min_m'],
                      not_just_translation=s['relative_trajectory_4s_median_m']>=t['relative_trajectory_min_m'],
                      changing_shape=s['pair_distance_change_4s_median_m']>=t['pair_change_min_m'])
    return result


def encode_targets(targets,rig,level=.7):
    from ..beat_conditioned_render import _angles
    frames=np.zeros((len(targets),512),np.uint8)
    for i,f in enumerate(rig.by_kind('moving_head')):
        pan,tilt=_angles(targets[:,i]-f.position)
        ticks=np.rint((.5+np.stack((pan,tilt),-1)/[f.pan_range_deg,f.tilt_range_deg])*65535).astype(int)
        a=f.address-1
        frames[:,a:a+4]=np.stack((ticks[:,0]>>8,ticks[:,0]&255,ticks[:,1]>>8,ticks[:,1]&255),-1)
        frames[:,a+4]=round(level*255);frames[:,a+6:a+9]=255;frames[:,a+9]=72
    return frames


def controls(rig,fps=40.,seconds=24):
    n=round(seconds*fps);time=np.arange(n)/fps
    base=np.array([[-5,0,-2],[-3,0,-4],[-1,0,-2],[1,0,-4],[3,0,-2],[5,0,-4]],float)
    spread=np.broadcast_to(base,(n,6,3)).copy()
    common=np.broadcast_to([0,0,-4.5],(n,6,3)).copy()
    # Piecewise paths ONLY in deliberate diagnostic controls, never production.
    knots=np.array([0,2,4,6,8,10,12,14,16,18,20,22,24])
    seq=np.array([0,1,-1,.4,-.8,.8,0,-1,1,-.4,.5,-.7,0])
    translation=np.interp(time,knots,seq)
    meaningful=spread.copy()
    for i in range(6):
        meaningful[:,i,0]+=.8*np.interp(time,knots,np.roll(seq,i))
        meaningful[:,i,2]+=1.4*np.interp(time,knots,np.roll(seq,i+3))
    jitter=spread.copy();jitter[:,:,0]+=.025*((np.arange(n)%2)*2-1)[:,None]
    translated=spread.copy();translated[:,:,0]+=translation[:,None]
    result={k:encode_targets(v,rig) for k,v in {'common_point':common,'static_spread':spread,'jitter':jitter,'translation':translated,'separated_motion':meaningful}.items()}
    flashing=result['static_spread'].copy()
    for f in rig.by_kind('moving_head'):flashing[:,f.address-1+4]=np.where(np.arange(n)%40<20,230,0)
    result['flashing_spread']=flashing
    return result
