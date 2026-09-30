"""RGB-only candidate delivery over accepted C mechanics and brightness."""
from __future__ import annotations
import colorsys
import numpy as np
from lightsem.conductor.rig import canonical_rig
from lightsem.model.light_layers.overlay import apply_overlay
from lightsem.model.light_layers.transitions import fade_rgb_targets
from .palettes import fixture_groups,group_rgb_mask,prototype_representatives
from .oracle import GROUP_ORDER


def global_palette_rgb(prototypes):
    """Same saturation-weighted circular visualization as the accepted06 demo."""
    p=np.asarray(prototypes,np.float64).reshape(-1,12,6)
    h=np.arange(12)/12;s=np.arange(6)/5
    z=np.sum(p*s[None,None,:]*np.exp(2j*np.pi*h)[None,:,None],axis=(1,2))
    hue=np.where(np.abs(z)==0,0.,np.angle(z)/(2*np.pi)%1)
    saturation=np.minimum(np.abs(z),1)
    return np.asarray([colorsys.hsv_to_rgb(float(a),float(b),1.) for a,b in zip(hue,saturation)])


def color_delivery(baseline_c,companion_before_color,times,ids10,cues10,prototypes,*,layout='global'):
    """Cue/history decisions come only from the candidate's actual audio rollout.

    Before its first cue, use the original companion's RGB, avoiding leakage
    from C's learned colors. Keep all479non-RGB C bytes exact throughout. The
    optional fixed2/4group layouts are labeled palette-design previews, not a
    trained audio-to-spatial allocator or privileged H4 oracle.
    """
    c=np.asarray(baseline_c);base=np.asarray(companion_before_color);t=np.asarray(times,float)
    ids=np.asarray(ids10);cues=np.asarray(cues10)
    if c.dtype!=np.uint8 or c.shape!=base.shape or c.ndim!=2 or c.shape[1]!=512 or len(c)!=len(ids)*4:
        raise ValueError('Full-record DMX and10Hz palette timeline required')
    if cues.dtype!=bool or cues.shape!=ids.shape or ids.dtype.kind not in 'iu' or np.any((ids<0)|(ids>=32)):
        raise ValueError('Invalid candidate palette/cue outputs')
    if not np.array_equal(t,np.arange(len(c))/40):raise ValueError('Fixed40Hz source clock required')
    if np.any((ids[1:]!=ids[:-1])&~cues[1:]):raise ValueError('Palette state changed outside model cues')
    rgb=global_palette_rgb(prototypes);choices=np.floor(rgb*255+.5).astype(np.uint8)[:,None,:]
    if layout!='global':
        if layout not in ('two_groups','four_groups'):raise ValueError('Unknown layout preview')
        representatives=np.asarray([[m['rgb'] for m in prototype_representatives(p)] for p in prototypes])
        choices=np.concatenate([choices,np.floor(representatives*255+.5).astype(np.uint8)],axis=1)
    cue_times=np.flatnonzero(cues)/10
    active=np.zeros(len(c),bool) if not len(cue_times) else t>=cue_times[0]
    ids40=np.repeat(ids,4);groups=fixture_groups(canonical_rig());tracks={}
    assignment={'global':[0,0,0,0],'two_groups':[1,2,1,2],'four_groups':[1,2,0,3]}[layout]
    for f in groups:
        slots=f['rgb_slots_zero_based'];track=base[:,slots].astype(float)/255
        mode=assignment[GROUP_ORDER.index(f['o4'])]
        track[active]=choices[ids40[active],mode]/255
        tracks[f['name']]=fade_rgb_targets(track,t,duration_s=.3,active=active)
    stream=apply_overlay(c,canonical_rig(),t,rgb=tracks)
    mask=group_rgb_mask(groups)
    if not np.array_equal(stream[:,~mask],c[:,~mask]):raise RuntimeError('Protected479bytes changed')
    if not np.array_equal(stream[~active][:,mask],base[~active][:,mask]):raise RuntimeError('Learned C RGB leaked before candidate activation')
    return stream,{'layout':layout,'protected_nonRGB_slots':479,'history':'own_generated_state',
        'fade_s':.3,'color_cue_count':len(cue_times),'active_color_seconds':float(active.sum()/40),
        'subsequent_palette_changes':int(np.count_nonzero(np.diff(ids40[active]))) if active.any() else 0,
        'fixed_group_mode_indices':assignment,'scope':'RGB-only personal review; C movement and dimmers fixed; source companion RGB before first predicted cue.'}
