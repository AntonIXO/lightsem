"""Moonless, empty festival: Cycles light transport and passive real-scale assets.

The same DMX bytes drive this backend. Every photon comes from a patched
fixture, including actual textured projector lights rather than glowing screens.
"""
from __future__ import annotations

import math
import bpy
from mathutils import Vector

from .blender_scene import point_at, curves, screen, _math, _key
from .realistic_geometry import build_environment, _material


def configure_cycles(scene, samples):
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX'
    prefs.get_devices()
    for device in prefs.devices:
        device.use=device.type=='OPTIX'
    active=[d.name for d in prefs.devices if d.use]
    if not active:
        raise RuntimeError('The photographic renderer requires a working OptiX GPU')
    scene.render.engine='CYCLES'
    scene.cycles.device='GPU'
    scene.cycles.samples=samples
    scene.cycles.use_adaptive_sampling=True
    scene.cycles.adaptive_threshold=.035
    scene.cycles.adaptive_min_samples=min(16,samples)
    scene.cycles.use_denoising=True
    scene.cycles.denoiser='OPTIX'
    scene.cycles.max_bounces=7
    scene.cycles.diffuse_bounces=3
    scene.cycles.glossy_bounces=4
    scene.cycles.transmission_bounces=4
    scene.cycles.volume_bounces=1
    scene.cycles.transparent_max_bounces=8
    scene.cycles.sample_clamp_indirect=3
    scene.cycles.use_animated_seed=False
    scene['Render GPU']=', '.join(active)
    return active


def rounded_box(name, position, size, material, bevel=.012, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1,location=position)
    obj=bpy.context.object; obj.name=name
    obj.scale=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    obj.data.materials.append(material)
    mod=obj.modifiers.new('Machined edge radius','BEVEL'); mod.width=bevel; mod.segments=3
    obj.modifiers.new('Weighted face normals','WEIGHTED_NORMAL')
    if parent:
        obj.parent=parent; obj.location=position
    return obj


def spot(name, pos, target):
    data=bpy.data.lights.new(name,'SPOT')
    data.energy=0; data.spot_blend=.22; data.shadow_soft_size=.035
    data.use_shadow=True
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj)
    obj.location=pos; point_at(obj,target)
    obj['dmx_fixture']=name.split(' · ')[0]
    return obj


def projector_shader(light):
    """Real light-local perspective gobo; shadowing is handled by Cycles."""
    data=light.data; data.use_nodes=True; data.color=(1,1,1)
    nodes=data.node_tree.nodes; links=data.node_tree.links; nodes.clear()
    out=nodes.new('ShaderNodeOutputLight'); emit=nodes.new('ShaderNodeEmission')
    links.new(emit.outputs[0],out.inputs['Surface'])
    tex=nodes.new('ShaderNodeTexCoord'); sep=nodes.new('ShaderNodeSeparateXYZ')
    links.new(tex.outputs['Normal'],sep.inputs[0])
    depth=_math(nodes,links,'MULTIPLY',sep.outputs['Z'],-1)
    zoom=nodes.new('ShaderNodeValue'); zoom.name='DMX Zoom'; zoom.outputs[0].default_value=2
    x=_math(nodes,links,'MULTIPLY',_math(nodes,links,'DIVIDE',sep.outputs['X'],depth),zoom.outputs[0])
    y=_math(nodes,links,'MULTIPLY',_math(nodes,links,'DIVIDE',sep.outputs['Y'],depth),zoom.outputs[0])
    r=_math(nodes,links,'SQRT',_math(nodes,links,'ADD',_math(nodes,links,'MULTIPLY',x,x),_math(nodes,links,'MULTIPLY',y,y)))
    theta=_math(nodes,links,'ARCTAN2',y,x)
    clock=nodes.new('ShaderNodeValue'); clock.name='Media Clock'
    pattern=nodes.new('ShaderNodeValue'); pattern.name='DMX Pattern'; pattern.outputs[0].default_value=10
    petals=_math(nodes,links,'SINE',_math(nodes,links,'ADD',_math(nodes,links,'MULTIPLY',theta,pattern.outputs[0]),_math(nodes,links,'MULTIPLY',clock.outputs[0],.25)))
    phase=_math(nodes,links,'ADD',_math(nodes,links,'MULTIPLY',r,46),_math(nodes,links,'MULTIPLY',petals,.9))
    stripes=_math(nodes,links,'POWER',_math(nodes,links,'ABSOLUTE',_math(nodes,links,'SINE',_math(nodes,links,'SUBTRACT',phase,clock.outputs[0]))),12)
    lattice=_math(nodes,links,'POWER',_math(nodes,links,'ABSOLUTE',_math(nodes,links,'SINE',_math(nodes,links,'ADD',_math(nodes,links,'MULTIPLY',theta,16),_math(nodes,links,'MULTIPLY',r,18)))),20)
    mask=_math(nodes,links,'ADD',stripes,_math(nodes,links,'MULTIPLY',lattice,.28))
    gate=_math(nodes,links,'MULTIPLY',_math(nodes,links,'LESS_THAN',_math(nodes,links,'ABSOLUTE',x),1),_math(nodes,links,'LESS_THAN',_math(nodes,links,'ABSOLUTE',y),.625))
    gate=_math(nodes,links,'MULTIPLY',gate,_math(nodes,links,'GREATER_THAN',depth,0))
    links.new(_math(nodes,links,'MULTIPLY',mask,gate),emit.inputs['Strength'])
    ramp=nodes.new('ShaderNodeValToRGB'); ramp.name='Media palette'
    links.new(_math(nodes,links,'PINGPONG',_math(nodes,links,'ADD',_math(nodes,links,'MULTIPLY',r,1.8),_math(nodes,links,'MULTIPLY',clock.outputs[0],.025)),1),ramp.inputs[0])
    ramp.color_ramp.elements[0].color=(.025,.28,.8,1)
    ramp.color_ramp.elements[1].color=(1,.16,.04,1)
    e=ramp.color_ramp.elements.new(.42); e.color=(.55,.015,.35,1)
    e=ramp.color_ramp.elements.new(.7); e.color=(.04,.68,.42,1)
    rgb=nodes.new('ShaderNodeRGB'); rgb.name='DMX Color'
    mix=nodes.new('ShaderNodeMixRGB'); mix.blend_type='MULTIPLY'; mix.inputs[0].default_value=1
    links.new(ramp.outputs[0],mix.inputs[1]); links.new(rgb.outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],emit.inputs['Color'])


def _fixture_body(name, origin, material):
    body=rounded_box(name+' · housing',(0,0,.24),(.34,.40,.45),material,parent=origin)
    for sign in (-1,1):
        rounded_box(name+f' · yoke {sign}',(sign*.23,0,.35),(.055,.22,.56),material,.007,parent=origin)
    rounded_box(name+' · rear ventilation',(0,.204,.29),(.25,.012,.15),material,.002,parent=origin)
    # Lens sits behind the source aperture, so housing cannot shadow its own beam.
    mat=_material(name+' · lens',(.012,.014,.017),roughness=.13)
    mat['dmx_fixture']=name
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=12,radius=.125)
    lens=bpy.context.object; lens.name=name+' · glass lens'
    lens.parent=origin; lens.location=(0,0,.008); lens.scale=(1,1,.10)
    lens.data.materials.append(mat); lens.visible_shadow=False
    for poly in lens.data.polygons: poly.use_smooth=True
    return body,mat


def build_scene(patch, width=1920, height=1080, samples=64, haze=.008):
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    configure_cycles(scene,samples)
    scene.render.resolution_x=width; scene.render.resolution_y=height; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False; scene.render.fps=30
    scene.render.dither_intensity=0
    scene.world=bpy.data.worlds.new('Absolute black environment'); scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(0,0,0,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0
    scene.view_settings.view_transform='AgX'; scene.view_settings.look='AgX - Medium High Contrast'
    scene.view_settings.exposure=1.3
    env=build_environment(patch)
    haze_mat=bpy.data.materials.new('Unlit heterogeneous haze'); haze_mat.use_nodes=True
    n=haze_mat.node_tree.nodes; l=haze_mat.node_tree.links; n.clear()
    out=n.new('ShaderNodeOutputMaterial'); scatter=n.new('ShaderNodeVolumeScatter')
    scatter.inputs['Anisotropy'].default_value=.55
    tex=n.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value=7; tex.inputs['Detail'].default_value=3
    density=_math(n,l,'MULTIPLY',_math(n,l,'ADD',_math(n,l,'MULTIPLY',tex.outputs['Fac'],.7),.25),haze)
    l.new(density,scatter.inputs['Density']); l.new(scatter.outputs[0],out.inputs['Volume'])
    rounded_box('Atmospheric scattering region',(0,0,8),(64,70,24),haze_mat,0)
    shell=_material('Fixture black anodised aluminium',(.006,.007,.009),roughness=.30,metallic=.78,texture='metal')
    cloth=_material('Matte woven projection fabric',(.64,.65,.61),roughness=.89,texture='cloth')
    fixtures=[]; pidx=0; gidx=0
    for i,f in enumerate(patch['fixtures']):
        kind=f['kind']; name=f['name']; pos=f['position']; target=f['target']
        ref={'index':i,'kind':kind,'fixture':f}
        if kind in ('moving_head','wash','blinder','strobe'):
            optical=spot(name+' · optical output',pos,target)
            body,lens=_fixture_body(name,optical,shell)
            ref.update(light=optical,mount=body,lens=lens)
        elif kind=='projector':
            optical=spot(name+' · projection origin',pos,target)
            optical.data.shadow_soft_size=.002; optical.data.spot_blend=0
            projector_shader(optical)
            rounded_box(name+' · housing',(0,0,.36),(.64,.42,.62),shell,parent=optical)
            size=[(11,7),(4,5),(4,5),(6,1.6)][pidx%4]
            receiver=screen(name+' · receiving surface',target,*size,cloth,sail=pidx%4 in (1,2))
            solid=receiver.modifiers.new('Thin projection scrim','SOLIDIFY'); solid.thickness=.0015
            ref.update(projector=optical,light=optical,surface=receiver,base_rotation=optical.rotation_euler.copy())
            pidx+=1
        elif kind=='laser':
            # Submillimetre virtual scan rays; reduced ray count and no diffuse
            # reflection avoids the glowing solid rods of the former preview.
            mat=bpy.data.materials.new(name+' · virtual laser rays'); mat.use_nodes=True; mat['dmx_fixture']=name
            n=mat.node_tree.nodes; n.clear(); e=n.new('ShaderNodeEmission'); e.name='DMX Emission'
            e.inputs['Strength'].default_value=0
            out=n.new('ShaderNodeOutputMaterial'); mat.node_tree.links.new(e.outputs[0],out.inputs['Surface'])
            lines=[]
            for j in range(7):
                a=(j-3)*.043
                lines.append([(0,0,0),(math.sin(a)*27,math.cos(a)*27,0)])
            fan=curves(name+' · scanned laser fan',lines,mat,.00075)
            fan.location=pos; fan.visible_shadow=False; fan.visible_diffuse=False; fan.visible_glossy=False
            rounded_box(name+' · housing',pos,(.34,.38,.17),shell)
            ref.update(fan=fan,material=mat)
        elif kind=='garland':
            mat=env['garland_materials'][gidx]; mat['dmx_fixture']=name
            ref.update(material=mat); gidx+=1
        else:
            raise ValueError('Unsupported fixture kind: '+kind)
        fixtures.append(ref)
    cd=bpy.data.cameras.new('Photographic festival camera'); camera=bpy.data.objects.new(cd.name,cd)
    bpy.context.collection.objects.link(camera); scene.camera=camera
    cd.lens=29; cd.sensor_width=36; cd.clip_start=.10; cd.clip_end=200
    cd.dof.use_dof=True; cd.dof.aperture_fstop=3.5; cd.dof.aperture_blades=9
    scene.use_nodes=True
    n=scene.node_tree.nodes; n.clear(); l=scene.node_tree.links
    rl=n.new('CompositorNodeRLayers'); glow=n.new('CompositorNodeGlare')
    glow.glare_type='FOG_GLOW'; glow.quality='HIGH'; glow.threshold=2.5; glow.size=7; glow.mix=-.9
    out=n.new('CompositorNodeComposite'); l.new(rl.outputs['Image'],glow.inputs['Image']); l.new(glow.outputs[0],out.inputs[0])
    scene['LightSem scene module']=__name__
    scene['Photographic profile']='Empty venue; zero world/ambient/static emission; Cycles OptiX; shadowed projected light'
    scene['Projection model']='Cycles textured spot lights cast actual patterned illumination and occlusion onto passive cloth'
    scene['LightSem connection']='Authenticated DMX -> configurable venue patch -> physically shaded optical scene'
    return {'scene':scene,'fixtures':fixtures,'environment':env,'camera':camera,'apply_dmx':apply_dmx}


def apply_dmx(runtime,controls,t,frame=1,bake=False,blackout=False):
    for ref in runtime['fixtures']:
        i=ref['index']; f=ref['fixture']; kind=ref['kind']
        dim=0. if blackout else float(controls['dimmer'][i])
        rgb=tuple(float(x)/12.92 if x<=.04045 else ((float(x)+.055)/1.055)**2.4 for x in controls['rgb'][i])
        pan=float(controls['pan'][i]); tilt=float(controls['tilt'][i]); zoom=float(controls['zoom'][i])
        shutter=float(controls['strobe'][i])
        if shutter>0 and (t*(1+19*shutter))%1>.5:dim=0.
        if kind in ('moving_head','wash','blinder','strobe','projector'):
            obj=ref['light']; data=obj.data
            angle=f['beam_min_deg']+zoom*(f['beam_max_deg']-f['beam_min_deg'])
            data.spot_size=math.radians(angle)
            fraction=max((1-math.cos(data.spot_size/2))/2,1e-6)
            power={'moving_head':30.,'wash':65.,'projector':4000.,'blinder':40.,'strobe':40.}[kind]
            data.energy=dim*float(f.get('photographic_flux_watts',power))/fraction
            data.color=(1,1,1) if kind=='projector' else rgb
            if kind=='moving_head':
                yaw=math.radians((pan-.5)*f['pan_range_deg']+float(f.get('mount_yaw_deg',0)))
                elev=math.radians((tilt-.5)*f['tilt_range_deg'])
                obj.rotation_euler=Vector((math.sin(yaw)*math.cos(elev),math.cos(yaw)*math.cos(elev),math.sin(elev))).to_track_quat('-Z','Y').to_euler()
                _key(obj,'rotation_euler',frame,bake)
            for prop in ('energy','color','spot_size'):_key(data,prop,frame,bake)
            if kind=='projector':
                n=data.node_tree.nodes
                n['DMX Color'].outputs[0].default_value=(*rgb,1)
                n['DMX Zoom'].outputs[0].default_value=1/math.tan(math.radians(angle)/2)
                n['DMX Pattern'].outputs[0].default_value=8+int(controls['gobo'][i])%8
                n['Media Clock'].outputs[0].default_value=t*.22
                for name in ('DMX Color','DMX Zoom','DMX Pattern','Media Clock'):_key(n[name].outputs[0],'default_value',frame,bake)
                rot=ref['base_rotation'].copy(); rot.x+=(tilt-.5)*.5; rot.z+=(pan-.5)*.5
                obj.rotation_euler=rot; _key(obj,'rotation_euler',frame,bake)
            elif 'lens' in ref:
                p=ref['lens'].node_tree.nodes['Principled BSDF']
                p.inputs['Emission Color'].default_value=(*rgb,1)
                p.inputs['Emission Strength'].default_value=dim*6
                for name in ('Emission Color','Emission Strength'):_key(p.inputs[name],'default_value',frame,bake)
        elif kind=='garland':
            p=ref['material'].node_tree.nodes['Principled BSDF']
            p.inputs['Emission Color'].default_value=(*rgb,1)
            p.inputs['Emission Strength'].default_value=dim*45
            for name in ('Emission Color','Emission Strength'):_key(p.inputs[name],'default_value',frame,bake)
        elif kind=='laser':
            e=ref['material'].node_tree.nodes['DMX Emission']
            e.inputs['Color'].default_value=(*rgb,1); e.inputs['Strength'].default_value=dim*4
            for name in ('Color','Strength'):_key(e.inputs[name],'default_value',frame,bake)
            delta=Vector(f['target'])-Vector(f['position'])
            yaw=math.atan2(-delta.x,delta.y)+math.radians((pan-.5)*f['pan_range_deg'])
            elevation=math.atan2(delta.z,math.hypot(delta.x,delta.y))+math.radians((tilt-.5)*f['tilt_range_deg'])
            fan=ref['fan']; fan.rotation_euler=(elevation,0,yaw); fan.scale.x=.55+zoom*.8
            fan.hide_render=dim<=0; fan.hide_viewport=dim<=0
            for prop in ('rotation_euler','scale','hide_render','hide_viewport'):_key(fan,prop,frame,bake)


def set_camera(runtime,elapsed,duration,shot='journey',frame=1,bake=False):
    camera=runtime['camera']; u=min(1,max(0,elapsed/max(duration,1e-6)))
    s=u*u*(3-2*u)
    if shot=='wide':pos=(12,24,8); target=(0,-3,6)
    elif shot=='floor':pos=(1.8,18,2.35); target=(0,-5,6)
    else:pos=(2.4-2.8*s,18-.8*s,2.4); target=(0,-6,5.8)
    camera.location=pos; point_at(camera,target)
    camera.data.dof.focus_distance=(Vector(target)-camera.location).length
    for prop in ('location','rotation_euler'):_key(camera,prop,frame,bake)
    _key(camera.data.dof,'focus_distance',frame,bake)


def recover_runtime(patch):
    refs=[]; gi=0
    for i,f in enumerate(patch['fixtures']):
        name=f['name']; kind=f['kind']; ref={'index':i,'kind':kind,'fixture':f}
        if kind in ('moving_head','wash','blinder','strobe'):
            ref.update(light=bpy.data.objects[name+' · optical output'],mount=bpy.data.objects[name+' · housing'],lens=bpy.data.materials[name+' · lens'])
        elif kind=='projector':
            obj=bpy.data.objects[name+' · projection origin']
            base=(Vector(f['target'])-Vector(f['position'])).to_track_quat('-Z','Y').to_euler()
            ref.update(light=obj,projector=obj,surface=bpy.data.objects[name+' · receiving surface'],base_rotation=base)
        elif kind=='laser':
            ref.update(fan=bpy.data.objects[name+' · scanned laser fan'],material=bpy.data.materials[name+' · virtual laser rays'])
        elif kind=='garland':
            ref['material']=bpy.data.materials[f'Garland_{gi+1:02d}']; gi+=1
        refs.append(ref)
    return {'scene':bpy.context.scene,'fixtures':refs,'camera':bpy.context.scene.camera,'apply_dmx':apply_dmx}
