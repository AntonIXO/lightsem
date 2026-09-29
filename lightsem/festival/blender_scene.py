"""Blender scene assembly and DMX application (run inside Blender Python)."""
from __future__ import annotations

import math
import bpy
from mathutils import Vector

from lightsem.festival.geometry import build_environment


def material(name, color, emission=0., metallic=0.):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Metallic'].default_value = metallic
    p.inputs['Roughness'].default_value = .32
    p.inputs['Emission Color'].default_value = (*color, 1)
    p.inputs['Emission Strength'].default_value = emission
    return m


def cube(name, pos, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    o = bpy.context.object
    o.name = name
    o.scale = size
    o.data.materials.append(mat)
    return o


def point_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def light(name, position, target, kind='SPOT', energy=0, color=(1,1,1)):
    d = bpy.data.lights.new(name, kind)
    d.energy, d.color = energy, color
    if kind == 'SPOT':
        d.spot_size = math.radians(14)
        d.spot_blend = .24
        d.shadow_soft_size = .07
    d.use_shadow = False
    o = bpy.data.objects.new(name, d)
    bpy.context.collection.objects.link(o)
    o.location = position
    point_at(o, target)
    return o


def curves(name, lines, mat, radius=.012):
    d = bpy.data.curves.new(name, 'CURVE')
    d.dimensions = '3D'
    d.resolution_u = 1
    d.bevel_depth = radius
    d.bevel_resolution = 0
    for line in lines:
        s = d.splines.new('POLY')
        s.points.add(len(line)-1)
        for p, xyz in zip(s.points, line):
            p.co = (*xyz, 1)
    o = bpy.data.objects.new(name, d)
    bpy.context.collection.objects.link(o)
    d.materials.append(mat)
    return o


def _math(nodes, links, operation, a, b=None, name=None):
    n = nodes.new('ShaderNodeMath')
    n.operation = operation
    if name:
        n.name = name
    for idx, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[idx].default_value = v
        else:
            links.new(v, n.inputs[idx])
    return n.outputs[0]


def projection_material(name, projector, aspect):
    """Projective receiver shader, with an explicit virtual-media clock.

    Coordinates come from projector space (x/-z, y/-z), so projector aiming
    and zoom change the image footprint. This approximates mapped projection;
    it is not a ray traced occlusion/photometry model.
    """
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    n.clear()
    out = n.new('ShaderNodeOutputMaterial')
    bsdf = n.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (.012,.016,.025,1)
    bsdf.inputs['Roughness'].default_value = .85
    l.new(bsdf.outputs[0], out.inputs['Surface'])
    tc = n.new('ShaderNodeTexCoord')
    tc.object = projector
    sep = n.new('ShaderNodeSeparateXYZ')
    l.new(tc.outputs['Object'], sep.inputs[0])
    depth = _math(n,l,'MULTIPLY',sep.outputs['Z'],-1)
    x = _math(n,l,'DIVIDE',sep.outputs['X'],depth)
    y = _math(n,l,'DIVIDE',sep.outputs['Y'],depth)
    zoom = n.new('ShaderNodeValue'); zoom.name = 'DMX Zoom'; zoom.outputs[0].default_value = 1
    x = _math(n,l,'MULTIPLY',x,zoom.outputs[0])
    y = _math(n,l,'MULTIPLY',y,zoom.outputs[0])
    xx = _math(n,l,'MULTIPLY',x,x)
    yy = _math(n,l,'MULTIPLY',y,y)
    radius = _math(n,l,'SQRT',_math(n,l,'ADD',xx,yy))
    angle = _math(n,l,'ARCTAN2',y,x)
    clock = n.new('ShaderNodeValue'); clock.name = 'Media Clock'
    gobo = n.new('ShaderNodeValue'); gobo.name = 'DMX Pattern'; gobo.outputs[0].default_value=10
    petals = _math(n,l,'MULTIPLY',angle,gobo.outputs[0])
    petals = _math(n,l,'ADD',petals,clock.outputs[0])
    petals = _math(n,l,'SINE',petals)
    petals = _math(n,l,'MULTIPLY',petals,.5)
    rings = _math(n,l,'MULTIPLY',radius,55)
    rings = _math(n,l,'ADD',rings,petals)
    rings = _math(n,l,'SUBTRACT',rings,clock.outputs[0])
    rings = _math(n,l,'ABSOLUTE',_math(n,l,'SINE',rings))
    rings = _math(n,l,'POWER',rings,8)
    lace = _math(n,l,'ABSOLUTE',_math(n,l,'SINE',_math(n,l,'ADD',_math(n,l,'MULTIPLY',angle,20),_math(n,l,'MULTIPLY',radius,35))))
    lace = _math(n,l,'POWER',lace,14)
    pattern = _math(n,l,'ADD',_math(n,l,'MULTIPLY',rings,.8),_math(n,l,'MULTIPLY',lace,.35))
    # Rectangular projector gate; no image outside the forward frustum.
    gate_x = _math(n,l,'LESS_THAN',_math(n,l,'ABSOLUTE',x),.83)
    gate_y = _math(n,l,'LESS_THAN',_math(n,l,'ABSOLUTE',y),.83/aspect)
    gate = _math(n,l,'MULTIPLY',gate_x,gate_y)
    gate = _math(n,l,'MULTIPLY',gate,_math(n,l,'GREATER_THAN',depth,0))
    pattern = _math(n,l,'MULTIPLY',pattern,gate)
    rgb = n.new('ShaderNodeRGB'); rgb.name = 'DMX Color'
    # The media is a coloured mandala, globally tinted by the DMX RGB channels.
    # Its local image colours are content, not extra learned fixture decisions.
    ramp=n.new('ShaderNodeValToRGB'); ramp.name='Procedural media palette'
    l.new(_math(n,l,'PINGPONG',_math(n,l,'ADD',_math(n,l,'MULTIPLY',radius,3),_math(n,l,'MULTIPLY',clock.outputs[0],.08)),1),ramp.inputs[0])
    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[1])
    for j,(position,color) in enumerate(((0,(.3,.015,.8,1)),(.32,(.005,.8,1,1)),(.64,(1,.015,.27,1)),(1,(1,.55,.06,1)))):
        e=ramp.color_ramp.elements[0] if j==0 else ramp.color_ramp.elements.new(position)
        e.position=position; e.color=color
    mix=n.new('ShaderNodeMixRGB'); mix.blend_type='MULTIPLY'; mix.inputs[0].default_value=1
    l.new(ramp.outputs[0],mix.inputs[1]); l.new(rgb.outputs[0],mix.inputs[2])
    l.new(mix.outputs[0],bsdf.inputs['Emission Color'])
    dim = n.new('ShaderNodeValue'); dim.name='DMX Dimmer'
    strength = _math(n,l,'MULTIPLY',_math(n,l,'MULTIPLY',_math(n,l,'ADD',pattern,.035),dim.outputs[0]),gate)
    l.new(strength,bsdf.inputs['Emission Strength'])
    return m


def screen(name, target, width, height, mat, sail=False):
    x,y,z = target
    if sail:
        verts=[(x-width/2,y,z-height/2),(x+width/2,y,z-height/2),(x+width*.36,y-.18,z+height/2),(x-width*.36,y-.18,z+height/2),(x,y+.2,z)]
        faces=[(0,1,4),(1,2,4),(2,3,4),(3,0,4)]
    else:
        verts=[(x-width/2,y,z-height/2),(x+width/2,y,z-height/2),(x+width/2,y,z+height/2),(x-width/2,y,z+height/2)]
        faces=[(0,1,2,3)]
    mesh=bpy.data.meshes.new(name); mesh.from_pydata(verts,[],[tuple(reversed(f)) for f in faces]); mesh.update()
    o=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(o)
    mesh.materials.append(mat)
    return o


def build_scene(patch, width=1920, height=1080, samples=24, haze=.008):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.engine='BLENDER_EEVEE_NEXT'
    scene.render.resolution_x=width; scene.render.resolution_y=height
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False
    scene.render.fps=30
    scene.eevee.taa_render_samples=samples
    scene.eevee.volumetric_samples=48
    scene.eevee.volumetric_tile_size='8'
    scene.eevee.use_volumetric_shadows=False
    scene.world=bpy.data.worlds.new('Moonless festival sky')
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.055,.075,.14,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.20
    scene.view_settings.view_transform='AgX'
    scene.view_settings.look='AgX - Medium High Contrast'
    scene.view_settings.exposure=.7
    env=build_environment(garland_count=sum(f['kind']=='garland' for f in patch['fixtures']),
                          fixture_mounts=[f['position'] for f in patch['fixtures'] if f['kind']=='moving_head'],
                          garland_routes=[f for f in patch['fixtures'] if f['kind']=='garland'])
    haze_mat=bpy.data.materials.new('Bounded atmospheric haze'); haze_mat.use_nodes=True
    hn=haze_mat.node_tree.nodes; hn.clear()
    vo=hn.new('ShaderNodeVolumeScatter'); vo.inputs['Density'].default_value=haze
    vo.inputs['Anisotropy'].default_value=.3
    out=hn.new('ShaderNodeOutputMaterial'); haze_mat.node_tree.links.new(vo.outputs[0],out.inputs['Volume'])
    cube('Haze volume',(0,0,8),(62,72,24),haze_mat)
    # A low fixed moon/environment contribution lets unlit physical structure read.
    moon=light('Moon rim',(0,0,30),(4,-6,0),'AREA',energy=1700,color=(.19,.30,.6))
    moon.data.shape='DISK'; moon.data.size=35
    body_mat=material('Anodised fixture housing',(.022,.027,.035),metallic=.65)
    fixtures=[]
    pidx=0; gidx=0
    for idx,f in enumerate(patch['fixtures']):
        kind=f['kind']; name=f['name']; pos=f['position']
        target=f.get('target',(0,5,0))
        ref={'fixture':f,'index':idx,'kind':kind}
        mount=cube(name+' · housing',pos,(.36,.32,.3),body_mat)
        mount['DMX universe']=f['universe']; mount['DMX address']=f['address']
        mount['LightSem source']=f['source_fixture']; mount['Fixture kind']=kind
        if kind in ('moving_head','wash'):
            spot=light(name+' · optical output',pos,target)
            lens=material(name+' · lens',(1,1,1))
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=6,radius=.12)
            bulb=bpy.context.object; bulb.name=name+' · lens'; bulb.parent=spot; bulb.location=(0,0,-.18)
            bulb.scale=(1,1,.35); bulb.data.materials.append(lens)
            ref.update(light=spot,lens=lens,mount=mount)
        elif kind=='projector':
            projector=bpy.data.objects.new(name+' · projection origin',None); bpy.context.collection.objects.link(projector)
            projector.location=pos; point_at(projector,target)
            mat=projection_material(name+' · mapped media',projector,1.65)
            size=[(11,7),(4,5),(4,5),(6,1.6)][pidx%4]
            surface=screen(name+' · receiving surface',target,*size,mat,sail=pidx%4 in (1,2))
            ref.update(projector=projector,material=mat,surface=surface,base_rotation=projector.rotation_euler.copy())
            pidx+=1
        elif kind in ('laser','virtual_laser'):
            mat=material(name+' · virtual laser rays',(.1,1,.6),emission=0)
            lines=[]
            for j in range(15):
                a=(j-7)*.055
                lines.append([(0,0,0),(math.sin(a)*30,math.cos(a)*30,0)])
            fan=curves(name+' · scanned laser fan',lines,mat,.005)
            fan.location=pos
            ref.update(fan=fan,material=mat)
        elif kind=='garland':
            mats=env['garland_materials']
            ref['material']=mats[gidx]; gidx+=1
        elif kind in ('blinder','strobe'):
            spot=light(name+' · optical output',pos,target)
            ref.update(light=spot,mount=mount)
        else:
            raise ValueError('No Blender fixture renderer for '+kind)
        fixtures.append(ref)
    camera_data=bpy.data.cameras.new('Festival camera'); camera=bpy.data.objects.new('Festival camera',camera_data)
    bpy.context.collection.objects.link(camera); scene.camera=camera
    camera.data.lens=25; camera.data.clip_end=200
    scene.use_nodes=True
    nodes=scene.node_tree.nodes; nodes.clear()
    rl=nodes.new('CompositorNodeRLayers')
    glow=nodes.new('CompositorNodeGlare'); glow.glare_type='FOG_GLOW'; glow.quality='MEDIUM'; glow.threshold=1.4; glow.size=8
    output=nodes.new('CompositorNodeComposite')
    scene.node_tree.links.new(rl.outputs['Image'],glow.inputs['Image'])
    scene.node_tree.links.new(glow.outputs['Image'],output.inputs[0])
    scene['LightSem connection']='Source DMX -> configurable fixture patch -> decoded DMX -> optical scene'
    scene['Projection model']='Projector-space procedural media on assigned receivers; no occlusion/photometric calibration'
    return {'scene':scene,'fixtures':fixtures,'environment':env,'camera':camera}


def _key(obj, path, frame, bake):
    if bake:
        obj.keyframe_insert(data_path=path,frame=frame)


def apply_dmx(runtime, controls, t, frame=1, bake=False, blackout=False):
    """Every controllable intensity has a true zero; no procedural music driver."""
    for ref in runtime['fixtures']:
        i=ref['index']; kind=ref['kind']
        dim=0. if blackout else float(controls['dimmer'][i])
        # Treat RGB DMX as display-referred sRGB and feed Blender linear light.
        # This prevents every saturated controller colour washing toward white.
        rgb=tuple(float(x)/12.92 if x<=.04045 else ((float(x)+.055)/1.055)**2.4 for x in controls['rgb'][i])
        pan=float(controls['pan'][i]); tilt=float(controls['tilt'][i]); zoom=float(controls['zoom'][i])
        strobe=float(controls['strobe'][i])
        # Open shutter at zero; otherwise an explicitly simulated shutter at source time.
        if strobe>0 and ((t*(1+19*strobe))%1)>.5:
            dim=0
        if kind in ('moving_head','wash','blinder','strobe'):
            o=ref['light']; d=o.data
            d.color=rgb
            f=ref['fixture']
            d.spot_size=math.radians(f['beam_min_deg']+zoom*(f['beam_max_deg']-f['beam_min_deg']))
            # Blender's spot wattage is specified before cone restriction. Keep
            # approximately constant in-cone flux as zoom changes; the soft edge
            # is not a manufacturer photometric calibration.
            if kind=='moving_head':
                fraction=(1-math.cos(d.spot_size/2))/2
                d.energy=dim*float(f.get('beam_flux_watts',100))/max(fraction,1e-6)
            else:
                d.energy=dim*float(f.get('power_watts',10500))
            d.volume_factor=1
            if kind=='moving_head':
                yaw=(pan-.5)*math.radians(f['pan_range_deg'])
                elevation=(tilt-.5)*math.radians(f['tilt_range_deg'])
                # Patch installation yaw is physical orientation, not new model output.
                yaw+=math.radians(float(ref['fixture'].get('mount_yaw_deg',0)))
                direction=Vector((math.sin(yaw)*math.cos(elevation),math.cos(yaw)*math.cos(elevation),math.sin(elevation)))
                o.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
                ref['mount'].rotation_euler=o.rotation_euler
                _key(o,'rotation_euler',frame,bake); _key(ref['mount'],'rotation_euler',frame,bake)
            for prop in ('energy','color','spot_size'):
                _key(d,prop,frame,bake)
            if 'lens' in ref:
                p=ref['lens'].node_tree.nodes['Principled BSDF']
                p.inputs['Emission Color'].default_value=(*rgb,1)
                p.inputs['Emission Strength'].default_value=dim*15
                _key(p.inputs['Emission Color'],'default_value',frame,bake)
                _key(p.inputs['Emission Strength'],'default_value',frame,bake)
        elif kind=='projector':
            n=ref['material'].node_tree.nodes
            n['DMX Color'].outputs[0].default_value=(*rgb,1)
            n['DMX Dimmer'].outputs[0].default_value=dim*22
            f=ref['fixture']
            angle=f['beam_min_deg']+zoom*(f['beam_max_deg']-f['beam_min_deg'])
            n['DMX Zoom'].outputs[0].default_value=.83/math.tan(math.radians(angle)/2)
            n['DMX Pattern'].outputs[0].default_value=8+int(controls['gobo'][i])%8
            n['Media Clock'].outputs[0].default_value=t*.38
            for name in ('DMX Color','DMX Dimmer','DMX Zoom','DMX Pattern','Media Clock'):
                _key(n[name].outputs[0],'default_value',frame,bake)
            rot=ref['base_rotation'].copy(); rot.x+=(tilt-.5)*.5; rot.z+=(pan-.5)*.5
            ref['projector'].rotation_euler=rot; _key(ref['projector'],'rotation_euler',frame,bake)
        elif kind in ('laser','virtual_laser','garland'):
            p=ref['material'].node_tree.nodes.get('Principled BSDF')
            p.inputs['Emission Color'].default_value=(*rgb,1)
            p.inputs['Emission Strength'].default_value=dim*(3 if kind!='garland' else 18)
            _key(p.inputs['Emission Color'],'default_value',frame,bake)
            _key(p.inputs['Emission Strength'],'default_value',frame,bake)
            if kind!='garland':
                f=ref['fixture']; delta=Vector(f['target'])-Vector(f['position'])
                yaw=math.atan2(-delta.x,delta.y)+(pan-.5)*math.radians(f['pan_range_deg'])
                elevation=math.atan2(delta.z,math.hypot(delta.x,delta.y))+(tilt-.5)*math.radians(f['tilt_range_deg'])
                fan=ref['fan']; fan.rotation_euler=(elevation,0,yaw)
                fan.scale.x=.5+zoom*1.4
                # A laser ray is light, not a rod reflecting the moon when off.
                fan.hide_render=dim<=0
                fan.hide_viewport=dim<=0
                for prop in ('rotation_euler','scale','hide_render','hide_viewport'):
                    _key(fan,prop,frame,bake)


def set_camera(runtime, elapsed, duration, shot='journey', frame=1, bake=False):
    c=runtime['camera']
    u=min(1.,max(0.,elapsed/max(duration,1e-9)))
    if shot=='wide':
        pos=(22,30,16); target=(0,0,5)
    elif shot=='floor':
        pos=(1.5,18,3.5); target=(0,-9,6)
    else:
        # Continuous dolly under the roof keeps optical motion visible.
        s=u*u*(3-2*u)
        pos=(3-6*s,18-1.5*s,4.4-.8*s); target=(0,-8,6.1)
    c.location=pos; point_at(c,target)
    _key(c,'location',frame,bake); _key(c,'rotation_euler',frame,bake)


def recover_runtime(patch):
    """Reconnect an existing saved scene without rebuilding any geometry."""
    refs=[]; gidx=0
    for i,f in enumerate(patch['fixtures']):
        name=f['name']; kind=f['kind']
        ref={'fixture':f,'index':i,'kind':kind}
        if kind in ('moving_head','wash','blinder','strobe'):
            ref.update(light=bpy.data.objects[name+' · optical output'],mount=bpy.data.objects[name+' · housing'])
            if kind in ('moving_head','wash'):
                ref['lens']=bpy.data.materials[name+' · lens']
        elif kind=='projector':
            origin=bpy.data.objects[name+' · projection origin']
            # Recover installation rotation, not the current DMX-adjusted pose.
            base=(Vector(f['target'])-Vector(f['position'])).to_track_quat('-Z','Y').to_euler()
            ref.update(projector=origin,material=bpy.data.materials[name+' · mapped media'],
                       surface=bpy.data.objects[name+' · receiving surface'],base_rotation=base)
        elif kind in ('laser','virtual_laser'):
            ref.update(fan=bpy.data.objects[name+' · scanned laser fan'],material=bpy.data.materials[name+' · virtual laser rays'])
        elif kind=='garland':
            ref['material']=bpy.data.materials[f'Garland_{gidx+1:02d}']; gidx+=1
        else:
            raise ValueError('Unknown fixture kind: '+kind)
        refs.append(ref)
    return {'scene':bpy.context.scene,'fixtures':refs,'camera':bpy.context.scene.camera}
