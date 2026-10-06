/** CPU-only contracts. No Chrome, GPU, or real-campaign media execution. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import vm from 'node:vm';
import {parseArguments,chromeArguments,validateManifest,authenticateInputs,digest,
  snapshotExpression,validateDOM,assertProgress,samplePlayback} from '../scripts/rendering/verify_dynamic_combo_gallery.mjs';

function fixture(){
  const tracks=[{id:'shpongle',duration_s:277.3066666666667,video_frames:8320},{id:'cure_for_me',duration_s:201.64,video_frames:6050}];
  const plan={tracks,algorithms:[{id:'C3'},{id:'C4'}],files:{}};
  const manifest={plan_sha256:'a'.repeat(64),tracks,algorithms:plan.algorithms,baseline_promoted:false,reference:'D7_reference',source_start_s:0,
    variants:['C3','C4'].flatMap(method=>tracks.map(track=>({algorithm_id:method,track_id:track.id,status:'complete',
      duration_s:track.video_frames/30,source_duration_s:track.duration_s,title:method+' / '+track.id,
      banner:'FAILURE DIAGNOSTIC | original failure',sha256:'b'.repeat(64),comparison:{physical_engineering_status:'fail'},
      download_name:method+'_'+track.id+'_full_1080p.mp4'}))).map((r,i)=>({...r,id:String(i+1).padStart(2,'0')}))};
  return {plan,manifest};
}

test('explicit loopback input and software-only owned browser configuration',()=>{
  const args=parseArguments(['--root','/tmp/fixture','--url','http://127.0.0.1:8776','--plan-sha256','a'.repeat(64)]);
  assert.equal(args.timeoutMs,30000);
  assert.throws(()=>parseArguments(['--root','/tmp/x','--url','http://example.com','--plan-sha256','a'.repeat(64)]),/loopback/);
  const flags=chromeArguments('/tmp/owned');
  for(const flag of ['--disable-gpu','--disable-accelerated-video-decode','--use-angle=swiftshader','--user-data-dir=/tmp/owned'])assert(flags.includes(flag));
});

test('four strict identities preserve full source durations and forbid baseline promotion',()=>{
  const {plan,manifest}=fixture();validateManifest(manifest,plan,manifest.plan_sha256);
  assert.equal(manifest.variants[0].duration_s,8320/30);assert.equal(manifest.variants[1].duration_s,6050/30);
  manifest.baseline_promoted=true;assert.throws(()=>validateManifest(manifest,plan,manifest.plan_sha256),/Baseline/);
  manifest.baseline_promoted=false;manifest.variants[1].duration_s=120;assert.throws(()=>validateManifest(manifest,plan,manifest.plan_sha256),/cropped/);
  manifest.variants[1].duration_s=6050/30;manifest.variants[2].track_id='cure_for_me';assert.throws(()=>validateManifest(manifest,plan,manifest.plan_sha256),/identity/);
});

test('light snapshots never inspect card prose, lazy evidence, review notes or growing media log',()=>{
  const card={id:'variant-01',classList:{contains:()=>false},get textContent(){throw Error('card prose');},querySelector:s=>{
    if(s!=='video')throw Error('hidden DOM queried '+s);
    return {getAttribute:()=>'/media/01.mp4',paused:false,currentTime:12,duration:8320/30,readyState:4,seeking:false,muted:true,error:null,getVideoPlaybackQuality:()=>({totalVideoFrames:40})};}};
  const context={performance:{timeOrigin:1},masterId:'01',desiredPlaying:true,
    document:{querySelectorAll:()=>[card],getElementById:()=>({value:'12',max:String(8320/30),textContent:'ready'})},
    window:{__comboSeekingCounts:{'01':1},get __comboMediaEvents(){throw Error('growing events queried');}}};
  const state=vm.runInNewContext(snapshotExpression(),context);
  assert.equal(state.masterId,'01');assert.equal(state.cards[0].video.duration,8320/30);
  assert(!('evidence'in state.cards[0]));assert(JSON.stringify(state).length<1500);
});

test('one-time DOM evidence and downloaded full-song identities match the manifest',()=>{
  const {manifest}=fixture();const state={cards:manifest.variants.map(row=>({id:row.id,title:row.algorithm_id+' | '+row.title,badge:row.banner,
    evidence:JSON.stringify({video_sha256:row.sha256,duration_s:row.duration_s,source_duration_s:row.source_duration_s,comparison:row.comparison}),
    video:{src:'/media/'+row.id+'.mp4?v='+row.sha256},links:[{href:'/media/'+row.id+'.mp4?download=1&v='+row.sha256,download:row.download_name}]}))};
  validateDOM(state,manifest);state.cards[2].badge='New accepted baseline';assert.throws(()=>validateDOM(state,manifest),/diagnostic/);
});

function state(time,frames,{duration=8320/30,seeking=false,counts=0,skew=.02,master='01',audible='01'}={}){
  return {masterId:master,seekCounts:{'03':counts},cards:['01','03'].map((id,i)=>({id,visible:true,video:{time:time+i*skew,frames,duration,seeking:i===1&&seeking,paused:false,ready:4,error:null,muted:id!==audible}}))};
}
function stream(states){let elapsed=0,count=0;return {snapshot:async()=>states[Math.min(count++,states.length-1)],timing:{clock:()=>elapsed,sleep:async ms=>elapsed+=ms},elapsed:()=>elapsed,reads:()=>count};}

test('same fixed progress/skew gates apply to variable full-song durations',()=>{
  for(const duration of [8320/30,6050/30]){
    const result=assertProgress(state(12,10,{duration}),state(12.7,30,{duration}),['01','03'],duration);
    assert.equal(result.limits.minimum_advance_s,.12);assert.equal(result.limits.maximum_skew_s,.35);
    assert.throws(()=>assertProgress(state(12,10,{duration}),state(12.7,30,{duration:120}),['01','03'],duration),/duration/);
    assert.throws(()=>assertProgress(state(12,10,{duration}),state(12.7,30,{duration,skew:.5}),['01','03'],duration),/skew/);
  }
});

test('audio-selected master can change while settled playback remains measured, never repaired',async()=>{
  const s=stream([state(12,10,{seeking:true,counts:1,master:'03',audible:'03'}),state(12.075,13,{counts:1,master:'03',audible:'03'}),
    state(12.775,35,{seeking:true,counts:2,master:'03',audible:'03'}),state(12.85,38,{counts:2,master:'03',audible:'03'})]),telemetry={};
  const sample=await samplePlayback(s.snapshot,['01','03'],8320/30,2000,telemetry,s.timing);
  assert.equal(sample.after.masterId,'03');assert.equal(telemetry.requested_sample_ms,700);assert.equal(telemetry.actual_sample_ms,775);
  assert.equal(telemetry.total_settling_ms,150);assert.equal(telemetry.seek_event_deltas['03'],1);
});

test('real stutter is not retried; persistent seek exhausts the original deadline',async()=>{
  let s=stream([state(12,10),state(12.05,20)]),telemetry={};
  await assert.rejects(samplePlayback(s.snapshot,['01','03'],8320/30,2000,telemetry,s.timing),/did not advance/);
  assert.equal(s.reads(),2);assert.equal(telemetry.status,'failed');
  s=stream([state(12,10),state(12.7,30,{seeking:true,counts:1})]);telemetry={};
  await assert.rejects(samplePlayback(s.snapshot,['01','03'],8320/30,900,telemetry,s.timing),/deadline/);
  assert.equal(s.elapsed(),900);assert.equal(telemetry.after.seeking_poll_observations['03'],3);
});

test('all four media/audit/audio identities authenticate without accepting a reused short video',async()=>{
  const root=await fs.mkdtemp(path.join(tmpdir(),'combo-gallery-unit-'));
  try{
    const {plan,manifest}=fixture();await fs.mkdir(path.join(root,'review'));
    const html=path.join(root,'review/index.html');await fs.writeFile(html,'unit-test contract, never delivery media');
    for(const track of plan.tracks){track.audio=path.join(root,track.id+'.wav');await fs.writeFile(track.audio,'unit source '+track.id);plan.files[track.audio]=await digest(track.audio);}
    const planPath=path.join(root,'PLAN.json');await fs.writeFile(planPath,JSON.stringify(plan));manifest.plan_sha256=await digest(planPath);
    for(const track of plan.tracks){const dir=path.join(root,'audio',track.id);await fs.mkdir(dir,{recursive:true});const reference=path.join(dir,'reference.m4a');await fs.writeFile(reference,'unit AAC reference '+track.id);
      const duration=track.video_frames/30,record={plan_sha256:manifest.plan_sha256,source_audio:track.audio,source_audio_sha256:plan.files[track.audio],source_start_s:0,playback_rate:1,duration_s:duration,end_padding_s:duration-track.duration_s,reference,reference_sha256:await digest(reference),decoded_pcm_sha256:'c'.repeat(64)};
      await fs.writeFile(path.join(dir,'REFERENCE.json'),JSON.stringify(record));}
    for(const row of manifest.variants){const track=plan.tracks.find(t=>t.id===row.track_id),dir=path.join(root,'media',row.algorithm_id,row.track_id);await fs.mkdir(dir,{recursive:true});
      row.path=path.join(dir,'full.mp4');await fs.writeFile(row.path,'Unit bytes only, not a delivery verdict '+row.id);row.sha256=await digest(row.path);row.bytes=(await fs.stat(row.path)).size;
      row.inputs={template_sha256:await digest(html),audio_record_sha256:await digest(path.join(root,'audio',row.track_id,'REFERENCE.json'))};
      const audit={video_sha256:row.sha256,decoded_frames:track.video_frames,pts_uniform:true,full_software_AV_decode:true,audio_matches_full_reference:true,source_start_s:0,audio_speed:1,source_duration_s:track.duration_s,video_duration_s:row.duration_s,pcm_sha256:'c'.repeat(64),inputs:row.inputs};
      const auditPath=path.join(dir,'full_audit.json');await fs.writeFile(auditPath,JSON.stringify(audit));
      await fs.writeFile(path.join(dir,'full_verified.json'),JSON.stringify({video_sha256:row.sha256,complete_full_song:true,source_audio_unchanged_speed:true,audit_sha256:await digest(auditPath),inputs:row.inputs}));
      await fs.writeFile(path.join(dir,'full_render.json'),JSON.stringify({sha256:row.sha256,frames:track.video_frames,start_s:0,audio_speed:1,duration_s:row.duration_s,inputs:row.inputs}));
    }
    await fs.writeFile(path.join(root,'review/MANIFEST.json'),JSON.stringify(manifest));
    const result=await authenticateInputs(root,manifest.plan_sha256);assert.equal(result.manifest.variants.length,4);
    await fs.writeFile(manifest.variants[3].path,'old 120-second substituted video');
    await assert.rejects(authenticateInputs(root,manifest.plan_sha256),/SHA256 mismatch/);
  }finally{await fs.rm(root,{recursive:true,force:true});}
});
