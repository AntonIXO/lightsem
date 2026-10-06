/** Pure CPU contracts only. Never launch Chrome or use fake media for delivery. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import vm from 'node:vm';
import {authenticateInputs, chromeArguments, CDP, digest, parseArguments,
  validateDOMSnapshot, validateOrigin, assertPlaybackProgress, snapshotExpression,
  sampleSettledPlayback} from '../scripts/rendering/verify_dynamic_dance_gallery.mjs';

test('origins and revisions cannot escape local exclusive review', () => {
  assert.equal(validateOrigin('http://127.0.0.1:8775/'), 'http://127.0.0.1:8775');
  for (const value of ['https://example.com', 'http://127.0.0.1/private', 'http://user@localhost', 'http://localhost?file=secret']) {
    assert.throws(() => validateOrigin(value), /loopback/);
  }
  const good = ['--root', '/tmp/example', '--plan-sha256', 'a'.repeat(64)];
  assert.equal(parseArguments(good).timeoutMs, 30000);
  assert.throws(() => parseArguments([...good, '--revision', '../outside']), /Unsafe/);
  assert.throws(() => parseArguments([...good, '--timeout-ms', '600000']), /timeout/);
  assert.throws(() => parseArguments(['--root', '/tmp']), /externally frozen/);
});

test('owned Chrome always disables physical GPU decoding and has isolated profile', () => {
  const args = chromeArguments('/tmp/owned-only');
  for (const required of ['--disable-gpu', '--disable-accelerated-video-decode', '--disable-accelerated-video-encode',
    '--use-angle=swiftshader', '--user-data-dir=/tmp/owned-only', '--remote-debugging-address=127.0.0.1']) {
    assert(args.includes(required));
  }
  assert(!args.some(arg => arg === '--remote-debugging-address=0.0.0.0'));
});

function manifestFixture() {
  return {variants: Array.from({length: 16}, (_, i) => ({id:String(i+1).padStart(2,'0'),
    algorithm_id:`D${Math.floor(i/2)+1}`,track_id:i%2?'cure_for_me':'shpongle',status:'complete',
    title:'Real comparison '+i,sha256:String(i%10).repeat(64),comparison_sha256:'a'.repeat(64),
    banner:'FAILED ENGINEERING | diagnostic',physical_engineering_status:'fail',
    failure_notes:['physical check failed'],download_name:`D${i}_song_dark.mp4`}))};
}

function DOMFixture(manifest) {
  return {cards:manifest.variants.map(row=>({id:row.id,title:row.algorithm_id+' | '+row.title,badge:row.banner,
    text:'Physical engineering: fail. physical check failed',
    evidence:JSON.stringify({video_sha256:row.sha256,comparison_sha256:row.comparison_sha256}),
    links:[{href:'/media/'+row.id+'.mp4?download=1&v='+row.sha256,download:row.download_name}],
    video:{src:'/media/'+row.id+'.mp4?v='+row.sha256,error:null}}))};
}

test('DOM validates all sixteen evidence, failures, exact video identities and downloads', () => {
  const manifest=manifestFixture(), snapshot=DOMFixture(manifest);
  validateDOMSnapshot(snapshot,manifest);
  snapshot.cards[1].video.src='/media/01.mp4?v='+manifest.variants[0].sha256;
  assert.throws(()=>validateDOMSnapshot(snapshot,manifest),/authenticated movie/);
  snapshot.cards[1]=DOMFixture(manifest).cards[1];
  snapshot.cards[0].text='Physical engineering: fail.';
  assert.throws(()=>validateDOMSnapshot(snapshot,manifest),/Failure reason/);
});

test('exact duplicates need representative links and cannot show distinct videos', () => {
  const manifest=manifestFixture(),snapshot=DOMFixture(manifest);
  manifest.variants[2].duplicate_of='01';
  snapshot.cards[2].video=null;
  snapshot.cards[2].badge='Exact body + DMX duplicate · representative 01';
  snapshot.cards[2].links=[{href:'#variant-01'}];
  validateDOMSnapshot(snapshot,manifest);
  snapshot.cards[2].links=[];
  assert.throws(()=>validateDOMSnapshot(snapshot,manifest),/representative/);
});

test('playback needs actual frame/time progress in every player and bounded skew', () => {
  const before={cards:['01','03','05'].map(id=>({id,video:{time:12,frames:10}}))};
  const after={cards:['01','03','05'].map((id,i)=>({id,video:{time:12.7+i*.03,frames:30,paused:false,
    ready:4,seeking:false,duration:120,error:null}}))};
  const result=assertPlaybackProgress(before,after,['01','03','05']);
  assert(result.max_skew_s<.1);
  after.cards[1].video.frames=10;
  assert.throws(()=>assertPlaybackProgress(before,after,['01','03','05']),/no new frames/);
  after.cards[1].video.frames=30;after.cards[1].video.time=13.5;
  assert.throws(()=>assertPlaybackProgress(before,after,['01','03','05']),/diverged/);
  after.cards[1].video.time=12.7;after.cards[2].video.paused=true;
  assert.throws(()=>assertPlaybackProgress(before,after,['01','03','05']),/not playing/);
});

test('light playback polling never reads or serializes hidden evidence, card text, notes or event logs', () => {
  let fullReads=0, largeReadsAllowed=false;
  const large='dense scientific evidence '.repeat(5000);
  const hidden=()=>{fullReads++;assert(largeReadsAllowed,'light snapshot touched hidden data');return large;};
  const cards=Array.from({length:16},(_,i)=>({id:'variant-'+String(i+1).padStart(2,'0'),
    classList:{contains:()=>false},get textContent(){return hidden();},
    querySelector(selector){
      if(selector==='video')return {getAttribute:()=>'/media/01.mp4',currentSrc:'http://localhost/media/01.mp4',
        paused:false,currentTime:12,duration:120,muted:true,seeking:false,readyState:4,
        getVideoPlaybackQuality:()=>({totalVideoFrames:20}),error:null};
      if(selector==='input[type=checkbox]')return {checked:i<2};
      if(selector==='textarea')throw Error('unrequested notes inspected');
      if(selector==='pre')return {get textContent(){return hidden();}};
      return {textContent:'short title'};
    },querySelectorAll:()=>[]}));
  const context={document:{querySelectorAll:()=>cards,getElementById:id=>({textContent:'ready',value:id==='seek'?'12':'all'})},
    master:null,performance:{timeOrigin:123},window:{__galleryCheckSeekingCounts:{'03':2},
      get __galleryCheckMediaEvents(){throw Error('light snapshot serialized growing event log');}}};
  const light=vm.runInNewContext(snapshotExpression(),context);
  assert.equal(light.cards.length,16);assert.equal(fullReads,0);
  assert(JSON.stringify(light).length<10000);
  assert(!('evidence' in light.cards[0])&&!('text' in light.cards[0])&&!('note' in light.cards[0]));
  largeReadsAllowed=true;
  const full=vm.runInNewContext(snapshotExpression({full:true}),context);
  assert.equal(fullReads,32);
  assert(JSON.stringify(full).length>JSON.stringify(light).length*100);
});

function playbackState(time,frames,{seeking=false,counts=0,skew=.02}={}) {
  return {seekEventCounts:{'03':counts},cards:['01','03'].map((id,i)=>({id,visible:true,
    video:{time:time+i*skew,frames,paused:false,ready:4,seeking:i===1&&seeking,duration:120,
      muted:true,checked:true,error:null}}))};
}

function virtualPlayback(states) {
  let ms=0,index=0;
  return {snapshot:async()=>states[Math.min(index++,states.length-1)],
    timing:{clock:()=>ms,sleep:async delay=>{ms+=delay;}},elapsed:()=>ms,reads:()=>index};
}

test('settled sampling records transient seeks without relaxing 700ms, progress, frame or skew limits', async () => {
  const stream=virtualPlayback([playbackState(12,10,{seeking:true,counts:1}),playbackState(12.075,12,{counts:1}),
    playbackState(12.775,33,{seeking:true,counts:2}),playbackState(12.85,35,{counts:2})]);
  const telemetry={};
  const sample=await sampleSettledPlayback(stream.snapshot,['01','03'],2000,telemetry,stream.timing);
  assert.equal(telemetry.status,'passed');assert.equal(telemetry.requested_sample_ms,700);
  assert.equal(telemetry.actual_sample_ms,775);assert.equal(telemetry.total_settling_ms,150);
  assert.deepEqual(telemetry.seek_event_deltas,{'01':0,'03':1});
  assert.equal(telemetry.before.seeking_poll_observations['03'],1);
  assert.equal(telemetry.after.seeking_poll_observations['03'],1);
  assert.equal(sample.result.limit_s,.35);assert(sample.result.players.every(row=>row.advance_s>=.12&&row.decoded_frames>0));
});

test('settled sampler reports real stutter and excessive skew without resampling favorable outcomes', async () => {
  for(const [after,pattern] of [[playbackState(12.05,20),/did not advance/],
    [playbackState(12.7,10),/no new frames/],[playbackState(12.7,30,{skew:.5}),/diverged/]]){
    const stream=virtualPlayback([playbackState(12,10),after]),telemetry={};
    await assert.rejects(sampleSettledPlayback(stream.snapshot,['01','03'],2000,telemetry,stream.timing),pattern);
    assert.equal(stream.reads(),2);assert.equal(stream.elapsed(),700);assert.equal(telemetry.status,'failed');
    assert(telemetry.after.last_snapshot);
  }
});

test('persistent seeking exhausts the original deadline and retains settling diagnostics', async () => {
  const stream=virtualPlayback([playbackState(12,10),playbackState(12.7,30,{seeking:true,counts:1})]),telemetry={};
  await assert.rejects(sampleSettledPlayback(stream.snapshot,['01','03'],900,telemetry,stream.timing),/deadline/);
  assert.equal(stream.elapsed(),900);assert.equal(telemetry.status,'failed');
  assert.equal(telemetry.after.seeking_poll_observations['03'],3);
  assert.equal(telemetry.after.last_snapshot.cards[1].video.seeking,true);
});

test('CDP correlates sessions and rejects errors and timeouts without a browser', async () => {
  class Socket extends EventTarget {
    sent=[];
    send(text){this.sent.push(JSON.parse(text));}
    answer(value){this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify(value)}));}
  }
  const socket=new Socket(),cdp=new CDP(socket,20);
  const request=cdp.send('Page.test',{sample:true},'owned-session');
  assert.equal(socket.sent[0].sessionId,'owned-session');
  socket.answer({id:socket.sent[0].id,result:{ok:true}});
  assert.deepEqual(await request,{ok:true});
  const bad=cdp.send('Page.bad');
  socket.answer({id:socket.sent.at(-1).id,error:{message:'rejected'}});
  await assert.rejects(bad,/rejected/);
  await assert.rejects(cdp.send('Page.unanswered'),/timeout/);
  const closed=cdp.send('Page.pending');socket.dispatchEvent(new Event('close'));
  await assert.rejects(closed,/closed/);
});

test('all16 movie hashes/receipts authenticate and changed or outside media fails', async () => {
  const root=await fs.mkdtemp(path.join(tmpdir(),'gallery-contract-test-'));
  try {
    const manifest=manifestFixture();
    Object.assign(manifest,{review_start_s:60,review_duration_s:120,mode:'dark'});
    await fs.mkdir(path.join(root,'review'));
    await fs.writeFile(path.join(root,'PLAN.json'),'{}');
    manifest.plan_sha256=await digest(path.join(root,'PLAN.json'));
    await fs.writeFile(path.join(root,'review/index.html'),'test contract only');
    for(const row of manifest.variants){
      const folder=path.join(root,'media',row.algorithm_id,row.track_id);
      await fs.mkdir(folder,{recursive:true});
      row.path=path.join(folder,'dark.mp4');
      await fs.writeFile(row.path,'Unit-test bytes, never a delivery verdict '+row.id);
      row.sha256=await digest(row.path);row.bytes=(await fs.stat(row.path)).size;
      const audit=path.join(folder,'dark_audit.json');await fs.writeFile(audit,'{}');
      await fs.writeFile(path.join(folder,'dark_verified.json'),JSON.stringify({video_sha256:row.sha256,
        audio_matches_source_crop:true,audit_sha256:await digest(audit)}));
    }
    await fs.writeFile(path.join(root,'review/MANIFEST.json'),JSON.stringify(manifest));
    const result=await authenticateInputs(root,manifest.plan_sha256);
    assert.equal(Object.keys(result.pins).length,51);
    await fs.writeFile(manifest.variants[4].path,'substituted');
    await assert.rejects(authenticateInputs(root,manifest.plan_sha256),/SHA256 mismatch/);
  } finally {await fs.rm(root,{recursive:true,force:true});}
});
