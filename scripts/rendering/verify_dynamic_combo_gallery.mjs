#!/usr/bin/env node
/** CPU/software-only real-browser checks for four full-song combo comparisons.
 * node scripts/rendering/verify_dynamic_combo_gallery.mjs --root ROOT
 *   --plan-sha256 SHA --url http://127.0.0.1:PORT --revision browser_v1
 * Uses no synthetic delivery media; preserves existing reports and product baseline.
 */
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const HERE=fileURLToPath(import.meta.url);
const SONGS=['shpongle','cure_for_me'];
const ORDER=[['C3','shpongle'],['C3','cure_for_me'],['C4','shpongle'],['C4','cure_for_me']];
const LIMITS={sample_ms:700,minimum_advance_s:.12,maximum_skew_s:.35,duration_tolerance_s:.06};
export const check=(condition,message)=>{if(!condition)throw Error(message);};
const read=async filename=>JSON.parse(await fs.readFile(filename,'utf8'));
export async function digest(filename){const hash=createHash('sha256');for await(const chunk of createReadStream(filename))hash.update(chunk);return hash.digest('hex');}
const pin=async(filename,expected)=>check(/^[a-f0-9]{64}$/.test(expected||'')&&await digest(filename)===expected,'SHA256 mismatch: '+filename);
const writeNew=async(filename,value)=>fs.writeFile(filename,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
async function floor(root){const s=await fs.statfs(root,{bigint:true});check(s.bavail*s.bsize>=(20n*1024n**3n+128n*1024n**2n),'20 GiB floor plus evidence allowance reached');}

export function parseArguments(argv){
  const args={chrome:'/opt/google/chrome/chrome',revision:'browser_v1',timeoutMs:30000};
  const names={'--root':'root','--plan-sha256':'planSha256','--url':'url','--revision':'revision','--chrome':'chrome','--timeout-ms':'timeoutMs'};
  for(let i=0;i<argv.length;i+=2){check(names[argv[i]]&&argv[i+1]&&!argv[i+1].startsWith('--'),'Unknown argument or missing value');args[names[argv[i]]]=argv[i+1];}
  check(args.root&&/^[a-f0-9]{64}$/.test(args.planSha256||'')&&args.url,'Require root, externally frozen PLAN digest and loopback URL');
  const url=new URL(args.url);
  check(url.protocol==='http:'&&['localhost','127.0.0.1','[::1]'].includes(url.hostname)&&url.pathname==='/'&&!url.username&&!url.password&&!url.search&&!url.hash,'Plain loopback origin required');
  check(/^[A-Za-z0-9_-]+$/.test(args.revision),'Unsafe revision');
  args.root=path.resolve(args.root);args.url=url.origin;args.timeoutMs=Number(args.timeoutMs);
  check(Number.isInteger(args.timeoutMs)&&args.timeoutMs>=15000&&args.timeoutMs<=60000,'Browser budget must be 15000–60000 ms');
  return args;
}

export function chromeArguments(profile){return ['--headless=new','--no-sandbox','--no-first-run','--no-default-browser-check',
  '--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions','--disable-dev-shm-usage',
  '--disable-gpu','--disable-accelerated-video-decode','--disable-accelerated-video-encode','--use-gl=angle','--use-angle=swiftshader',
  '--disable-features=Vulkan,VaapiVideoDecoder,VaapiVideoEncoder,AcceleratedVideoDecodeLinuxGL','--autoplay-policy=no-user-gesture-required',
  '--mute-audio','--window-size=1600,1100','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'];}

export function validateManifest(manifest,plan,planSha){
  check(manifest.plan_sha256===planSha&&manifest.baseline_promoted===false&&manifest.reference==='D7_reference'&&manifest.source_start_s===0,'Baseline/reference/start policy differs');
  check(JSON.stringify(manifest.tracks)===JSON.stringify(plan.tracks)&&JSON.stringify(manifest.algorithms)===JSON.stringify(plan.algorithms),'Manifest differs from PLAN songs/methods');
  check(manifest.variants?.length===4,'Exactly four comparison cards required');
  for(let i=0;i<4;i++){
    const row=manifest.variants[i],[method,song]=ORDER[i],track=plan.tracks.find(t=>t.id===song);
    check(track&&row.id===String(i+1).padStart(2,'0')&&row.algorithm_id===method&&row.track_id===song&&row.status==='complete','Wrong/missing completed comparison identity');
    check(Number.isFinite(track.duration_s)&&track.duration_s>0&&track.video_frames===Math.ceil(track.duration_s*30),'Frozen full-song clock invalid');
    check(row.duration_s===track.video_frames/30&&row.source_duration_s===track.duration_s&&row.duration_s-track.duration_s>=0&&row.duration_s-track.duration_s<1/30+1e-10,'Movie is cropped or exceeds fractional-frame padding');
  }
}

export async function authenticateInputs(root,planSha){
  const planPath=path.join(root,'PLAN.json'),manifestPath=path.join(root,'review/MANIFEST.json'),html=path.join(root,'review/index.html');
  await pin(planPath,planSha);const plan=await read(planPath),manifest=await read(manifestPath);validateManifest(manifest,plan,planSha);
  const pins={[planPath]:planSha,[manifestPath]:await digest(manifestPath),[html]:await digest(html)};
  const mediaRoot=await fs.realpath(path.join(root,'media'));
  for(const row of manifest.variants){
    const expected=path.join(root,'media',row.algorithm_id,row.track_id,'full.mp4'),movie=await fs.realpath(row.path);
    check(movie.startsWith(mediaRoot+path.sep)&&movie===await fs.realpath(expected),'Movie escapes exact method/song allowlist');
    await pin(movie,row.sha256);check((await fs.stat(movie)).size===row.bytes,'Movie byte count differs');pins[movie]=row.sha256;
    const dir=path.dirname(movie),verifiedPath=path.join(dir,'full_verified.json'),auditPath=path.join(dir,'full_audit.json'),renderPath=path.join(dir,'full_render.json');
    const verified=await read(verifiedPath),audit=await read(auditPath),render=await read(renderPath);
    check(verified.video_sha256===row.sha256&&verified.complete_full_song===true&&verified.source_audio_unchanged_speed===true,'Missing verified full-song identity');
    await pin(auditPath,verified.audit_sha256);
    const track=plan.tracks.find(t=>t.id===row.track_id);
    check(audit.video_sha256===row.sha256&&audit.decoded_frames===track.video_frames&&audit.pts_uniform===true&&audit.full_software_AV_decode===true&&audit.audio_matches_full_reference===true,'Full software A/V audit missing');
    check(audit.source_start_s===0&&audit.audio_speed===1&&audit.source_duration_s===track.duration_s&&audit.video_duration_s===row.duration_s,'Audit full-song timing mismatch');
    check(render.sha256===row.sha256&&render.frames===track.video_frames&&render.start_s===0&&render.audio_speed===1&&render.duration_s===row.duration_s,'Render full-song timing mismatch');
    check(JSON.stringify(row.inputs)===JSON.stringify(verified.inputs)&&JSON.stringify(row.inputs)===JSON.stringify(render.inputs)&&JSON.stringify(row.inputs)===JSON.stringify(audit.inputs),'Gallery/render/audit inputs differ');
    const audioRecord=path.join(root,'audio',row.track_id,'REFERENCE.json'),audio=await read(audioRecord);
    check(audio.plan_sha256===planSha&&audio.source_audio===track.audio&&audio.source_audio_sha256===plan.files[track.audio]&&audio.source_start_s===0&&audio.playback_rate===1&&audio.duration_s===row.duration_s,'Full audio reference binding mismatch');
    check(audio.end_padding_s>=0&&audio.end_padding_s<1/30+1e-10&&Math.abs(audio.end_padding_s-(row.duration_s-track.duration_s))<1e-10,'Audio padding changed');
    check(audit.pcm_sha256===audio.decoded_pcm_sha256,'Delivered/reference audio PCM hashes differ');
    await pin(audio.reference,audio.reference_sha256);await pin(track.audio,audio.source_audio_sha256);
    check(row.inputs.template_sha256===pins[html]&&row.inputs.audio_record_sha256===await digest(audioRecord),'Gallery template/audio record no longer matches render inputs');
    for(const filename of [verifiedPath,auditPath,renderPath,audioRecord])pins[filename]=await digest(filename);
    pins[audio.reference]=audio.reference_sha256;pins[track.audio]=audio.source_audio_sha256;
  }
  return {manifest,plan,pins};
}

export function snapshotExpression({evidence=false}={}){return `(() => ({
  navigation:performance.timeOrigin,status:document.getElementById('status')?.textContent,
  song:document.getElementById('song')?.value,audible:document.getElementById('audible')?.value,
  seek:Number(document.getElementById('seek')?.value),seekMax:Number(document.getElementById('seek')?.max),
  masterId:typeof masterId==='undefined'?null:masterId,desiredPlaying:typeof desiredPlaying==='undefined'?null:desiredPlaying,
  seekCounts:window.__comboSeekingCounts||{},cards:[...document.querySelectorAll('.card')].map(card=>({
    id:card.id.slice(8),visible:!card.classList.contains('hidden'),
    ${evidence?`title:card.querySelector('h2')?.textContent,badge:card.querySelector('.badge')?.textContent,
    evidence:card.querySelector('pre')?.textContent,links:[...card.querySelectorAll('a')].map(a=>({href:a.getAttribute('href'),download:a.getAttribute('download')})),`:''}
    video:(()=>{const v=card.querySelector('video');return v?{src:v.getAttribute('src'),paused:v.paused,time:v.currentTime,
      duration:Number.isFinite(v.duration)?v.duration:null,ready:v.readyState,seeking:v.seeking,muted:v.muted,
      frames:v.getVideoPlaybackQuality?.().totalVideoFrames??null,error:v.error?{code:v.error.code,message:v.error.message}:null}:null})()
  }))}))()`;}

export function validateDOM(snapshot,manifest){
  check(snapshot.cards.length===4,'Browser card count differs');
  for(const row of manifest.variants){const card=snapshot.cards.find(c=>c.id===row.id);check(card&&card.title===row.algorithm_id+' | '+row.title,'Card identity differs');
    const evidence=JSON.parse(card.evidence);check(evidence.video_sha256===row.sha256&&evidence.duration_s===row.duration_s&&evidence.source_duration_s===row.source_duration_s&&JSON.stringify(evidence.comparison)===JSON.stringify(row.comparison),'Lazy evidence differs from manifest');
    if(row.duplicate_of){check(!card.video&&card.links.some(a=>a.href==='#variant-'+row.duplicate_of)&&card.badge.includes('representative '+row.duplicate_of),'Exact duplicate is not disclosed');}
    else{check(card.video?.src==='/media/'+row.id+'.mp4?v='+encodeURIComponent(row.sha256)&&card.badge===row.banner,'Wrong media or diagnostic banner');
      check(card.links.some(a=>a.href==='/media/'+row.id+'.mp4?download=1&v='+encodeURIComponent(row.sha256)&&a.download===row.download_name),'Full-song download link missing');}
  }
}

export function assertProgress(before,after,ids,expectedDuration){
  const audible=after.cards.filter(c=>c.video?.muted===false);check(audible.length<=1&&audible.every(c=>ids.includes(c.id)),'Multiple or hidden audio sources unmuted');
  const values=ids.map(id=>{const a=before.cards.find(c=>c.id===id)?.video,b=after.cards.find(c=>c.id===id)?.video;
    check(a&&b&&!b.error&&!b.paused&&b.ready>=2&&!b.seeking,'Selected player is not settled and playing: '+id);
    check(b.duration!==null&&Math.abs(b.duration-expectedDuration)<=LIMITS.duration_tolerance_s,'Browser duration differs from full track');
    check(b.time-a.time>=LIMITS.minimum_advance_s,'Real media did not advance: '+id);
    if(a.frames!==null&&b.frames!==null)check(b.frames>a.frames,'No newly decoded frames: '+id);
    return {id,time_s:b.time,advance_s:b.time-a.time,decoded_frames:a.frames===null?null:b.frames-a.frames};});
  const skew=Math.max(...values.map(v=>v.time_s))-Math.min(...values.map(v=>v.time_s));check(skew<=LIMITS.maximum_skew_s,'Playback skew exceeds fixed .35 s');
  return {players:values,max_skew_s:skew,limits:{...LIMITS},expected_duration_s:expectedDuration};
}

export async function samplePlayback(snapshot,ids,expectedDuration,deadline,telemetry={},
  {clock=()=>performance.now(),sleep=delay}={}){
  const started=clock();telemetry.requested_sample_ms=700;let initialCounts;
  const settled=async phase=>{const begin=clock(),stats=telemetry[phase]={polls:0,seeking_poll_observations:{},settling_ms:0};
    while(clock()<deadline){const state=await snapshot();stats.polls++;stats.settling_ms=clock()-begin;stats.last_snapshot=state;check(clock()<deadline,'Browser deadline while observing '+phase);
      initialCounts??={...state.seekCounts};const videos=ids.map(id=>{const v=state.cards.find(c=>c.id===id)?.video;check(v&&!v.error,'Missing or failed media: '+id);if(v.seeking)stats.seeking_poll_observations[id]=(stats.seeking_poll_observations[id]||0)+1;return v;});
      telemetry.seek_event_deltas=Object.fromEntries(ids.map(id=>[id,Math.max(0,(state.seekCounts?.[id]||0)-(initialCounts[id]||0))]));
      if(videos.every(v=>!v.paused&&v.ready>=2&&!v.seeking))return state;
      await sleep(Math.max(0,Math.min(75,deadline-clock())));
    }stats.settling_ms=clock()-begin;throw Error('Browser deadline: '+phase+' never settled');};
  try{const before=await settled('before'),begin=clock();check(begin+700<=deadline,'Deadline cannot fit unchanged 700 ms interval');await sleep(700);const after=await settled('after');telemetry.actual_sample_ms=clock()-begin;
    const result=assertProgress(before,after,ids,expectedDuration);telemetry.status='passed';return {before,after,result};
  }catch(error){telemetry.status='failed';telemetry.error=error.message;throw error;}
  finally{telemetry.elapsed_ms=clock()-started;telemetry.total_settling_ms=(telemetry.before?.settling_ms||0)+(telemetry.after?.settling_ms||0);}
}

class CDP{
  constructor(socket){this.socket=socket;this.next=0;this.pending=new Map();socket.addEventListener('message',event=>{const value=JSON.parse(event.data);if(value.id){const p=this.pending.get(value.id);if(!p)return;clearTimeout(p.timer);this.pending.delete(value.id);value.error?p.reject(Error(JSON.stringify(value.error))):p.resolve(value.result||{});}else this.onEvent?.(value);});
    socket.addEventListener('close',()=>{for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(Error('Owned Chrome CDP closed'));}this.pending.clear();});}
  send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++this.next,timer=setTimeout(()=>{this.pending.delete(id);reject(Error('CDP timeout: '+method));},5000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});}
  async evaluate(expression,session){const r=await this.send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true,userGesture:true},session);check(!r.exceptionDetails,'Browser expression failed: '+JSON.stringify(r.exceptionDetails));return r.result?.value;}
}

async function waitFor(predicate,deadline,label){while(performance.now()<deadline){const value=await predicate();if(value)return value;await delay(75);}throw Error('Browser deadline: '+label);}

async function fetchPinned(url,route,filename){const response=await fetch(url+route,{redirect:'error',signal:AbortSignal.timeout(5000)});check(response.ok,'HTTP failed: '+route);let size=0;const hash=createHash('sha256');for await(const chunk of response.body){size+=chunk.length;check(size<16*1024**2,'Oversized gallery document');hash.update(chunk);}check(hash.digest('hex')===await digest(filename),'Served document differs: '+route);}

async function downloadRoutes(url,root,manifest){const result=[];for(const row of manifest.variants){const route='/media/'+row.id+'.mp4';const head=await fetch(url+route+'?download=1',{method:'HEAD',redirect:'error',signal:AbortSignal.timeout(5000)});
    check(head.status===200&&Number(head.headers.get('content-length'))===row.bytes&&head.headers.get('content-type')==='video/mp4','Media HEAD mismatch');
    check(head.headers.get('content-disposition')==="attachment; filename*=UTF-8''"+encodeURIComponent(row.download_name),'Wrong downloaded video filename');
    const response=await fetch(url+route,{headers:{Range:'bytes=0-63'},redirect:'error',signal:AbortSignal.timeout(5000)});
    check(response.status===206&&response.headers.get('content-range')===`bytes 0-63/${row.bytes}`,'Video Range route unavailable');
    const bytes=Buffer.from(await response.arrayBuffer()),handle=await fs.open(row.path,'r'),expected=Buffer.alloc(64);try{await handle.read(expected,0,64,0);}finally{await handle.close();}check(bytes.equals(expected),'Browser route serves different movie bytes');result.push({id:row.id,download_header:true,range_bytes:true});}
  return result;}

async function browserChecks(cdp,session,manifest,destination,deadline,report){
  const evaluate=x=>cdp.evaluate(x,session),expression=snapshotExpression();report.observer={light_calls:0,max_payload_bytes:0};
  const snapshot=async()=>{const s=await evaluate(expression);report.observer.light_calls++;report.observer.max_payload_bytes=Math.max(report.observer.max_payload_bytes,Buffer.byteLength(JSON.stringify(s)));return s;};
  const action=evaluate,click=id=>action(`document.getElementById(${JSON.stringify(id)}).click()`);
  const select=(id,value)=>action(`(()=>{const x=document.getElementById(${JSON.stringify(id)});x.value=${JSON.stringify(value)};x.dispatchEvent(new Event('change',{bubbles:true}));return true})()`);
  const seek=t=>action(`(()=>{const x=document.getElementById('seek');x.value=${t};x.dispatchEvent(new Event('input',{bubbles:true}));return true})()`);
  let state=await waitFor(async()=>{const s=await snapshot();return s.cards.length===4&&!s.status.includes('unavailable')&&s;},deadline,'four gallery cards');
  // Lazy evidence is opened once before playback, then closed. Polls never touch it.
  await action(`document.querySelectorAll('.card details').forEach(x=>x.open=true)`);
  await waitFor(()=>evaluate(`document.querySelectorAll('.card details pre').length===4`),deadline,'lazy evidence');
  const full=await evaluate(snapshotExpression({evidence:true}));
  validateDOM(full,manifest);report.checks.cards_evidence={cards:4,full_payload_bytes:Buffer.byteLength(JSON.stringify(full)),validation_count:1};
  await action(`document.querySelectorAll('.card details').forEach(x=>x.open=false)`);
  report.playback_observations=[];
  const observe=async(ids,duration,label)=>{const telemetry={label,ids};report.playback_observations.push(telemetry);const sample=await samplePlayback(snapshot,ids,duration,deadline,telemetry);state=sample.after;return sample.result;};
  for(const song of SONGS){const rows=manifest.variants.filter(r=>r.track_id===song),ids=rows.filter(r=>!r.duplicate_of).map(r=>r.id),duration=rows[0].duration_s;
    check(ids.length===2,'Two distinct same-song representatives needed for real comparison/handoff; exact duplicates cannot count twice');
    await select('song',song);
    state=await waitFor(async()=>{const s=await snapshot();return s.cards.filter(c=>c.visible).length===2&&s.seek===0&&s.cards.every(c=>!c.video||c.video.paused)&&s;},deadline,'song reset');
    check(Math.abs(state.seekMax-duration)<1e-9&&state.cards.filter(c=>c.visible).every(c=>ids.includes(c.id)),'Wrong variable full-song seek range/filter');
    await seek(12);
    await waitFor(async()=>{const s=await snapshot();return ids.every(id=>{const v=s.cards.find(c=>c.id===id)?.video;return v&&v.ready>=2&&!v.seeking&&Math.abs(v.time-12)<.15;});},deadline,'shared song seek');
    await select('audible',ids[0]);await click('play');
    report.checks[song+'_playback']=await observe(ids,duration,song+'_playback');
    check(state.masterId===ids[0]&&state.cards.filter(c=>c.video&&!c.video.muted).map(c=>c.id).join()===ids[0],'Audio/master selection mismatch');
    const beforeMaster=state.masterId;
    await select('audible',ids[1]);
    report.checks[song+'_master_handoff']={old_master:beforeMaster,...await observe(ids,duration,song+'_master_handoff')};
    check(state.masterId===ids[1]&&state.cards.filter(c=>c.video&&!c.video.muted).map(c=>c.id).join()===ids[1],'Live audio-selected master handoff failed');
    await select('audible','none');state=await snapshot();check(state.cards.every(c=>!c.video||c.video.muted),'Muted selection left audible soundtrack');
    await click('pause');state=await snapshot();check(state.cards.every(c=>!c.video||c.video.paused),'Pause left a movie running');
    // The end of each full track, beyond the old 180-second prefix, must seek.
    const late=duration-2;
    await seek(late);
    await waitFor(async()=>{const s=await snapshot();return ids.every(id=>{const v=s.cards.find(c=>c.id===id).video;return v.ready>=2&&!v.seeking&&Math.abs(v.time-late)<.15;});},deadline,'full-song late seek');
    await click('play');report.checks[song+'_late_playback']=await observe(ids,duration,song+'_late_playback');
    await click('reset');state=await snapshot();
    check(state.seek===0&&state.cards.every(c=>!c.video||c.video.paused),'Start/reset does not stop and rewind');
    report.checks[song+'_controls']={shared_seek_s:12,late_seek_s:late,duration_s:duration,pause_and_reset:true};
  }
  const noteId='02',marker='Automated full-song gallery mechanics only; no baseline promotion or artistic acceptance. '+report.revision;
  await action(`(()=>{const x=document.querySelector('#variant-${noteId} textarea');x.value=${JSON.stringify(marker)};x.dispatchEvent(new Event('input',{bubbles:true}));return true})()`);
  const previous=(await snapshot()).navigation;report.media_events_before_reload=await evaluate('window.__comboMediaEvents||[]');
  await cdp.send('Page.reload',{ignoreCache:true},session);
  await waitFor(async()=>{const s=await snapshot();return s.navigation!==previous&&s.cards.length===4&&s;},deadline,'reload note persistence');
  check(await evaluate(`document.querySelector('#variant-${noteId} textarea')?.value`)===marker,'Review note lost on reload');
  await click('export');const notesPath=path.join(destination,'downloads/dynamic_combo_review_notes.json');
  const notes=await waitFor(async()=>{try{return await read(notesPath);}catch(e){if(e.code==='ENOENT'||e instanceof SyntaxError)return false;throw e;}},deadline,'real notes download');
  check(notes.plan_sha256===manifest.plan_sha256&&notes.notes?.[noteId]?.text===marker&&notes.notes[noteId].video_sha256===manifest.variants.find(r=>r.id===noteId).sha256,'Exported notes not bound to exact movie');
  report.checks.notes={path:notesPath,sha256:await digest(notesPath),persistence:true};
  report.final_snapshot=await snapshot();report.media_events_after_reload=await evaluate('window.__comboMediaEvents||[]');
  check(![...report.media_events_before_reload,...report.media_events_after_reload].some(e=>e.type==='error'),'Media decode/playback error recorded');
  const screenshot=await cdp.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false},session),filename=path.join(destination,'gallery.png');
  await fs.writeFile(filename,Buffer.from(screenshot.data,'base64'),{flag:'wx'});report.screenshot={path:filename,sha256:await digest(filename)};
}

export async function run(args){
  await floor(args.root);const dest=path.join(args.root,'delivery_verification',args.revision);await fs.mkdir(path.dirname(dest),{recursive:true});await fs.mkdir(dest);
  const report={status:'running',revision:args.revision,started_at:new Date().toISOString(),verifier_sha256:await digest(HERE),plan_sha256:args.planSha256,url:args.url,
    checks:{},browser_errors:[],network_errors:[],console_errors:[],decoder_properties:[],limits:LIMITS,baseline_promoted:false,
    scope:'Real full-song gallery mechanics only. CPU/software-only browser, no scientific or artistic acceptance; authenticated full A/V audits remain separate.'};
  let cdp,session,child,profile,watchdog,started;
  try{const inputs=await authenticateInputs(args.root,args.planSha256);report.inputs=inputs.pins;
    await fetchPinned(args.url,'/',path.join(args.root,'review/index.html'));await fetchPinned(args.url,'/manifest.json',path.join(args.root,'review/MANIFEST.json'));
    report.checks.media_routes=await downloadRoutes(args.url,args.root,inputs.manifest);
    await fs.mkdir(path.join(dest,'downloads'));profile=await fs.mkdtemp(path.join(tmpdir(),'lightsem-combo-browser-'));const flags=chromeArguments(profile);
    report.chrome_command=[args.chrome,...flags];started=performance.now();const deadline=started+args.timeoutMs;
    child=spawn(args.chrome,flags,{stdio:['ignore','ignore','pipe'],detached:true});report.owned_browser_pid=child.pid;let launchError;child.on('error',error=>launchError=error);report.chrome_stderr='';
    child.stderr.on('data',chunk=>{if(report.chrome_stderr.length<100000)report.chrome_stderr+=chunk.toString();});
    watchdog=setTimeout(()=>{report.deadline_exceeded=true;if(child?.pid){try{process.kill(-child.pid,'SIGTERM');}catch{}}},args.timeoutMs);
    const endpoint=await waitFor(async()=>{if(launchError)throw launchError;check(child.exitCode===null&&child.signalCode===null,'Owned Chrome exited: '+report.chrome_stderr.slice(-2000));try{return(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).trim().split('\n');}catch(e){if(e.code==='ENOENT')return false;throw e;}},deadline,'owned Chrome CDP');
    const socket=new WebSocket('ws://127.0.0.1:'+endpoint[0]+endpoint[1]);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',()=>reject(Error('Owned browser socket failed')),{once:true});});cdp=new CDP(socket);
    cdp.onEvent=e=>{if(e.method==='Runtime.exceptionThrown')report.browser_errors.push(e.params);if(e.method==='Network.loadingFailed')report.network_errors.push(e.params);if(e.method==='Log.entryAdded'&&e.params.entry.level==='error')report.console_errors.push(e.params.entry);if(e.method==='Media.playerPropertiesChanged')report.decoder_properties.push(e.params);};
    report.browser_version=await cdp.send('Browser.getVersion');await cdp.send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:path.join(dest,'downloads'),eventsEnabled:true});
    const {targetId}=await cdp.send('Target.createTarget',{url:'about:blank'});({sessionId:session}=await cdp.send('Target.attachToTarget',{targetId,flatten:true}));
    for(const domain of ['Page','Runtime','Network','Log','Media'])await cdp.send(domain+'.enable',{},session);
    await cdp.send('Emulation.setDeviceMetricsOverride',{width:1600,height:1100,deviceScaleFactor:1,mobile:false},session);
    await cdp.send('Page.addScriptToEvaluateOnNewDocument',{source:`window.__comboMediaEvents=[];window.__comboSeekingCounts={};for(const type of ['error','stalled','waiting','playing','pause','seeking','seeked'])document.addEventListener(type,e=>{if(!(e.target instanceof HTMLMediaElement))return;const id=e.target.closest('.card')?.id.slice(8);if(type==='seeking'&&id)window.__comboSeekingCounts[id]=(window.__comboSeekingCounts[id]||0)+1;if(window.__comboMediaEvents.length<1000)window.__comboMediaEvents.push({type,id,time:performance.now(),media_time:e.target.currentTime,error:e.target.error?{code:e.target.error.code,message:e.target.error.message}:null})},true);`},session);
    await cdp.send('Page.navigate',{url:args.url+'/'},session);await browserChecks(cdp,session,inputs.manifest,dest,deadline,report);
    check(!report.browser_errors.length&&!report.network_errors.some(e=>!e.canceled),'Browser JavaScript or network errors recorded');
    check(report.decoder_properties.flatMap(e=>e.properties||[]).filter(p=>/IsPlatformVideoDecoder/i.test(p.name)).every(p=>!['true','1'].includes(String(p.value).toLowerCase())),'Browser used platform video decoder despite software-only flags');
    check(!report.deadline_exceeded,'Browser exceeded fixed budget');report.status='passed';
  }catch(error){report.status='failed';report.error={name:error.name,message:error.message};if(cdp&&session&&!report.deadline_exceeded){try{report.failure_snapshot=await cdp.evaluate(snapshotExpression(),session);report.failure_media_events=await cdp.evaluate('window.__comboMediaEvents||[]',session);const shot=await cdp.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false},session),filename=path.join(dest,'failure.png');await fs.writeFile(filename,Buffer.from(shot.data,'base64'),{flag:'wx'});report.failure_screenshot={path:filename,sha256:await digest(filename)};}catch(e){report.failure_capture_error=e.message;}}}
  finally{clearTimeout(watchdog);if(started)report.live_browser_elapsed_ms=Math.round(performance.now()-started);if(cdp){try{await cdp.send('Browser.close');}catch{}cdp.socket.close();}
    if(child?.pid){for(let i=0;i<20&&child.exitCode===null&&child.signalCode===null;i++)await delay(50);if(child.exitCode===null&&child.signalCode===null){try{process.kill(-child.pid,'SIGTERM');}catch{}}for(let i=0;i<10&&child.exitCode===null&&child.signalCode===null;i++)await delay(50);if(child.exitCode===null&&child.signalCode===null){try{process.kill(-child.pid,'SIGKILL');}catch{}}}
    if(profile){try{await fs.rm(profile,{recursive:true});}catch(e){report.profile_cleanup_error=e.message;}}}
  if(report.inputs){try{for(const[file,hash]of Object.entries(report.inputs))await pin(file,hash);}catch(e){report.status='failed';report.input_drift_error=e.message;}}
  await floor(args.root);report.finished_at=new Date().toISOString();const output=path.join(dest,'REPORT.json');await writeNew(output,report);await writeNew(output+'.sha256.json',{sha256:await digest(output)});
  console.log('COMBO_GALLERY_'+report.status.toUpperCase(),output);return report;
}

if(process.argv[1]&&path.resolve(process.argv[1])===HERE){try{const result=await run(parseArguments(process.argv.slice(2)));if(result.status!=='passed')process.exitCode=1;}catch(error){console.error(error.message);process.exitCode=1;}}
