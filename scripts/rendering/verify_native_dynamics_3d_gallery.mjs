#!/usr/bin/env node
/** Actual audited v2 clips; software-only Chrome. No GPU and no media copies on campaign disk. */
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {check,digest,parseArguments,chromeArguments} from './verify_dynamic_combo_gallery.mjs';
const HERE=fileURLToPath(import.meta.url),METHODS=['D7_reference','C3','C4'],SONGS=['shpongle','cure_for_me'];
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const pin=async(p,h)=>check(await digest(p)===h,'Hash mismatch: '+p);
async function floor(root){const s=await fs.statfs(root,{bigint:true});check(s.bavail*s.bsize>=20n*1024n**3n,'20 GiB free-space floor reached');}

export async function authenticate(root,planHash){
  const review=path.join(root,'preview_3d_v2'),planPath=path.join(root,'PLAN.json');await pin(planPath,planHash);
  const plan=await read(planPath),pins={[planPath]:planHash},clips=[];
  const remember=async(p,h)=>{h??=await digest(p);await pin(p,h);pins[p]=h;return h;};
  await remember(path.join(review,'index.html'));
  for(const song of SONGS){
    const track=plan.tracks.find(t=>t.id===song),refDir=path.join(review,'audio',song),refPath=path.join(refDir,'REFERENCE.json'),ref=await read(refPath);
    check(ref.plan_sha256===planHash&&ref.source_start_s===90&&ref.duration_s===60&&ref.playback_rate===1,'Wrong audio crop binding');
    check(ref.source_audio===track.audio&&ref.source_audio_sha256===plan.files[track.audio],'Wrong original song');
    await remember(track.audio,ref.source_audio_sha256);await remember(ref.reference,ref.reference_sha256);const refHash=await remember(refPath);
    for(const method of METHODS){
      const dir=path.join(review,method,song),movie=path.join(dir,'preview.mp4'),renderPath=path.join(dir,'RENDER.json'),auditPath=path.join(dir,'AUDIT.json');
      check((await fs.realpath(movie)).startsWith((await fs.realpath(review))+path.sep),'Movie escapes review root');
      const render=await read(renderPath),audit=await read(auditPath),status=method==='D7_reference'?'experimental_control':'failure_diagnostic';
      check(render.method===method&&render.song===song&&render.status===status&&!render.baseline_promotion,'Wrong method/song/status');
      check(render.frames===1800&&render.source_start_s===90&&render.duration_s===60&&render.audio_speed===1,'Wrong clip clock');
      check(audit.decoded_frames===1800&&audit.pts_uniform&&audit.full_software_AV_decode&&audit.audio_matches_prepared_reference,'Incomplete A/V audit');
      check(audit.source_start_s===90&&audit.duration_s===60&&audit.audio_speed===1&&audit.pcm_sha256===ref.decoded_pcm_sha256,'Audit audio/clock mismatch');
      check(JSON.stringify(audit.inputs)===JSON.stringify(render.inputs),'Audit/render input mismatch');
      check(render.inputs.plan_sha256===planHash&&render.inputs.audio.audio_record_sha256===refHash&&render.inputs.audio.audio_reference_sha256===ref.reference_sha256&&render.inputs.audio.decoded_pcm_sha256===ref.decoded_pcm_sha256,'Audio/source pin mismatch');
      check(audit.movie_sha256===render.sha256&&(await fs.stat(movie)).size===render.bytes,'Movie receipt mismatch');
      await remember(movie,render.sha256);await remember(renderPath,audit.render_sha256);await remember(auditPath);
      const timeline=path.join(root,'timelines',method,song),body=path.join(root,'bodies',method,song);
      for(const[file,key]of [[path.join(timeline,'timeline.npz'),'timeline_sha256'],[path.join(timeline,'patch.json'),'patch_sha256'],[path.join(timeline,'report.json'),'numeric_report_sha256'],[path.join(body,'body.npz'),'body_sha256'],[path.join(body,'body.json'),'body_metadata_sha256']])await remember(file,render.inputs[key]);
      clips.push({method,song,status,path:movie,route:`/${method}/${song}/preview.mp4`,bytes:render.bytes,sha256:render.sha256,download:`${song}_${method}_90-150s.mp4`});
    }
  }
  return {review,pins,clips};
}

export function snapshotExpression(){return `(()=>({song:document.querySelector('#song').value,master:document.querySelector('#master').value,source:Number(document.querySelector('#seek').value),min:Number(document.querySelector('#seek').min),max:Number(document.querySelector('#seek').max),clock:document.querySelector('#clock').value,status:document.querySelector('#status').textContent,cards:[...document.querySelectorAll('article')].map((a,i)=>{const v=a.querySelector('video'),link=a.querySelector('a');return {index:i,visible:!a.hidden,title:a.querySelector('h2').textContent,badge:a.querySelector('.warning').textContent,src:v.getAttribute('src'),download:link.download,href:link.getAttribute('href'),time:v.currentTime,paused:v.paused,muted:v.muted,ready:v.readyState,seeking:v.seeking,duration:Number.isFinite(v.duration)?v.duration:null,frames:v.getVideoPlaybackQuality().totalVideoFrames,error:v.error?{code:v.error.code,message:v.error.message}:null};})}))()`;}

export function validateState(s,song){
  const visible=s.cards.filter(c=>c.visible);check(s.cards.length===6&&visible.length===3,'Expected six cards and three visible');
  check(visible.every(c=>c.src.split('/')[1]===song),'Cross-song visible player');
  check(s.min===90&&s.max===150&&s.source>=90&&s.source<=150,'Wrong source clock range');
  check(s.cards.filter(c=>!c.muted).length===1&&visible.some(c=>!c.muted),'Expected one active audible master');
  check(s.cards.filter(c=>!c.visible).every(c=>c.paused),'Hidden song is playing');
  check(!s.status&&!s.cards.some(c=>c.error),'Gallery/media reported an error');
  for(const c of s.cards){const[method,track]=c.src.split('/');check(c.href===c.src&&c.download===`${track}_${method}_90-150s.mp4`,'Download link differs');check(c.badge===(method==='D7_reference'?'Experimental control':'Failure diagnostic'),'Wrong experimental status');}
  return visible;
}

class CDP{
  constructor(socket){this.socket=socket;this.next=0;this.pending=new Map();socket.addEventListener('message',event=>{const x=JSON.parse(event.data);if(x.id){const p=this.pending.get(x.id);if(!p)return;clearTimeout(p.timer);this.pending.delete(x.id);x.error?p.reject(Error(JSON.stringify(x.error))):p.resolve(x.result||{});}else this.onEvent?.(x);});}
  send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++this.next,timer=setTimeout(()=>{this.pending.delete(id);reject(Error('CDP timeout '+method));},5000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});}
  async evaluate(expression,session){const r=await this.send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true},session);check(!r.exceptionDetails,'Browser expression failed '+JSON.stringify(r.exceptionDetails));return r.result?.value;}
}
async function wait(predicate,deadline,label){while(performance.now()<deadline){const result=await predicate();if(result)return result;await delay(100);}throw Error('Browser deadline: '+label);}
async function routes(url,inputs){
  const r=await fetch(url+'/');check(r.ok,'Gallery HTTP failed');check(createHash('sha256').update(Buffer.from(await r.arrayBuffer())).digest('hex')===inputs.pins[path.join(inputs.review,'index.html')],'Served HTML differs');
  const checked=[];
  for(const c of inputs.clips){const head=await fetch(url+c.route+'?download=1',{method:'HEAD'});check(head.status===200&&Number(head.headers.get('content-length'))===c.bytes&&head.headers.get('content-type')==='video/mp4','HEAD mismatch');check(head.headers.get('content-disposition')===`attachment; filename="${c.download}"`,'Download header mismatch');
    const range=await fetch(url+c.route,{headers:{Range:'bytes=0-63'}});check(range.status===206&&range.headers.get('content-range')===`bytes 0-63/${c.bytes}`,'Range failed');const h=await fs.open(c.path,'r'),b=Buffer.alloc(64);try{await h.read(b,0,64,0);}finally{await h.close();}check(b.equals(Buffer.from(await range.arrayBuffer())),'Range bytes differ');checked.push({song:c.song,method:c.method,head:true,range:true});}
  return checked;
}

export async function run(args){
  await floor(args.root);const inputs=await authenticate(args.root,args.planSha256),dest=path.join(inputs.review,args.revision);await fs.mkdir(dest);
  const report={status:'running',scope:'Actual audited six-clip gallery mechanics; software-only Chrome; no artistic/scientific acceptance.',inputs:inputs.pins,checks:{},errors:[],decoder_properties:[],url:args.url,verifier_sha256:await digest(HERE)};
  let child,profile,cdp,session,watchdog,started;
  try{
    report.checks.routes=await routes(args.url,inputs);profile=await fs.mkdtemp('/tmp/lightsem-native3d-browser-');const downloads=path.join(profile,'downloads');await fs.mkdir(downloads);
    const flags=chromeArguments(profile);report.chrome_command=[args.chrome,...flags];started=performance.now();const deadline=started+args.timeoutMs;
    child=spawn(args.chrome,flags,{stdio:['ignore','ignore','pipe'],detached:true});let launchError;child.on('error',e=>launchError=e);let stderr='';child.stderr.on('data',x=>{stderr=(stderr+x.toString()).slice(-8000);});
    watchdog=setTimeout(()=>{report.deadline_exceeded=true;try{process.kill(-child.pid,'SIGTERM');}catch{}},args.timeoutMs);
    const endpoint=await wait(async()=>{if(launchError)throw launchError;check(child.exitCode===null&&child.signalCode===null,'Chrome exited '+stderr);try{return(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).trim().split('\n');}catch(e){if(e.code==='ENOENT')return false;throw e;}},deadline,'Chrome start');
    const socket=new WebSocket('ws://127.0.0.1:'+endpoint[0]+endpoint[1]);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});cdp=new CDP(socket);
    cdp.onEvent=e=>{if(e.method==='Runtime.exceptionThrown'||e.method==='Log.entryAdded'&&e.params.entry.level==='error'||e.method==='Network.loadingFailed'&&!e.params.canceled)report.errors.push({method:e.method,params:e.params});if(e.method==='Media.playerPropertiesChanged')for(const p of e.params.properties||[])if(/Decoder|Codec|Resolution/i.test(p.name))report.decoder_properties.push(p);};
    await cdp.send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads});const{targetId}=await cdp.send('Target.createTarget',{url:'about:blank'});({sessionId:session}=await cdp.send('Target.attachToTarget',{targetId,flatten:true}));
    for(const domain of ['Page','Runtime','Network','Log','Media'])await cdp.send(domain+'.enable',{},session);
    await cdp.send('Emulation.setDeviceMetricsOverride',{width:1600,height:1100,deviceScaleFactor:1,mobile:false},session);
    await cdp.send('Page.navigate',{url:args.url+'/'},session);
    const evaluate=x=>cdp.evaluate(x,session),snapshot=()=>evaluate(snapshotExpression()),click=id=>evaluate(`document.querySelector('#${id}').click()`),select=(id,value)=>evaluate(`(()=>{const e=document.querySelector('#${id}');e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
    await wait(async()=>await evaluate('document.querySelectorAll("article").length===6'),deadline,'six cards');
    report.checks.songs=[];
    for(const song of SONGS){
      await select('song',song);let s=await snapshot();const cards=validateState(s,song);check(s.source===90&&s.cards.every(c=>c.paused),'Song switch did not pause/reset');
      await wait(async()=>{const x=await snapshot();return x.cards.filter(c=>c.visible).every(c=>c.ready>=1&&Math.abs(c.duration-60)<.06&&c.time<.1);},deadline,'song metadata');
      await evaluate(`(()=>{const e=document.querySelector('#seek');e.value=135;e.dispatchEvent(new Event('input',{bubbles:true}));})()`);
      await wait(async()=>{const x=await snapshot();return x.cards.filter(c=>c.visible).every(c=>!c.seeking&&c.ready>=2&&Math.abs(c.time-45)<.15);},deadline,'seek135');
      const sought=await snapshot();check(sought.clock==='135.00 s'&&sought.source===135,'Displayed source seek mismatch');
      await click('play');
      await wait(async()=>{const x=await snapshot();return x.cards.filter(c=>c.visible).every(c=>!c.paused&&c.ready>=2&&!c.seeking);},deadline,'playing');
      const before=await snapshot();await delay(800);const after=await snapshot(),active=validateState(after,song);
      const progress=active.map(c=>{const old=before.cards[c.index];check(c.time-old.time>=.12&&c.frames>old.frames,'Actual media did not advance');return {method:c.src.split('/')[0],advance_s:c.time-old.time,decoded_frames:c.frames-old.frames};});
      const skew=Math.max(...active.map(c=>c.time))-Math.min(...active.map(c=>c.time));check(skew<=.35,'Playback skew exceeds .35 s');
      await select('master',String(cards[2].index));s=await snapshot();validateState(s,song);check(!s.cards[cards[2].index].muted,'Master handoff failed');
      await click('pause');s=await snapshot();check(s.cards.every(c=>c.paused),'Pause failed');
      // Real download click; temporary copy outside campaign disk, deleted after hashing.
      const chosen=inputs.clips.find(c=>c.song===song&&c.method==='D7_reference');await evaluate(`document.querySelectorAll('article')[${cards[0].index}].querySelector('a').click()`);
      const downloaded=path.join(downloads,chosen.download);await wait(async()=>{try{return(await fs.stat(downloaded)).size===chosen.bytes;}catch(e){if(e.code==='ENOENT')return false;throw e;}},deadline,'download');await pin(downloaded,chosen.sha256);await fs.unlink(downloaded);
      const shot=await cdp.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false},session),filename=path.join(dest,song+'.png');await fs.writeFile(filename,Buffer.from(shot.data,'base64'),{flag:'wx'});
      await click('restart');s=await snapshot();check(s.source===90&&s.cards.every(c=>c.paused),'Restart failed');
      report.checks.songs.push({song,source_seek_s:135,progress,max_skew_s:skew,master_handoff:true,pause_restart:true,download_sha256:chosen.sha256,screenshot:filename,screenshot_sha256:await digest(filename)});
    }
    report.final_snapshot=await snapshot();check(!report.errors.length,'Browser/network/console error');check(report.decoder_properties.filter(p=>/IsPlatformVideoDecoder/i.test(p.name)).every(p=>!['true','1'].includes(String(p.value).toLowerCase())),'Hardware video decoder used');check(!report.deadline_exceeded,'Browser deadline exceeded');report.status='passed';
  }catch(error){report.status='failed';report.error=error.message;if(cdp&&session&&!report.deadline_exceeded)try{report.failure_snapshot=await cdp.evaluate(snapshotExpression(),session);}catch{}}
  finally{clearTimeout(watchdog);if(started)report.browser_elapsed_ms=Math.round(performance.now()-started);if(cdp){try{await cdp.send('Browser.close');}catch{}cdp.socket.close();}if(child?.pid){for(let i=0;i<20&&child.exitCode===null&&child.signalCode===null;i++)await delay(50);if(child.exitCode===null&&child.signalCode===null)try{process.kill(-child.pid,'SIGKILL');}catch{}}if(profile)await fs.rm(profile,{recursive:true,force:true});}
  for(const[p,h]of Object.entries(inputs.pins))try{await pin(p,h);}catch(e){report.status='failed';report.input_drift=e.message;}
  await floor(args.root);const out=path.join(dest,'REPORT.json');await fs.writeFile(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log('NATIVE_3D_GALLERY_'+report.status.toUpperCase(),out);return report;
}
if(process.argv[1]&&path.resolve(process.argv[1])===HERE){try{const report=await run(parseArguments(process.argv.slice(2)));if(report.status!=='passed')process.exitCode=1;}catch(e){console.error(e);process.exitCode=1;}}
