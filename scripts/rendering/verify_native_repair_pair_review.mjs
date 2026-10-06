/** Real software-browser validation of the direct pair plots and unchanged media. */
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const args={};
for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].slice(2)]=process.argv[i+1];
if(!args.root||!args.url||!args.revision||!args['plan-sha256'])throw Error('Require root, url, revision, plan-sha256');
const helpers=path.join(args.root,'delivery_tools_v1');
const {authenticate}=await import(pathToFileURL(path.join(helpers,'verify_unified_review.mjs')));
const {chromeArguments,digest,check}=await import(pathToFileURL(path.join(helpers,'browser_common.mjs')));
const inputs=await authenticate(args.root,args['plan-sha256']);
const manifest=inputs.manifest,direct=manifest.direct_comparison;
check(direct,'No authenticated direct comparison publication');
check(await digest(direct.audit_path)===direct.audit_sha256,'Comparison audit hash changed');
const audit=JSON.parse(await fs.readFile(direct.audit_path,'utf8'));
for(const graph of direct.graphs){
 check(await digest(graph.path)===graph.sha256,'Graph hash changed');
 check(Object.values(graph.measurement).every(row=>row.clipped_samples===0),'Graph clips raw samples');
}
const dest=path.join(inputs.review,args.revision);await fs.mkdir(dest);
const report={status:'running',url:args.url,presentation_revision:manifest.revision,checks:[],errors:[],unchanged_media:inputs.clips.map(c=>({method:c.method,song:c.song,sha256:c.sha256})),verifier_sha256:await digest(import.meta.filename)};
let child,profile,cdp,session,watchdog;
class CDP{
 constructor(socket){this.socket=socket;this.next=0;this.pending=new Map();socket.addEventListener('message',event=>{const x=JSON.parse(event.data);if(x.id){const p=this.pending.get(x.id);if(!p)return;clearTimeout(p.timer);this.pending.delete(x.id);x.error?p.reject(Error(JSON.stringify(x.error))):p.resolve(x.result||{})}else this.onEvent?.(x)})}
 send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++this.next,timer=setTimeout(()=>{this.pending.delete(id);reject(Error('CDP timeout '+method))},5000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}))})}
 async evaluate(expression){const result=await this.send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true},session);check(!result.exceptionDetails,'Browser expression failed '+JSON.stringify(result.exceptionDetails));return result.result?.value}
}
const start=performance.now(),deadline=start+45000;
async function wait(predicate,label){while(performance.now()<deadline){const result=await predicate();if(result)return result;await delay(75)}throw Error('Browser deadline: '+label)}
try{
 const response=await fetch(args.url+'/');check(response.ok&&createHash('sha256').update(Buffer.from(await response.arrayBuffer())).digest('hex')===await digest(path.join(inputs.review,'index.html')),'Served page differs');
 profile=await fs.mkdtemp('/tmp/lightsem-native-pair-browser-');
 child=spawn('/opt/google/chrome/chrome',chromeArguments(profile),{stdio:['ignore','ignore','pipe'],detached:true});
 watchdog=setTimeout(()=>{report.deadline_exceeded=true;try{process.kill(-child.pid,'SIGTERM')}catch{}},45000);
 const endpoint=await wait(async()=>{check(child.exitCode===null&&child.signalCode===null,'Chrome exited');try{return(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).trim().split('\n')}catch(error){if(error.code==='ENOENT')return false;throw error}},'Chrome start');
 const socket=new WebSocket('ws://127.0.0.1:'+endpoint[0]+endpoint[1]);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true})});cdp=new CDP(socket);
 cdp.onEvent=event=>{if(event.method==='Runtime.exceptionThrown'||event.method==='Network.loadingFailed'&&!event.params.canceled)report.errors.push(event)};
 const {targetId}=await cdp.send('Target.createTarget',{url:'about:blank'});({sessionId:session}=await cdp.send('Target.attachToTarget',{targetId,flatten:true}));
 for(const domain of ['Page','Runtime','Network'])await cdp.send(domain+'.enable',{},session);
 await cdp.send('Emulation.setDeviceMetricsOverride',{width:1680,height:1400,deviceScaleFactor:1,mobile:false},session);
 await cdp.send('Page.navigate',{url:args.url+'/'},session);
 const evaluate=x=>cdp.evaluate(x),select=(id,value)=>evaluate(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
 await wait(()=>evaluate('document.querySelectorAll("article").length===6 && document.getElementById("pair-plot").complete'), 'comparison load');
 for(const song of ['shpongle','cure_for_me']){
  await select('song',song);
  const checks=[];
  for(const scope of ['full_song','90_150']){
   await select('pair-window',scope);
   for(const kind of ['difference','lights','body']){
    await select('pair-kind',kind);
    await wait(()=>evaluate('document.getElementById("pair-plot").complete&&document.getElementById("pair-plot").naturalWidth>0'),'image decode');
    const state=await evaluate(`(async()=>{const e=document.getElementById('pair-plot');const bytes=await(await fetch(e.src)).arrayBuffer();const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(n=>n.toString(16).padStart(2,'0')).join('');return {hash,dataset:{...e.dataset},width:e.naturalWidth,height:e.naturalHeight,table:[...document.querySelectorAll('#pair-measures tbody tr')].map(row=>[...row.children].map(cell=>cell.textContent)),identity:document.getElementById('pair-identity').textContent,caption:document.getElementById('pair-caption').textContent}})()`);
    const graph=direct.graphs.find(g=>g.song===song&&g.scope===scope&&g.kind===kind);
    check(state.hash===graph.sha256&&state.dataset.sha256===graph.sha256&&state.dataset.song===song&&state.dataset.scope===scope&&state.dataset.kind===kind,'Wrong rendered graph identity');
    const keys=kind==='lights'?['requested','emitted','brightness']:['left_arm','right_arm','lower','body_jerk','root_y','root_vertical_jerk'];
    const measurements=audit.songs.find(s=>s.song===song).windows[scope].measurements;
    check(state.table.length===keys.length,'Missing measurement rows');
    for(let i=0;i<keys.length;i++){
     const m=measurements[keys[i]],number=x=>Number(x).toLocaleString('en-US',{maximumSignificantDigits:4});
     check(state.table[i][0]===m.label&&state.table[i][1]===number(m.old_mean)&&state.table[i][2]===number(m.repaired_mean),'UI numbers differ from saved raw measurements');
    }
    checks.push({scope,kind,sha256:graph.sha256,decoded:true,table_matches_raw_report:true});
   }
  }
  await evaluate(`(()=>{const e=document.getElementById('pair-seek');e.value=122.04;e.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  await wait(()=>evaluate(`(()=>{const v=[...document.querySelectorAll('article')].filter(e=>!e.hidden).map(e=>e.querySelector('video'));return v.length===3&&v.every(e=>e.readyState>=2&&!e.seeking&&Math.abs(e.currentTime-122.04)<.15)})()`),'synchronized detail seek');
  const videos=()=>evaluate(`([...document.querySelectorAll('article')].filter(e=>!e.hidden).map(e=>({method:e.dataset.method,time:e.querySelector('video').currentTime,frames:e.querySelector('video').getVideoPlaybackQuality().totalVideoFrames,error:e.querySelector('video').error?.message||null,src:e.querySelector('video').getAttribute('src')})))`);
  const before=await videos();await evaluate('document.getElementById("play").click()');
  await wait(()=>evaluate(`([...document.querySelectorAll('article')].filter(e=>!e.hidden).every(e=>!e.querySelector('video').paused&&!e.querySelector('video').seeking))`),'real playback');
  await delay(800);const after=await videos();await evaluate('document.getElementById("pause").click()');
  check(after.every((v,i)=>!v.error&&v.time>before[i].time+.1&&v.frames>before[i].frames),'Actual native movie did not advance');
  const skew=Math.max(...after.map(v=>v.time))-Math.min(...after.map(v=>v.time));check(skew<=.35,'Video sync exceeds350ms');
  for(const video of after)check(video.src===manifest.clips.find(c=>c.song===song&&c.method===video.method).src,'Changed displayed movie');
  await evaluate(`document.getElementById('pair-comparison').scrollIntoView({block:'start'})`);
  const rect=await evaluate(`(()=>{const r=document.getElementById('pair-comparison').getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()`);
  const shot=await cdp.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:rect},session);
  const screenshot=path.join(dest,song+'.png');await fs.writeFile(screenshot,Buffer.from(shot.data,'base64'),{flag:'wx'});
  report.checks.push({song,graphs:checks,shared_video_seek:true,actual_playback:after,max_skew_s:skew,screenshot,screenshot_sha256:await digest(screenshot)});
 }
 await evaluate(`document.getElementById('diagnostic').open=true;loadDiagnosticGraphs()`);
 check(await evaluate('document.querySelectorAll("#diagnostic-graphs img").length===3'),'Historical plots disappeared');
 await select('song','shpongle');
 check(!report.errors.length,'Browser errors recorded');report.status='passed';
}catch(error){report.status='failed';report.error=error.stack}
finally{
 clearTimeout(watchdog);if(cdp){try{await cdp.send('Browser.close')}catch{}cdp.socket.close()}
 if(child?.pid){for(let i=0;i<20&&child.exitCode===null&&child.signalCode===null;i++)await delay(50);if(child.exitCode===null&&child.signalCode===null)try{process.kill(-child.pid,'SIGKILL')}catch{}}
 if(profile)await fs.rm(profile,{recursive:true,force:true});report.elapsed_ms=performance.now()-start;
}
for(const [filename,hash] of Object.entries(inputs.pins))if(await digest(filename)!==hash){report.status='failed';report.drift=filename}
await fs.writeFile(path.join(dest,'REPORT.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({status:report.status,error:report.error,path:path.join(dest,'REPORT.json')},null,2));
if(report.status!=='passed')process.exitCode=1;
