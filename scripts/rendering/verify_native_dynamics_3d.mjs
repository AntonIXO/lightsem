import {spawn} from 'node:child_process';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import {chromeArguments,digest,check} from './verify_dynamic_combo_gallery.mjs';
class CDP{
  constructor(socket){this.socket=socket;this.next=0;this.pending=new Map();socket.addEventListener('message',event=>{const value=JSON.parse(event.data);if(value.id){const p=this.pending.get(value.id);if(!p)return;clearTimeout(p.timer);this.pending.delete(value.id);value.error?p.reject(Error(JSON.stringify(value.error))):p.resolve(value.result||{});}else this.onEvent?.(value);});
    socket.addEventListener('close',()=>{for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(Error('Owned Chrome CDP closed'));}this.pending.clear();});}
  send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++this.next,timer=setTimeout(()=>{this.pending.delete(id);reject(Error('CDP timeout: '+method));},5000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});}
  async evaluate(expression,session){const r=await this.send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true,userGesture:true},session);check(!r.exceptionDetails,'Browser expression failed: '+JSON.stringify(r.exceptionDetails));return r.result?.value;}
}

async function waitFor(predicate,deadline,label){while(performance.now()<deadline){const value=await predicate();if(value)return value;await delay(75);}throw Error('Browser deadline: '+label);}


const root=path.resolve(process.argv[2]),url=new URL(process.argv[3]).origin;
check(root.endsWith('/preview_3d_v2')&&/^http:\/\/(127\.0\.0\.1|localhost):[0-9]+\/?$/.test(url),'Explicit local preview root/origin required');
const out=path.join(path.dirname(root),'delivery_verification/preview3d_browser_v1');await fs.mkdir(out,{recursive:true});
const report={status:'running',software_browser:true,baseline_promoted:false,checks:[],inputs:{},errors:[]};
let child,profile,cdp,session,watchdog;
try{
 const disk=await fs.statfs(root,{bigint:true});check(disk.bavail*disk.bsize>=20n*1024n**3n,'20GiB reserve');
 const manifest=JSON.parse(await fs.readFile(path.join(root,'MANIFEST.json'),'utf8'));check(manifest.variants.length===6,'Six clips required');
 for(const row of manifest.variants){check(await digest(row.path)===row.sha256,'Changed movie');report.inputs[row.path]=row.sha256;const response=await fetch(url+row.route,{headers:{Range:'bytes=0-63'}});check(response.status===206,'Range seeking unavailable');check((await response.arrayBuffer()).byteLength===64,'Wrong Range bytes');const head=await fetch(url+row.route+'?download=1',{method:'HEAD'});check(head.status===200&&Number(head.headers.get('content-length'))===row.bytes&&head.headers.get('content-disposition')?.includes(row.download_name),'Download identity');}
 report.inputs[path.join(root,'index.html')]=await digest(path.join(root,'index.html'));
 profile=await fs.mkdtemp(path.join(tmpdir(),'native-3d-browser-'));child=spawn('/opt/google/chrome/chrome',[...chromeArguments(profile),'--disk-cache-size=1048576','--media-cache-size=1048576'],{stdio:['ignore','ignore','pipe'],detached:true});
 child.stderr.on('data',()=>{});
 const deadline=performance.now()+45000;watchdog=setTimeout(()=>{try{process.kill(-child.pid,'SIGTERM')}catch{}},45000);
 const endpoint=await waitFor(async()=>{try{return(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).trim().split('\n')}catch{return false}},deadline,'Chrome');
 const socket=new WebSocket('ws://127.0.0.1:'+endpoint[0]+endpoint[1]);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true})});cdp=new CDP(socket);
 cdp.onEvent=e=>{if(e.method==='Runtime.exceptionThrown')report.errors.push(e.params)};
 const target=await cdp.send('Target.createTarget',{url:'about:blank'});({sessionId:session}=await cdp.send('Target.attachToTarget',{targetId:target.targetId,flatten:true}));await cdp.send('Page.enable',{},session);await cdp.send('Runtime.enable',{},session);await cdp.send('Page.navigate',{url},session);
 const ev=x=>cdp.evaluate(x,session),snap=()=>ev(`({song:document.querySelector('#song').value,clock:document.querySelector('#clock').value,status:document.querySelector('#status').textContent, videos:[...document.querySelectorAll('video')].map(v=>({visible:!v.closest('article').hidden,time:v.currentTime,paused:v.paused,muted:v.muted,ready:v.readyState,duration:v.duration,error:v.error?.message||null,frames:v.getVideoPlaybackQuality().totalVideoFrames})),downloads:[...document.querySelectorAll('a')].map(a=>a.download)})`);
 await waitFor(async()=>{try{const s=await snap();return s.videos.length===6&&s.videos.every(v=>v.ready>=1)&&s}catch{return false}},deadline,'six metadata');
 for(const song of ['shpongle','cure_for_me']){
  await ev(`(()=>{const s=document.querySelector('#song');s.value='${song}';s.dispatchEvent(new Event('change'));return true})()`);
  let s=await snap();check(s.videos.filter(v=>v.visible).length===3&&s.videos.every(v=>v.paused)&&Number.parseFloat(s.clock)===90,'Song reset');
  await ev(`document.querySelector('#play').click()`);await delay(1000);s=await snap();let active=s.videos.filter(v=>v.visible);check(active.every(v=>!v.paused&&v.time>.12&&v.frames>0)&&active.filter(v=>!v.muted).length===1&&s.videos.filter(v=>!v.visible).every(v=>v.paused&&v.muted),'Only active3 playing/oneaudio');check(Math.max(...active.map(v=>v.time))-Math.min(...active.map(v=>v.time))<.35,'Playback sync');
  await ev(`(()=>{const m=document.querySelector('#master');m.selectedIndex=2;m.dispatchEvent(new Event('change'));return true})()`);s=await snap();check(s.videos.filter(v=>!v.muted).length===1,'Audio handoff');
  await ev(`(()=>{const x=document.querySelector('#seek');x.value=135;x.dispatchEvent(new Event('input'));return true})()`);await delay(700);s=await snap();active=s.videos.filter(v=>v.visible);check(active.every(v=>v.time>=44.8&&v.time<48),'Source135s seek');
  await ev(`document.querySelector('#pause').click()`);s=await snap();check(s.videos.every(v=>v.paused),'Pause all');check(s.videos.every(v=>!v.error)&&!s.status,'Media errors');report.checks.push({song,passed:true,state:s});
 }
 check(report.errors.length===0,'JS errors');check((await snap()).downloads.every(Boolean),'Download controls');
 const shot=await cdp.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true},session);await fs.writeFile(path.join(out,'gallery.png'),Buffer.from(shot.data,'base64'));report.status='passed';
}catch(error){report.status='failed';report.error=String(error)}
finally{clearTimeout(watchdog);if(cdp){try{await cdp.send('Browser.close')}catch{}cdp.socket.close()}if(child?.pid){try{process.kill(-child.pid,'SIGTERM')}catch{}}await delay(100);if(profile)await fs.rm(profile,{recursive:true,force:true});}
await fs.writeFile(path.join(out,'REPORT.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log('NATIVE_3D_BROWSER',report.status,out);if(report.status!=='passed')process.exitCode=1;
