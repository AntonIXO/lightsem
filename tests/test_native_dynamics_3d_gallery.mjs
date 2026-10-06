import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.value='';this.currentTime=0;this.readyState=1;this.muted=true;this.paused=true;this.hidden=false;}
  append(...items){this.children.push(...items);if(this.tag==='select'&&this.children.length)this.value=this.children[0].value;}
  replaceChildren(){this.children=[];this.value='';}
  play(){this.paused=false;return Promise.resolve();}
  pause(){this.paused=true;}
}
const ids={};
for(const name of ['grid','master','seek','clock','status','song','play','pause','restart'])ids[name]=new Element(['master','song'].includes(name)?'select':name);
ids.song.value='shpongle';
const document={querySelector:selector=>ids[selector.slice(1)],createElement:tag=>new Element(tag),createTextNode:text=>text};
const clips=['shpongle','cure_for_me'].flatMap(song=>['D7_reference','C3','C4'].map((method,index)=>({song,method,label:['A control','B candidate','C candidate'][index],src:'test.mp4'})));
const html=fs.readFileSync(new URL('../scripts/rendering/native_dynamics_3d_review.html',import.meta.url),'utf8');
const script=html.split('<script>')[1].split('</script>')[0].replace('__CLIPS_JSON__',JSON.stringify(clips));
const context=vm.createContext({document,setInterval:()=>0,console});
vm.runInContext(script,context);
const run=expression=>vm.runInContext(expression,context);
assert.equal(run('activeVideos().length'),3);
assert.equal(run('articles.filter(a=>!a.hidden).length'),3);
await run('play()');
assert.equal(run('videos.filter(v=>!v.paused).length'),3);
assert.equal(run('videos.filter(v=>!v.muted).length'),1);
run('videos.forEach(v=>v.currentTime=12);songSelect.value="cure_for_me";switchSong()');
assert.equal(run('videos.every(v=>v.paused&&v.currentTime===0)'),true);
assert.equal(run('activeIndices().join(",")'),'3,4,5');
assert.equal(run('videos.filter(v=>!v.muted).length'),1);
assert.equal(run('primary()===videos[3]'),true);
await run('play()');
assert.equal(run('videos.slice(0,3).every(v=>v.paused)'),true);
assert.equal(run('videos.slice(3).every(v=>!v.paused)'),true);
console.log('Active-song playback, resets, visibility and single audible master passed');
