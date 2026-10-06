"""CPU JavaScript/DOM state tests; no browser, GPU, network or media decoding."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


PAGE = Path(__file__).resolve().parents[1] / 'scripts/rendering/dynamic_dance_review.html'
NODE_HARNESS = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const payload=JSON.parse(fs.readFileSync(0,'utf8'));
class Classes {
  constructor(){this.values=new Set()}
  contains(name){return this.values.has(name)}
  toggle(name,state){const value=state===undefined?!this.values.has(name):state;if(value)this.values.add(name);else this.values.delete(name);return value}
}
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.style={};this.classList=new Classes();this.value='';this.checked=false;this.handlers=new Map();this.textContent=''}
  append(...items){this.children.push(...items)}
  replaceChildren(...items){this.children=[...items]}
  get options(){return this.children}
  setAttribute(key,value){this[key]=value}
  addEventListener(name,handler,options={}){const rows=this.handlers.get(name)||[];rows.push({handler,once:options.once});this.handlers.set(name,rows)}
  dispatch(name){const rows=[...(this.handlers.get(name)||[])];for(const row of rows){if(row.once)this.handlers.set(name,this.handlers.get(name).filter(r=>r!==row));row.handler.call(this,{type:name,target:this})}}
}
class Video extends Element {
  constructor(){super('video');this.paused=true;this.muted=true;this.currentTime=0;this.duration=120;this.playbackRate=1}
  play(){if(this.paused){this.paused=false;queueMicrotask(()=>this.dispatch('play'))}return Promise.resolve()}
  pause(){if(!this.paused){this.paused=true;queueMicrotask(()=>this.dispatch('pause'))}}
}
class Option extends Element {constructor(text,value){super('option');this.text=text;this.value=value}}
const elements=new Map();
for(const id of ['song','play','pause','audible','seek','clock','reset','status','gallery','export'])elements.set(id,new Element(['song','audible'].includes(id)?'select':'div'));
elements.get('audible').value='none';elements.get('song').value='all';elements.get('seek').value='0';
const tracks=[{id:'a',title:'Song A'},{id:'b',title:'Song B'}];
const variants=[['01','D1','a'],['02','D2','a'],['03','D3','a'],['04','D1','b']].map(([id,algorithm_id,track_id])=>({id,algorithm_id,track_id,title:algorithm_id+' / '+track_id,status:'complete',sha256:'video-'+id,bytes:1048576,download_name:id+'.mp4'}));
const manifestFixture={campaign:'cpu-fixture',plan_sha256:'cpu-plan',tracks,variants,review_start_s:60,review_duration_s:120};
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const context=vm.createContext({assert,flush,Option,Promise,Map,Set,JSON,Number,String,Date,Blob,URL,setTimeout,
  document:{getElementById:id=>elements.get(id),createElement:tag=>tag==='video'?new Video():new Element(tag)},
  localStorage:{getItem:()=>null,setItem:()=>{}},
  fetch:async()=>({ok:true,json:async()=>manifestFixture})});
(async()=>{
  vm.runInContext(payload.script,context,{filename:'dynamic_dance_review.html'});
  await flush();await flush();
  await vm.runInContext('(async()=>{'+payload.scenario+'})()',context,{filename:'gallery-scenario.js'});
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1});
"""


def run_gallery(scenario):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the CPU gallery event harness')
    script = PAGE.read_text().split('<script>', 1)[1].split('</script>', 1)[0]
    result = subprocess.run([node, '-e', NODE_HARNESS], input=json.dumps({'script': script, 'scenario': scenario}),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_deselecting_playing_master_transfers_sync_and_removes_its_audio():
    run_gallery(r"""
      $('play').onclick();await flush();
      const former=master,formerId=[...players].find(([,v])=>v===former)[0];
      $('audible').value=formerId;audio();assert.equal(former.muted,false);
      checks.get(formerId).checked=false;selectChange(formerId);await flush();
      const [remainingId,remaining]=selected()[0];
      assert.equal(master,remaining,'remaining selected player must become master');
      assert.equal(former.paused,true);assert.equal(remaining.paused,false);
      assert.equal($('audible').value,'none');assert.equal(former.muted,true);
      remaining.currentTime=37;remaining.dispatch('timeupdate');
      assert.equal(seekTime,37);assert.equal(Number($('seek').value),37);
      $('audible').value=remainingId;audio();
      assert.deepEqual([...players].filter(([,v])=>!v.muted).map(([id])=>id),[remainingId]);
    """)


def test_filtering_clears_hidden_master_and_preserves_shared_time():
    run_gallery(r"""
      $('play').onclick();await flush();
      master.currentTime=42;master.dispatch('timeupdate');
      const previousId=[...players].find(([,v])=>v===master)[0];
      $('audible').value=previousId;audio();
      $('song').value='b';applyFilter();await flush();
      assert.equal(master,null,'hidden master must be released when new song has no selection');
      assert.equal(seekTime,42);assert.equal($('audible').value,'none');
      assert.equal([...players.values()].every(v=>v.paused&&v.muted),true);
      checks.get('04').checked=true;selectChange('04');await flush();
      assert.equal(master,players.get('04'));assert.equal(master.currentTime,42);
      $('audible').value='04';audio();$('play').onclick();await flush();
      master.currentTime=55;master.dispatch('timeupdate');
      assert.equal(seekTime,55);assert.equal(players.get('04').muted,false);
      assert.equal([...players].filter(([id])=>id!=='04').every(([,v])=>v.muted),true);
    """)


def test_master_handoff_preserves_a_different_selected_audio_source():
    run_gallery(r"""
      checks.get('03').checked=true;selectChange('03');
      $('play').onclick();await flush();
      assert.equal(master,players.get('01'),'programmatic follower play must not steal master');
      $('audible').value='03';audio();
      checks.get('01').checked=false;selectChange('01');await flush();
      assert.equal(master,players.get('02'));assert.equal($('audible').value,'03');
      master.currentTime=29;master.dispatch('timeupdate');
      assert.equal(players.get('03').currentTime,29);
      assert.deepEqual([...players].filter(([,v])=>!v.muted).map(([id])=>id),['03']);
      checks.get('03').checked=false;selectChange('03');await flush();
      assert.equal(master,players.get('02'));assert.equal(master.paused,false);
      assert.equal($('audible').value,'none');assert.equal([...players.values()].every(v=>v.muted),true);
    """)


def test_deselecting_last_master_preserves_seek_and_rejects_stale_source_audio():
    run_gallery(r"""
      checks.get('02').checked=false;selectChange('02');
      $('play').onclick();await flush();master.currentTime=21.5;
      $('audible').value='01';audio();
      checks.get('01').checked=false;selectChange('01');await flush();
      assert.equal(master,null);assert.equal(seekTime,21.5);
      players.get('01').currentTime=90;players.get('01').dispatch('timeupdate');
      assert.equal(seekTime,21.5);
      $('audible').value='01';audio();
      assert.equal($('audible').value,'none');assert.equal([...players.values()].every(v=>v.muted),true);
    """)


def test_hidden_late_play_cannot_retake_master_or_audio():
    run_gallery(r"""
      $('play').onclick();await flush();master.currentTime=17;
      checks.get('04').checked=true;
      $('song').value='b';applyFilter();await flush();
      assert.equal(master,players.get('04'));assert.equal(master.currentTime,17);
      $('audible').value='04';audio();$('play').onclick();await flush();
      players.get('01').paused=false;players.get('01').dispatch('play');await flush();
      assert.equal(master,players.get('04'));assert.equal(players.get('01').paused,true);
      assert.equal(players.get('01').muted,true);assert.equal(players.get('04').muted,false);
    """)


def test_filter_pause_is_not_undone_by_queued_play_events():
    run_gallery(r"""
      $('play').onclick();
      $('song').value='all';applyFilter();await flush();
      assert.equal([...players.values()].every(v=>v.paused),true,'queued old play events must not restart stopped group');
    """)
