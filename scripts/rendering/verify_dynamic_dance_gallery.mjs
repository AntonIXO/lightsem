#!/usr/bin/env node
/** Real-media gallery acceptance through an isolated, software-only Chrome CDP.
 * No packages, test media, app modifications, public hosting, or GPU rendering.
 * node scripts/rendering/verify_dynamic_dance_gallery.mjs --root ROOT
 *   --plan-sha256 FROZEN_DIGEST --revision browser_v1 --url http://127.0.0.1:8775
 * The 30-second default bounds the live browser checks, excluding input hashing.
 */
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import * as fs from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const HERE = fileURLToPath(import.meta.url);
const MIN_FREE = 20n * 1024n ** 3n;
const WRITE_ALLOWANCE = 128n * 1024n ** 2n;
const METHODS = Array.from({length: 8}, (_, i) => `D${i + 1}`);
const SONGS = ['shpongle', 'cure_for_me'];

export function check(condition, message) {
  if (!condition) throw new Error(message);
}

export function validateOrigin(value) {
  const url = new URL(value);
  check(url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)
    && !url.username && !url.password && !url.search && !url.hash && url.pathname === '/',
  'Expected a plain loopback HTTP origin');
  return url.origin;
}

export function parseArguments(argv) {
  const result = {url: 'http://127.0.0.1:8775', revision: 'browser_v1', chrome: '/opt/google/chrome/chrome', timeoutMs: 30000};
  const names = {'--root': 'root', '--plan-sha256': 'planSha256', '--revision': 'revision',
    '--url': 'url', '--chrome': 'chrome', '--timeout-ms': 'timeoutMs'};
  for (let i = 0; i < argv.length; i += 2) {
    check(names[argv[i]] && argv[i + 1] && !argv[i + 1].startsWith('--'), 'Unknown option or missing value: ' + argv[i]);
    result[names[argv[i]]] = argv[i + 1];
  }
  check(result.root && /^[a-f0-9]{64}$/.test(result.planSha256 || ''), 'Require --root and externally frozen --plan-sha256');
  check(/^[A-Za-z0-9_-]+$/.test(result.revision), 'Unsafe verification revision');
  result.root = path.resolve(result.root);
  result.url = validateOrigin(result.url);
  result.timeoutMs = Number(result.timeoutMs);
  check(Number.isInteger(result.timeoutMs) && result.timeoutMs >= 15000 && result.timeoutMs <= 60000,
    'Browser timeout must be 15000–60000 ms');
  return result;
}

export function chromeArguments(profile) {
  return ['--headless=new', '--no-sandbox', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-extensions',
    '--disable-dev-shm-usage', '--disable-gpu', '--disable-accelerated-video-decode',
    '--disable-accelerated-video-encode', '--use-gl=angle', '--use-angle=swiftshader',
    '--disable-features=Vulkan,VaapiVideoDecoder,VaapiVideoEncoder,AcceleratedVideoDecodeLinuxGL',
    '--autoplay-policy=no-user-gesture-required', '--mute-audio', '--window-size=1600,1100',
    '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'];
}

export async function digest(filename) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
async function writeNew(filename, value) {
  await fs.writeFile(filename, JSON.stringify(value, null, 2) + '\n', {flag: 'wx'});
}
async function floor(root) {
  const stat = await fs.statfs(root, {bigint: true});
  check(stat.bavail * stat.bsize >= MIN_FREE + WRITE_ALLOWANCE, '20 GiB reserve plus browser evidence allowance reached');
}
async function pinned(filename, expected) {
  check(/^[0-9a-f]{64}$/.test(expected || '') && await digest(filename) === expected, 'SHA256 mismatch: ' + filename);
}

export async function authenticateInputs(root, planSha256) {
  await pinned(path.join(root, 'PLAN.json'), planSha256);
  const manifestPath = path.join(root, 'review/MANIFEST.json');
  const htmlPath = path.join(root, 'review/index.html');
  const manifest = await read(manifestPath);
  check(manifest.plan_sha256 === planSha256 && manifest.review_start_s === 60 && manifest.review_duration_s === 120,
    'Gallery PLAN/crop mismatch');
  check(manifest.variants?.length === 16, 'Expected all sixteen real verified media rows');
  const mediaRoot = await fs.realpath(path.join(root, 'media'));
  const pins = {[path.join(root, 'PLAN.json')]: planSha256, [manifestPath]: await digest(manifestPath),
    [htmlPath]: await digest(htmlPath)};
  for (let index = 0; index < 16; index++) {
    const row = manifest.variants[index];
    const method = METHODS[Math.floor(index / 2)], song = SONGS[index % 2];
    check(row.id === String(index + 1).padStart(2, '0') && row.algorithm_id === method && row.track_id === song
      && row.status === 'complete', 'Unexpected or incomplete media row: ' + row.id);
    const real = await fs.realpath(row.path);
    check(real.startsWith(mediaRoot + path.sep) && path.basename(real) === manifest.mode + '.mp4',
      'Video path is outside the allowlisted media tree');
    check(real === await fs.realpath(path.join(root, 'media', method, song, manifest.mode + '.mp4')),
      'Wrong method/song movie');
    await pinned(real, row.sha256);
    check((await fs.stat(real)).size === row.bytes, 'Video byte count mismatch');
    const verifiedPath = path.join(path.dirname(real), manifest.mode + '_verified.json');
    const verified = await read(verifiedPath);
    check(verified.video_sha256 === row.sha256 && verified.audio_matches_source_crop === true, 'Movie lacks matching verification receipt');
    await pinned(path.join(path.dirname(real), manifest.mode + '_audit.json'), verified.audit_sha256);
    pins[real] = row.sha256;
    pins[verifiedPath] = await digest(verifiedPath);
    pins[path.join(path.dirname(real), manifest.mode + '_audit.json')] = verified.audit_sha256;
  }
  return {manifest, pins};
}

async function verifyServedBytes(origin, filename, route) {
  const response = await fetch(origin + route, {redirect: 'error', signal: AbortSignal.timeout(5000)});
  check(response.ok, 'HTTP gallery unavailable: ' + route + ' ' + response.status);
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of response.body) {
    bytes += chunk.length;
    check(bytes <= 16 * 1024 ** 2, 'Unexpected oversized gallery document');
    hash.update(chunk);
  }
  check(hash.digest('hex') === await digest(filename), 'Served gallery differs from pinned file: ' + route);
}

export class CDP {
  constructor(socket, timeout = 5000) {
    this.socket = socket; this.timeout = timeout; this.nextId = 0; this.pending = new Map(); this.events = [];
    socket.addEventListener('message', message => {
      const data = JSON.parse(message.data);
      if (data.id) {
        const pending = this.pending.get(data.id);
        if (!pending) return;
        this.pending.delete(data.id); clearTimeout(pending.timer);
        if (data.error) pending.reject(new Error('CDP ' + pending.method + ': ' + JSON.stringify(data.error)));
        else pending.resolve(data.result || {});
      } else {
        if (this.events.length < 2000) this.events.push(data);
        this.onEvent?.(data);
      }
    });
    socket.addEventListener('close', () => {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer); pending.reject(new Error('Owned Chrome CDP connection closed'));
      }
      this.pending.clear();
    });
  }
  send(method, params = {}, sessionId) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {this.pending.delete(id); reject(new Error('CDP timeout: ' + method));}, this.timeout);
      this.pending.set(id, {resolve, reject, timer, method});
      this.socket.send(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}));
    });
  }
  async evaluate(expression, sessionId) {
    const result = await this.send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true, userGesture: true}, sessionId);
    check(!result.exceptionDetails, 'Page evaluation failed: ' + JSON.stringify(result.exceptionDetails));
    return result.result?.value;
  }
}

export function snapshotExpression({full = false} = {}) {
  return `(() => ({
  navigation: performance.timeOrigin,
  status: document.getElementById('status')?.textContent,
  song: document.getElementById('song')?.value,
  audible: document.getElementById('audible')?.value,
  seek: Number(document.getElementById('seek')?.value),
  masterId: typeof master !== 'undefined' && master ? master.closest('.card')?.id.slice(8) : null,
  cards: [...document.querySelectorAll('.card')].map(card => ({
    id: card.id.slice(8), visible: !card.classList.contains('hidden'),
    ${full ? `
    title: card.querySelector('h2')?.textContent, badge: card.querySelector('.badge')?.textContent,
    text: card.textContent, evidence: card.querySelector('pre')?.textContent,
    links: [...card.querySelectorAll('a')].map(a => ({href: a.getAttribute('href'), download: a.getAttribute('download')})),
    ` : ''}
    video: (() => {const v = card.querySelector('video'); return v ? {
      src: v.getAttribute('src'), currentSrc: v.currentSrc, paused: v.paused, time: v.currentTime,
      duration: Number.isFinite(v.duration) ? v.duration : null, muted: v.muted, seeking: v.seeking,
      ready: v.readyState, frames: v.getVideoPlaybackQuality?.().totalVideoFrames ?? null,
      checked: card.querySelector('input[type=checkbox]')?.checked,
      error: v.error ? {code: v.error.code, message: v.error.message} : null
    } : null})()
  })),
  seekEventCounts: window.__galleryCheckSeekingCounts || {}
}))()`;
}

const SNAPSHOT = snapshotExpression();
const FULL_DOM_SNAPSHOT = snapshotExpression({full: true});
const MEDIA_EVENTS = 'window.__galleryCheckMediaEvents || []';

export function validateDOMSnapshot(snapshot, manifest) {
  check(snapshot.cards.length === 16, 'Browser did not create all sixteen cards');
  for (const row of manifest.variants) {
    const card = snapshot.cards.find(card => card.id === row.id);
    check(card && card.title === row.algorithm_id + ' | ' + row.title, 'Wrong gallery card identity');
    const evidence = JSON.parse(card.evidence);
    check(evidence.video_sha256 === row.sha256 && evidence.comparison_sha256 === row.comparison_sha256,
      'Browser evidence differs from pinned manifest: ' + row.id);
    check(card.text.includes('Physical engineering: ' + (row.physical_engineering_status || 'unqualified')),
      'Engineering status missing from card: ' + row.id);
    if (row.duplicate_of) {
      check(!card.video && card.badge.includes('representative ' + row.duplicate_of)
        && card.links.some(link => link.href === '#variant-' + row.duplicate_of), 'Duplicate representative disclosure missing');
    } else {
      check(card.video && card.video.src === '/media/' + row.id + '.mp4?v=' + encodeURIComponent(row.sha256),
        'Browser player does not use authenticated movie: ' + row.id);
      check(card.badge === row.banner, 'Browser failure/evidence banner differs from manifest: ' + row.id);
      check(card.links.some(link => link.href === '/media/' + row.id + '.mp4?download=1&v=' + encodeURIComponent(row.sha256)
        && link.download === row.download_name), 'Missing authenticated download link: ' + row.id);
    }
    for (const reason of row.failure_notes || []) check(card.text.includes(reason), 'Failure reason omitted: ' + row.id);
    check(!card.video?.error, 'Browser media error: ' + JSON.stringify(card.video?.error));
  }
}

export function assertPlaybackProgress(before, after, ids, skewLimit = .35) {
  const audible = after.cards.filter(card => card.video?.muted === false);
  check(audible.length <= 1 && audible.every(card => ids.includes(card.id)), 'Unselected or multiple audio sources are unmuted');
  const values = ids.map(id => {
    const old = before.cards.find(card => card.id === id)?.video;
    const now = after.cards.find(card => card.id === id)?.video;
    check(old && now && !now.error && !now.paused && now.ready >= 2 && !now.seeking, 'Selected video not playing: ' + id);
    check(now.duration !== null && Math.abs(now.duration - 120) <= .06, 'Browser video duration mismatch: ' + id);
    check(now.time - old.time >= .12, 'Selected real media did not advance: ' + id);
    if (old.frames !== null && now.frames !== null) check(now.frames > old.frames, 'Browser decoded no new frames: ' + id);
    return {id, advance_s: now.time - old.time, decoded_frames: old.frames === null ? null : now.frames - old.frames, time_s: now.time};
  });
  const skew = Math.max(...values.map(v => v.time_s)) - Math.min(...values.map(v => v.time_s));
  check(skew <= skewLimit, 'Selected players diverged by ' + skew + ' seconds');
  return {players: values, max_skew_s: skew, limit_s: skewLimit};
}

export async function sampleSettledPlayback(snapshot, ids, deadline, telemetry = {},
  {clock = () => performance.now(), sleep = delay} = {}) {
  // Observe readiness only. Do not repair players, extend the shared deadline,
  // wait for favorable progress/skew, or retry failed progress assertions.
  telemetry.requested_sample_ms = 700;
  telemetry.before = {polls: 0, seeking_poll_observations: {}, settling_ms: 0};
  telemetry.after = {polls: 0, seeking_poll_observations: {}, settling_ms: 0};
  const started = clock();
  let initialCounts;
  const settled = async phase => {
    const stats = telemetry[phase], begin = clock();
    while (clock() < deadline) {
      const state = await snapshot();
      stats.polls++;
      stats.settling_ms = clock() - begin;
      stats.last_snapshot = state;
      check(clock() < deadline, 'Browser deadline reached while observing ' + phase + ' playback sample');
      initialCounts ??= {...state.seekEventCounts};
      const videos = ids.map(id => {
        const video = state.cards.find(card => card.id === id)?.video;
        check(video, 'Missing selected playback player: ' + id);
        check(!video.error, 'Browser media error while settling: ' + id + ' ' + JSON.stringify(video.error));
        if (video.seeking) stats.seeking_poll_observations[id] = (stats.seeking_poll_observations[id] || 0) + 1;
        return video;
      });
      telemetry.seek_event_deltas = Object.fromEntries(ids.map(id => [id,
        Math.max(0, (state.seekEventCounts?.[id] || 0) - (initialCounts[id] || 0))]));
      if (videos.every(video => !video.paused && video.ready >= 2 && !video.seeking)) return state;
      await sleep(Math.max(0, Math.min(75, deadline - clock())));
    }
    stats.settling_ms = clock() - begin;
    throw new Error('Browser deadline reached: ' + phase + ' playback sample never settled');
  };
  try {
    const before = await settled('before');
    const sampleStart = clock();
    check(sampleStart + 700 <= deadline, 'Browser deadline cannot fit the unchanged 700 ms sample');
    await sleep(700);
    const after = await settled('after');
    telemetry.actual_sample_ms = clock() - sampleStart;
    telemetry.total_settling_ms = telemetry.before.settling_ms + telemetry.after.settling_ms;
    const result = assertPlaybackProgress(before, after, ids);
    telemetry.status = 'passed';
    return {before, after, result: {...result, observation: telemetry}};
  } catch (error) {
    telemetry.status = 'failed'; telemetry.error = error.message;
    throw error;
  } finally {
    telemetry.elapsed_ms = clock() - started;
    telemetry.total_settling_ms = telemetry.before.settling_ms + telemetry.after.settling_ms;
  }
}

async function waitFor(predicate, deadline, description) {
  while (performance.now() < deadline) {
    const value = await predicate();
    if (value) return value;
    await delay(75);
  }
  throw new Error('Browser deadline reached: ' + description);
}

async function browserChecks(cdp, session, manifest, destination, deadline, report) {
  const evaluate = expression => cdp.evaluate(expression, session);
  report.snapshot_observer = {light_calls: 0, light_max_payload_bytes: 0, light_total_response_ms: 0,
    scope: 'Playback polls omit card text, hidden evidence, notes and growing event logs.'};
  const snapshot = async () => {
    const started = performance.now(), value = await evaluate(SNAPSHOT);
    report.snapshot_observer.light_calls++;
    report.snapshot_observer.light_total_response_ms += performance.now() - started;
    report.snapshot_observer.light_max_payload_bytes = Math.max(report.snapshot_observer.light_max_payload_bytes,
      Buffer.byteLength(JSON.stringify(value)));
    return value;
  };
  const action = expression => evaluate(expression);
  const ready = () => waitFor(async () => {
    const state = await snapshot();
    return state.cards.length === 16 && !state.status?.includes('unavailable') && state;
  }, deadline, 'gallery loaded');
  const setSong = song => action(`(() => {const x=document.getElementById('song');x.value=${JSON.stringify(song)};x.dispatchEvent(new Event('change',{bubbles:true}));return true})()`);
  const choose = ids => action(`(() => {const wanted=${JSON.stringify(ids)};for(const card of document.querySelectorAll('.card')){const x=card.querySelector('input[type=checkbox]');if(x&&x.checked&&!wanted.includes(card.id.slice(8))){x.checked=false;x.dispatchEvent(new Event('change',{bubbles:true}))}}for(const id of wanted){const x=document.querySelector('#variant-'+id+' input[type=checkbox]');if(x&&!x.checked){x.checked=true;x.dispatchEvent(new Event('change',{bubbles:true}))}}return true})()`);
  const click = id => action(`document.getElementById(${JSON.stringify(id)}).click()`);
  const seek = value => action(`(() => {const x=document.getElementById('seek');x.value=${value};x.dispatchEvent(new Event('input',{bubbles:true}));return true})()`);
  const selected = state => state.cards.filter(card => card.visible && card.video?.checked);
  let state = await ready();
  // Inspect the large hidden evidence exactly once, while all players are paused.
  const fullDOM = await evaluate(FULL_DOM_SNAPSHOT);
  validateDOMSnapshot(fullDOM, manifest);
  report.full_dom_validation = {count: 1, payload_bytes: Buffer.byteLength(JSON.stringify(fullDOM)),
    sha256: createHash('sha256').update(JSON.stringify(fullDOM)).digest('hex')};
  report.playback_observations = [];
  const observe = async (ids, label) => {
    const telemetry = {label, player_ids: ids};
    report.playback_observations.push(telemetry);
    const sample = await sampleSettledPlayback(snapshot, ids, deadline, telemetry);
    state = sample.after;
    return sample.result;
  };
  report.checks.cards_evidence_downloads = {cards: state.cards.length, real_players: state.cards.filter(c => c.video).length};
  const representatives = manifest.variants.filter(row => !row.duplicate_of);
  check(representatives.length >= 5, 'Cannot exercise max-four limit: fewer than five distinct visual representatives');
  await setSong('all');
  const five = representatives.slice(0, 5).map(row => row.id);
  await choose(five);
  state = await snapshot();
  check(selected(state).length === 4 && !state.cards.find(card => card.id === five[4]).video.checked,
    'Gallery accepted more than four selected players');
  report.checks.maximum_four = {attempted: five, selected: selected(state).map(c => c.id)};

  const firstSong = SONGS[0];
  const four = representatives.filter(row => row.track_id === firstSong).slice(0, 4).map(row => row.id);
  check(four.length === 4, 'Four distinct same-song real players required for playback and master-handoff check');
  await choose([]); await setSong(firstSong); await choose(four.slice(0, 2));
  await seek(12);
  await waitFor(async () => {
    const s = await snapshot();
    return selected(s).every(c => c.video.ready >= 2 && !c.video.seeking && Math.abs(c.video.time - 12) < .15);
  }, deadline, 'shared seek to 12 seconds');
  report.checks.shared_seek = {song: firstSong, requested_s: 12, players: (await snapshot()).cards.filter(c => c.video?.checked).map(c => ({id:c.id,time:c.video.time}))};
  await action(`(() => {const x=document.getElementById('audible');x.value=${JSON.stringify(four[0])};x.dispatchEvent(new Event('change',{bubbles:true}));return true})()`);
  state = await snapshot();
  check(state.cards.filter(card => card.video && !card.video.muted).map(card => card.id).join() === four[0], 'More than one audio source is unmuted');
  report.checks.single_audio_source = {audible_id: four[0], browser_process_audio_muted: true};
  await click('play');
  report.checks.two_video_playback = await observe(four.slice(0, 2), 'two_video_playback');
  await click('pause'); await choose(four); await seek(18);
  await waitFor(async () => selected(await snapshot()).every(c => c.video.ready >= 2 && !c.video.seeking && Math.abs(c.video.time - 18) < .15), deadline, 'four-player shared seek');
  await click('play');
  report.checks.four_video_playback = await observe(four, 'four_video_playback');
  const oldMaster = state.masterId;
  check(four.includes(oldMaster), 'Playing group has no selected master');
  const remaining = four.filter(id => id !== oldMaster);
  await choose(remaining);
  await waitFor(async () => {
    const s = await snapshot();
    return remaining.includes(s.masterId) && s.cards.find(c => c.id === oldMaster).video.paused
      && selected(s).every(c => !c.video.paused && !c.video.seeking);
  }, deadline, 'master deselection keeps remaining players playing');
  const handoff = await observe(remaining, 'deselected_master_handoff');
  report.checks.deselected_master_handoff = {old_master: oldMaster, new_master: state.masterId,
    ...handoff};
  await click('pause');
  state = await snapshot();
  check(state.cards.filter(c => c.video).every(c => c.video.paused), 'Pause all left a player running');
  report.checks.pause_all = true;

  await choose([]); await setSong(SONGS[1]);
  const second = representatives.filter(row => row.track_id === SONGS[1]).slice(0, 2).map(row => row.id);
  check(second.length === 2, 'Need two independent Cure For Me visual representatives');
  await choose(second); await seek(12);
  state = await snapshot();
  check(state.cards.filter(c => c.visible).every(c => manifest.variants.find(r => r.id === c.id).track_id === SONGS[1])
    && state.cards.filter(c => c.visible).length === 8, 'Song filter leaks or loses cards');
  check(state.cards.filter(c => c.video && !c.visible).every(c => c.video.paused), 'Hidden previous-song media still playing');
  await click('play');
  report.checks.second_song_playback = await observe(second, 'second_song_playback');
  await click('pause');
  report.checks.song_filter = {visible: state.cards.filter(c => c.visible).map(c => c.id), separate_movie_ids: second};

  const noteId = second[0], marker = 'Automated gallery verification only; no artistic acceptance. ' + report.revision;
  await action(`(() => {const x=document.querySelector('#variant-${noteId} textarea');x.value=${JSON.stringify(marker)};x.dispatchEvent(new Event('input',{bubbles:true}));return true})()`);
  report.before_reload_media_events = await evaluate(MEDIA_EVENTS);
  const previousNavigation = (await snapshot()).navigation;
  await cdp.send('Page.reload', {ignoreCache: true}, session);
  state = await waitFor(async () => {const value = await snapshot();return value.navigation !== previousNavigation
    && value.cards.length === 16 && value;}, deadline, 'fresh page reload with retained notes');
  const retainedNote = await evaluate(`document.querySelector('#variant-${noteId} textarea')?.value`);
  check(retainedNote === marker, 'Notes did not persist across page reload');
  await click('export');
  const notesPath = path.join(destination, 'downloads/dynamic_dance_review_notes.json');
  const exported = await waitFor(async () => {try {return await read(notesPath);} catch(error) {if(error.code==='ENOENT'||error instanceof SyntaxError)return false;throw error;}}, deadline, 'actual notes export download');
  check(exported.plan_sha256 === manifest.plan_sha256 && exported.campaign === manifest.campaign
    && exported.notes?.[noteId]?.text === marker && exported.notes[noteId].video_sha256 === manifest.variants.find(r => r.id === noteId).sha256,
  'Downloaded notes differ from pinned media observations');
  report.checks.notes_persistence_export = {id: noteId, path: notesPath, sha256: await digest(notesPath), isolated_profile_only: true};
  await setSong(firstSong);
  const screenshot = await cdp.send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false}, session);
  await floor(path.dirname(path.dirname(destination)));
  const imagePath = path.join(destination, 'gallery.png');
  await fs.writeFile(imagePath, Buffer.from(screenshot.data, 'base64'), {flag: 'wx'});
  report.screenshot = {path: imagePath, sha256: await digest(imagePath)};
  report.final_snapshot = await snapshot();
  report.final_media_events = await evaluate(MEDIA_EVENTS);
  check(report.final_snapshot.cards.length === 16 && !report.final_snapshot.cards.some(c => c.video?.error),
    'Final gallery card/media status changed');
  check(!report.final_media_events.some(e => e.type === 'error')
    && !report.before_reload_media_events.some(e => e.type === 'error'), 'Browser reported a media decode/playback error');
}

export async function run(args) {
  await floor(args.root);
  const destination = path.join(args.root, 'delivery_verification', args.revision);
  await fs.mkdir(path.dirname(destination), {recursive: true});
  await fs.mkdir(destination); // Exclusive; never overwrite evidence or another profile.
  const report = {schema_version: 1, status: 'running', revision: args.revision,
    started_at: new Date().toISOString(), verifier_sha256: await digest(HERE),
    url: args.url, plan_sha256: args.planSha256, browser_timeout_ms: args.timeoutMs,
    checks: {}, browser_errors: [], network_errors: [], console_errors: [], media_properties: [],
    limitations: ['Short playback samples verify gallery mechanics, not artistic quality or full-movie integrity.',
      'Headless browser audio is muted at process level; DOM exclusivity and real audio decode are separate checks.',
      'Prior Python delivery verification authenticates full software decoding and source PCM.',
      'No substitute media is used; unsupported software decoding is a reported failure.']};
  let child, profile, cdp, session, watchdog, browserStarted;
  try {
    const inputs = await authenticateInputs(args.root, args.planSha256);
    report.inputs = inputs.pins;
    await verifyServedBytes(args.url, path.join(args.root, 'review/index.html'), '/');
    await verifyServedBytes(args.url, path.join(args.root, 'review/MANIFEST.json'), '/manifest.json');
    await fs.mkdir(path.join(destination, 'downloads'));
    profile = await fs.mkdtemp(path.join(tmpdir(), 'lightsem-gallery-browser-'));
    const flags = chromeArguments(profile);
    report.chrome_command = [args.chrome, ...flags];
    browserStarted = performance.now();
    const deadline = browserStarted + args.timeoutMs;
    child = spawn(args.chrome, flags, {stdio: ['ignore', 'ignore', 'pipe'], detached: true});
    report.owned_browser_pid = child.pid;
    let launchError;
    child.on('error', error => {launchError = error;});
    report.chrome_stderr = '';
    child.stderr.on('data', chunk => {if (report.chrome_stderr.length < 100000) report.chrome_stderr += chunk.toString();});
    watchdog = setTimeout(() => {
      report.browser_deadline_exceeded = true;
      if (child?.pid) {try {process.kill(-child.pid, 'SIGTERM');} catch {}}
    }, args.timeoutMs);
    const active = await waitFor(async () => {
      if (launchError) throw launchError;
      if (child.exitCode !== null || child.signalCode !== null) throw new Error('Owned Chrome exited early: ' + report.chrome_stderr.slice(-2000));
      try {return (await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).trim().split('\n');}
      catch (error) {if (error.code === 'ENOENT') return false; throw error;}
    }, deadline, 'isolated Chrome debugging endpoint');
    const socket = new WebSocket('ws://127.0.0.1:' + active[0] + active[1]);
    await new Promise((resolve, reject) => {
      socket.addEventListener('open', resolve, {once: true});
      socket.addEventListener('error', () => reject(new Error('Owned Chrome WebSocket connection failed')), {once: true});
    });
    cdp = new CDP(socket, Math.min(5000, args.timeoutMs));
    cdp.onEvent = event => {
      if (event.method === 'Runtime.exceptionThrown') report.browser_errors.push(event.params);
      if (event.method === 'Log.entryAdded' && event.params.entry.level === 'error') report.console_errors.push(event.params.entry);
      if (event.method === 'Network.loadingFailed') report.network_errors.push(event.params);
      if (event.method === 'Media.playerPropertiesChanged') report.media_properties.push(event.params);
    };
    report.browser_version = await cdp.send('Browser.getVersion');
    await cdp.send('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: path.join(destination, 'downloads'), eventsEnabled: true});
    const {targetId} = await cdp.send('Target.createTarget', {url: 'about:blank'});
    ({sessionId: session} = await cdp.send('Target.attachToTarget', {targetId, flatten: true}));
    for (const domain of ['Page', 'Runtime', 'Network', 'Log', 'Media']) await cdp.send(domain + '.enable', {}, session);
    await cdp.send('Emulation.setDeviceMetricsOverride', {width: 1600, height: 1100, deviceScaleFactor: 1, mobile: false}, session);
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', {source: `window.__galleryCheckMediaEvents=[];window.__galleryCheckSeekingCounts={};for(const type of ['error','stalled','waiting','playing','pause','seeking','seeked'])document.addEventListener(type,event=>{if(!(event.target instanceof HTMLMediaElement))return;const card=event.target.closest('.card')?.id;if(type==='seeking'&&card){const id=card.slice(8);window.__galleryCheckSeekingCounts[id]=(window.__galleryCheckSeekingCounts[id]||0)+1}if(window.__galleryCheckMediaEvents.length<1000)window.__galleryCheckMediaEvents.push({type,time:performance.now(),card,media_time:event.target.currentTime,error:event.target.error?{code:event.target.error.code,message:event.target.error.message}:null})},true);`}, session);
    await cdp.send('Page.navigate', {url: args.url + '/'}, session);
    await browserChecks(cdp, session, inputs.manifest, destination, deadline, report);
    const decoders = report.media_properties.flatMap(event => event.properties || [])
      .filter(property => /IsPlatformVideoDecoder/i.test(property.name));
    check(decoders.every(property => !['true', '1'].includes(String(property.value).toLowerCase())),
      'Browser reported a hardware/platform video decoder despite software-only flags');
    report.software_decoder_properties = decoders;
    check(!report.browser_errors.length, 'Page JavaScript exception recorded');
    check(!report.network_errors.some(error => !error.canceled), 'Non-canceled browser network failure recorded');
    check(!report.browser_deadline_exceeded, 'Live-browser check exceeded its time budget');
    report.status = 'passed';
  } catch (error) {
    report.status = 'failed'; report.error = {name: error.name, message: error.message};
    if (cdp && session && !report.browser_deadline_exceeded) {
      try {
        report.failure_snapshot = await cdp.evaluate(SNAPSHOT, session);
        report.failure_media_events = await cdp.evaluate(MEDIA_EVENTS, session);
        const screenshot = await cdp.send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false}, session);
        const imagePath = path.join(destination, 'failure.png');
        await fs.writeFile(imagePath, Buffer.from(screenshot.data, 'base64'), {flag: 'wx'});
        report.failure_screenshot = {path: imagePath, sha256: await digest(imagePath)};
      } catch (captureError) {report.failure_capture_error = captureError.message;}
    }
  } finally {
    clearTimeout(watchdog);
    if (browserStarted) report.live_browser_elapsed_ms = Math.round(performance.now() - browserStarted);
    if (cdp) {try {await cdp.send('Browser.close');} catch {} cdp.socket.close();}
    if (child?.pid) {
      for (let i = 0; i < 20 && child.exitCode === null && child.signalCode === null; i++) await delay(50);
      if (child.exitCode === null && child.signalCode === null) {try {process.kill(-child.pid, 'SIGTERM');} catch {}}
      for (let i = 0; i < 10 && child.exitCode === null && child.signalCode === null; i++) await delay(50);
      if (child.exitCode === null && child.signalCode === null) {try {process.kill(-child.pid, 'SIGKILL');} catch {}}
    }
    if (profile) {try {await fs.rm(profile, {recursive: true});} catch (error) {report.profile_cleanup_error = error.message;}}
  }
  if (report.inputs) {
    try {for (const [filename, hash] of Object.entries(report.inputs)) await pinned(filename, hash);}
    catch (error) {report.status = 'failed'; report.input_drift_error = error.message;}
  }
  await floor(args.root);
  report.finished_at = new Date().toISOString();
  const reportPath = path.join(destination, 'REPORT.json');
  await writeNew(reportPath, report);
  await writeNew(reportPath + '.sha256.json', {sha256: await digest(reportPath)});
  console.log('GALLERY_BROWSER_' + report.status.toUpperCase(), reportPath);
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === HERE) {
  try {const report = await run(parseArguments(process.argv.slice(2))); if (report.status !== 'passed') process.exitCode = 1;}
  catch (error) {console.error(error.message); process.exitCode = 1;}
}
