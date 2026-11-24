// BanglaLPR V3 front end (no framework). Talks only to the local server.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const norm = s => String(s || '').normalize('NFC').replace(/[0-9]/g, d => '০১২৩৪৫৬৭৮৯'[d]).replace(/[\s\-–—.:,_]/g, '');
const icon = (n, cls = '') => `<svg class="${cls}"><use href="#i-${n}"/></svg>`;
const STATUS = { verified: 'Verified', likely: 'Likely', review: 'Check manually' };
const STATUS_TIP = {
  verified: 'The readers agree and the plate has a valid BRTA layout',
  likely: 'Valid layout, but the readers did not fully agree',
  review: 'The text may be wrong: look at the plate picture',
};
let INFO = null;

// ---------------------------------------------------------------- small helpers
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = 'show' + (err ? ' err' : '');
  clearTimeout(t._h); t._h = setTimeout(() => (t.className = ''), err ? 7000 : 3200);
}
async function api(url, opt = {}) {
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
const media = rel => '/records/' + String(rel).split('/').map(encodeURIComponent).join('/');
const chip = st => `<span class="chip ${esc(st)}" title="${esc(STATUS_TIP[st] || '')}">${esc(STATUS[st] || st || '—')}</span>`;
const when = ts => {
  const d = new Date(ts * 1000), today = new Date();
  const t = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return d.toDateString() === today.toDateString() ? `Today ${t}` : `${d.toLocaleDateString()} ${t}`;
};
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied: ' + text); }
  catch { toast('Copy is not allowed here', true); }
}
function lightbox(src, caption = '') {
  const lb = $('#lightbox'); $('img', lb).src = src; $('p', lb).textContent = caption; lb.classList.remove('hidden');
}
$('#lightbox').addEventListener('click', () => $('#lightbox').classList.add('hidden'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') $('#lightbox').classList.add('hidden'); });
document.addEventListener('click', e => {
  const z = e.target.closest('[data-zoom]'); if (z) { lightbox(z.dataset.zoom, z.dataset.cap || ''); return; }
  const c = e.target.closest('[data-copy]'); if (c) copy(c.dataset.copy);
});

// ---------------------------------------------------------------- theme
(function theme() {
  let t = null;
  try { t = localStorage.getItem('lpr-theme'); } catch { /* storage blocked */ }
  if (t) document.documentElement.dataset.theme = t;
  $('#theme').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('lpr-theme', document.documentElement.dataset.theme); } catch { /* ignore */ }
  });
})();

// ---------------------------------------------------------------- night mode
// Auto = on from 18:00 to 06:00 by this device's clock. The photo itself cannot tell reliably: night photos lit by a
// flash or headlights are often not dark (results/NIGHT_MODE_RESULTS.md), so the page only SUGGESTS night mode for a
// dark photo and never switches by itself.
let nightPref = 'auto';
try { nightPref = localStorage.getItem('lpr-night') || 'auto'; } catch { /* storage blocked */ }
const clockNight = () => { const h = new Date().getHours(); return h >= 18 || h < 6; };
const nightOn = () => nightPref === 'on' || (nightPref === 'auto' && clockNight());
const nq = (url, force) => url + (url.includes('?') ? '&' : '?') + 'night=' + ((force ?? nightOn()) ? 1 : 0);
function showNight() {
  $$('#night-seg button').forEach(b => b.classList.toggle('on', b.dataset.n === nightPref));
  const s = $('#night-state'), on = nightOn();
  s.textContent = nightPref === 'auto' ? (on ? 'on now (evening)' : 'off now (daytime)') : (on ? 'on' : 'off');
  s.className = 'night-state' + (on ? ' on' : '');
  $('#cam-torch')?.classList.toggle('hidden', !(on && camTorchOK));
}
$('#night-seg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  nightPref = b.dataset.n;
  try { localStorage.setItem('lpr-night', nightPref); } catch { /* ignore */ }
  showNight();
  toast(nightOn() ? 'Night mode on: plates are read with the night settings' : 'Night mode off');
});
setInterval(showNight, 60000);

// ---------------------------------------------------------------- tabs
function showTab(t) {
  $$('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  $$('.tab').forEach(s => s.classList.toggle('on', s.id === 'tab-' + t));
  if (t === 'records') loadRecords();
  if (t === 'watch') loadWatch();
  if (t === 'about') renderAbout();
  try { history.replaceState(null, '', '#' + t); } catch { /* ignore */ }
}
$('#tabs').addEventListener('click', e => { const b = e.target.closest('button'); if (b) showTab(b.dataset.tab); });

// ---------------------------------------------------------------- engine status
async function pollEngine() {
  try {
    INFO = await api('/api/info');
    const el = $('#engine');
    el.className = 'engine ' + (INFO.engine === 'ready' ? 'ready' : 'loading');
    $('span', el).textContent = INFO.engine === 'ready' ? 'Model ready' : 'Loading model…';
    el.title = INFO.engine === 'ready' ? `Model loaded in ${INFO.load_s} s` : 'The model loads once (about 10-20 s)';
    if (INFO.engine !== 'ready') setTimeout(pollEngine, 1500);
    return INFO;
  } catch (e) {
    const el = $('#engine'); el.className = 'engine error'; $('span', el).textContent = 'Server offline';
    setTimeout(pollEngine, 4000);
  }
}

// ---------------------------------------------------------------- plate cards
function plateCard(p, i, opts = {}) {
  const readers = Object.entries(p.readers || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="bn">${esc(v) || '—'}</td></tr>`).join('');
  const reads = p.all_reads ? p.all_reads.map((t, k) => `<tr><td>read ${k + 1}</td><td class="bn">${esc(t) || '—'}</td></tr>`).join('') : '';
  const img = p.crop ? `<img class="crop" src="${media(p.crop)}" alt="plate picture" data-zoom="${media(p.crop)}" data-cap="${esc(p.text)}">` : '';
  const extra = (opts.chips || []).map(c => `<span class="chip">${esc(c)}</span>`).join('');
  return `<div class="pcard ${p.watch_hit ? 'hit' : ''}">
    ${img}
    <div class="plate-text"><div>
        <div class="ptext">${esc(p.text) || '<span class="muted">(could not read)</span>'}</div>
        <div class="pen">${esc(p.text_en)}</div></div>
      ${p.text ? `<button class="btn small" data-copy="${esc(p.text)}" title="Copy the Bangla text">${icon('copy')}Copy</button>` : ''}</div>
    <div class="chips">${chip(p.status)}${p.watch_hit ? `<span class="chip hit">${icon('bell')}Watchlist match</span>` : ''}
      ${p.valid ? `<span class="chip">${icon('check')}BRTA layout</span>` : '<span class="chip review">layout not valid</span>'}
      ${p.det_conf != null ? `<span class="chip" title="How sure the detector is that this is a plate">plate found ${(100 * p.det_conf).toFixed(0)} %</span>` : ''}
      ${p.upscaled ? '<span class="chip" title="Small plate, enlarged x2 before reading">small plate</span>' : ''}${extra}</div>
    ${p.id ? `<div class="saved">${icon('check')}<span>Saved as record #${p.id}${p.crop ? `<small>${esc(p.crop.split('/').pop())}</small>` : ''}</span></div>` : ''}
    ${readers ? `<details class="more"><summary>What each reader saw</summary><table class="rtable">${readers}</table></details>` : ''}
    ${reads ? `<details class="more"><summary>All readings of this vehicle (${p.all_reads.length})</summary><table class="rtable">${reads}</table></details>` : ''}
  </div>`;
}
const nightChip = on => on ? `<span class="chip night" title="Read with the night settings">${icon('night')}Night mode</span>` : '';
const REREAD = {};
function darkHint(r, key) {
  if (r.night_mode || !r.looks_dark || !key) return '';
  return `<div class="dark-hint">${icon('night')}<span>This photo looks dark. Night mode may read it better.</span>
    <button class="btn small" data-reread="${key}">Read again in night mode</button></div>`;
}
document.addEventListener('click', e => {
  const b = e.target.closest('[data-reread]'); if (!b) return;
  const job = REREAD[b.dataset.reread]; if (job) job();
});
function resultBlock(r, title, truth, key) {
  const best = r.plates.find(p => p.valid) || r.plates[0];
  let tr = '';
  if (truth) {
    const ok = best && norm(best.text) === norm(truth);
    tr = `<div class="truth">True plate: <b>${esc(truth)}</b> → ${ok ? '<span class="ok">read correctly ✓</span>' : '<span class="bad">not exact</span>'}</div>`;
  }
  const tot = r.ms?.total;
  const photo = r.annotated ? media(r.annotated) : null;
  return `<div class="result">
    <div class="result-head"><span class="fname">${esc(title)}</span>
      <span class="chip">${r.plates.length} plate${r.plates.length === 1 ? '' : 's'}</span>
      ${r.retry_used ? '<span class="chip likely" title="The first detector found nothing; a second try found the plate">second try</span>' : ''}
      ${nightChip(r.night_mode)}
      ${tot ? `<span class="timing">${tot} ms</span>` : ''}</div>
    ${darkHint(r, key)}
    <div class="result-body">
      <div class="photo" ${photo ? `data-zoom="${photo}"` : ''}>${photo ? `<img src="${photo}" alt="photo with plate boxes">` : ''}</div>
      <div class="plates">${tr}${r.plates.length ? r.plates.map((p, i) => plateCard(p, i)).join('')
        : '<div class="noplate"><b>No plate found</b>Try a photo where the plate is larger and in focus.</div>'}</div>
    </div></div>`;
}
function placeholder(text) {
  const out = $('#photo-out');
  const h = document.createElement('div');
  h.innerHTML = `<div class="result loading-card"><div class="spinner"></div><div>${esc(text)}${INFO?.engine !== 'ready' ? '<br><small>The model is still loading the first time (about 10-20 s).</small>' : ''}</div></div>`;
  out.prepend(h);
  $('#drop').classList.add('compact');
  return h;
}
function updateCount(n) { $('#rec-count').textContent = n ? String(n) : ''; }
async function refreshCount() { try { updateCount((await api('/api/stats')).plates); } catch { /* ignore */ } }

// ---------------------------------------------------------------- photo tab
async function sendPhoto(file, name, night) {
  name = name || file.name || 'pasted photo';
  const h = placeholder(`Reading ${name}…${(night ?? nightOn()) ? ' (night mode)' : ''}`);
  try {
    const r = await api(nq('/api/recognize', night), { method: 'POST', body: file, headers: { 'X-Filename': encodeURIComponent(name) } });
    const key = 'p' + Math.random().toString(36).slice(2);
    REREAD[key] = () => sendPhoto(file, name, true);
    h.innerHTML = resultBlock(r, name, null, key);
    if (r.plates.length) toast(`${r.plates.length} plate${r.plates.length > 1 ? 's' : ''} saved to records`);
    refreshCount();
  } catch (e) { h.innerHTML = `<div class="result error-card">Could not read ${esc(name)}: ${esc(e.message)}</div>`; toast(e.message, true); }
}
async function sendSample(s, night) {
  const h = placeholder(`Reading test photo ${s.file}…${(night ?? nightOn()) ? ' (night mode)' : ''}`);
  try {
    const r = await api(nq(`/api/sample?file=${encodeURIComponent(s.file)}`, night), { method: 'POST' });
    const key = 's' + Math.random().toString(36).slice(2);
    REREAD[key] = () => sendSample(s, true);
    h.innerHTML = resultBlock(r, `Test photo ${s.file}`, r.truth, key);
    refreshCount();
  } catch (e) { h.innerHTML = `<div class="result error-card">${esc(e.message)}</div>`; toast(e.message, true); }
}
['#file-photo', '#file-snap'].forEach(id => $(id).addEventListener('change', e => {
  [...e.target.files].forEach(f => sendPhoto(f)); e.target.value = '';
}));
const drop = $('#drop');
['dragenter', 'dragover'].forEach(ev => document.addEventListener(ev, e => {
  if (!$('#tab-photo').classList.contains('on')) return;
  e.preventDefault(); drop.classList.add('over');
}));
['dragleave', 'drop'].forEach(ev => document.addEventListener(ev, e => {
  if (ev === 'dragleave' && e.relatedTarget) return;
  e.preventDefault(); drop.classList.remove('over');
}));
document.addEventListener('drop', e => {
  if (!$('#tab-photo').classList.contains('on')) return;
  const files = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith('image/'));
  if (!files.length) return toast('Drop a photo file (JPG, PNG or WebP)', true);
  files.forEach(f => sendPhoto(f));
});
document.addEventListener('paste', e => {
  const items = [...(e.clipboardData?.items || [])].filter(i => i.type.startsWith('image/'));
  if (!items.length) return;
  showTab('photo');
  items.forEach((it, k) => sendPhoto(it.getAsFile(), `pasted photo ${new Date().toLocaleTimeString()}${k ? ' ' + (k + 1) : ''}`));
});

// ---------------------------------------------------------------- video tab
$('#file-video').addEventListener('change', e => {
  const f = e.target.files[0]; e.target.value = '';
  if (f) runVideoJob(() => api(nq('/api/video'), { method: 'POST', body: f, headers: { 'X-Filename': encodeURIComponent(f.name) } }), f.name);
});
$('#demo-video').addEventListener('click', () =>
  runVideoJob(() => api(nq('/api/video_demo'), { method: 'POST' }), 'Demo clip (made from test photos)'));
async function runVideoJob(start, name) {
  const prog = $('#video-prog'), out = $('#video-out');
  prog.classList.remove('hidden'); $('i', prog).style.width = '0%'; $('span', prog).textContent = 'Uploading…';
  out.innerHTML = '';
  try {
    const { job } = await start();
    for (;;) {
      await new Promise(r => setTimeout(r, 800));
      const j = await api('/api/job/' + job);
      const pct = j.total ? Math.min(100, 100 * j.frame / j.total) : 0;
      $('i', prog).style.width = pct + '%';
      $('span', prog).textContent = j.total ? `Frame ${j.frame} of ${j.total} (${pct.toFixed(0)} %)` : 'Loading the model…';
      if (j.state === 'error') throw new Error(j.error);
      if (j.state === 'done') { showVideo(j.result, name); break; }
    }
    refreshCount();
  } catch (err) { toast(err.message, true); out.innerHTML = `<div class="result error-card">${esc(err.message)}</div>`; }
  prog.classList.add('hidden');
}
function showVideo(r, name) {
  const ev = r.events.map((p, i) => plateCard(p, i, { chips: [`vehicle #${p.track_id}`, `${p.agree}/${p.reads} reads agree`, `${p.first_s}–${p.last_s} s`] })).join('');
  $('#video-out').innerHTML = `<div class="result"><div class="result-head"><span class="fname">${esc(name)}</span>
     <span class="chip">${r.events.length} vehicle${r.events.length === 1 ? '' : 's'} saved</span>${nightChip(r.night_mode)}
     <span class="timing">${r.frames} frames · ${r.processing_fps} FPS</span></div>
     <div class="vgrid"><div><video src="${media(r.video)}" controls></video>
       <p class="empty-s">If the video does not play here, <a href="${media(r.video)}" download>download it</a> and open it in VLC.</p></div>
     <div class="events">${ev || '<div class="noplate"><b>No plates read</b></div>'}</div></div></div>`;
}

// ---------------------------------------------------------------- camera tab
let camTimer = null, camBusy = false, camTorchOK = false, camTorch = false;
async function setTorch(on) {
  const t = $('#cam').srcObject?.getVideoTracks()[0];
  if (!t) return;
  try { await t.applyConstraints({ advanced: [{ torch: on }] }); camTorch = on; $('#cam-torch').classList.toggle('on', on); }
  catch (e) { toast('This camera has no light the browser can switch', true); }
}
$('#cam-torch').addEventListener('click', () => setTorch(!camTorch));
$('#cam-start').addEventListener('click', async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('The live camera needs https on a phone. Start the server with --phone and open the https address it prints.', true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: 'environment' } } });
    const v = $('#cam'); v.srcObject = stream; await v.play();
    $('#cam-start').disabled = true; $('#cam-stop').disabled = false; $('#cam-off').classList.add('hidden');
    const tr = stream.getVideoTracks()[0];
    camTorchOK = !!(tr?.getCapabilities && tr.getCapabilities().torch);       // Android Chrome: phone torch
    showNight();
    if (camTorchOK && nightOn()) setTorch(true);
    camTimer = setInterval(camTick, 350);
  } catch (e) { toast('Camera: ' + e.message, true); }
});
$('#cam-stop').addEventListener('click', stopCam);
function stopCam() {
  clearInterval(camTimer); camTimer = null;
  const v = $('#cam'); (v.srcObject?.getTracks() || []).forEach(t => t.stop()); v.srcObject = null;
  camTorchOK = camTorch = false; $('#cam-torch').classList.remove('on'); showNight();
  $('#cam-start').disabled = false; $('#cam-stop').disabled = true; $('#cam-stat').textContent = '';
  $('#cam-off').classList.remove('hidden');
  const c = $('#cam-overlay'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
}
async function camTick() {
  if (camBusy) return;
  const v = $('#cam'); if (!v.videoWidth) return;
  camBusy = true;
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.85));
  try {
    const t0 = performance.now();
    const r = await api(nq('/api/frame'), { method: 'POST', body: blob });
    const ov = $('#cam-overlay'); ov.width = v.videoWidth; ov.height = v.videoHeight;
    const g = ov.getContext('2d'); g.lineWidth = Math.max(3, ov.width / 300); g.font = `600 ${Math.max(18, ov.width / 45)}px Inter, sans-serif`;
    const col = { verified: '#2fbf71', likely: '#ffb020', review: '#ff4d5e' };
    r.plates.forEach(p => {
      const [x1, y1, x2, y2] = p.box; g.strokeStyle = g.fillStyle = col[p.status] || '#fff';
      g.strokeRect(x1, y1, x2 - x1, y2 - y1); g.fillText(p.text_en || '?', x1, Math.max(20, y1 - 8));
    });
    $('#cam-stat').textContent = `${r.ms.total} ms model · ${(performance.now() - t0).toFixed(0)} ms round trip · ${r.plates.length} plate(s)`
      + (r.night_mode ? ' · night mode' : r.looks_dark ? ' · looks dark: turn night mode on' : '');
    if (r.stored.length) {
      const log = $('#cam-events'); if ($('.empty-s', log)) log.innerHTML = '';
      r.stored.forEach(p => log.insertAdjacentHTML('afterbegin', `<div class="ev">${p.crop ? `<img src="${media(p.crop)}" alt="">` : ''}
        <div><div class="bn">${esc(p.text)}</div><div class="pen">${esc(p.text_en)}</div>
        <div class="chips">${chip(p.status)}${p.watch_hit ? '<span class="chip hit">Watchlist</span>' : ''}<small class="muted">${new Date().toLocaleTimeString()}</small></div></div></div>`));
      if (r.stored.some(p => p.watch_hit)) toast('Watchlist match: ' + r.stored.find(p => p.watch_hit).text, true);
      refreshCount();
    }
  } catch (e) { $('#cam-stat').textContent = 'error: ' + e.message; }
  camBusy = false;
}

// ---------------------------------------------------------------- records tab
let recView = 'cards', recLimit = 120;
$('#r-view').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#r-view button').forEach(x => x.classList.toggle('on', x === b));
  recView = b.dataset.v;
  try { localStorage.setItem('lpr-view', recView); } catch { /* ignore */ }
  loadRecords();
});
try {
  const v = localStorage.getItem('lpr-view');
  if (v) { recView = v; $$('#r-view button').forEach(x => x.classList.toggle('on', x.dataset.v === v)); }
} catch { /* ignore */ }
function sinceValue() {
  const v = $('#r-since').value;
  if (!v) return null;
  if (v === 'today') { const d = new Date(); d.setHours(0, 0, 0, 0); return d.getTime() / 1000; }
  return Date.now() / 1000 - Number(v) * 86400;
}
function filters() { return { q: $('#r-q').value.trim(), status: $('#r-status').value, since: sinceValue() }; }
async function loadRecords() {
  try {
    const s = await api('/api/stats');
    updateCount(s.plates);
    const bs = s.by_status || {};
    $('#stats').innerHTML = [
      [s.plates, 'plates saved', ''], [s.today, 'saved today', ''], [bs.verified || 0, 'verified', 'good'],
      [(bs.review || 0), 'need a manual check', (bs.review ? 'warn' : '')],
      [s.watch_hits, 'watchlist matches', s.watch_hits ? 'bad' : ''],
      [(s.folder_bytes / 1048576).toFixed(1) + ' MB', 'records folder size', ''],
    ].map(([b, t, c]) => `<div class="stat ${c}"><b>${esc(b)}</b><span>${t}</span></div>`).join('');
    $('#csv-path').textContent = s.csv;
    $('#img-path').textContent = s.folder + (s.folder.includes('\\') ? '\\' : '/') + 'plate_images';
    const f = filters();
    const q = new URLSearchParams({ q: f.q, status: f.status, limit: recLimit });
    if (f.since) q.set('since', f.since);
    const rows = await api('/api/history?' + q);
    renderRecords(rows, s.plates);
  } catch (e) { toast(e.message, true); }
}
function renderRecords(rows, total) {
  const el = $('#r-list');
  if (!rows.length) {
    const filtered = $('#r-q').value || $('#r-status').value || $('#r-since').value;
    el.innerHTML = `<div class="empty"><b>${filtered ? 'No records match these filters' : 'No records yet'}</b>
      ${filtered ? 'Clear the search or pick another status or date.' : 'Read a photo, a video or use the live camera: every plate found is saved here.'}</div>`;
    return;
  }
  const src = r => `${esc(r.kind || '')}${r.track_id != null ? ' · vehicle #' + r.track_id : ''}`;
  if (recView === 'table') {
    el.innerHTML = `<div class="table-wrap"><table><thead><tr><th>No</th><th>Plate picture</th><th>Plate (Bangla)</th><th>English</th>
      <th>Status</th><th>Saved</th><th>Source</th><th></th></tr></thead><tbody>${rows.map(r => `<tr class="${r.watch_hit ? 'hit' : ''}">
      <td class="mono">${r.id}</td>
      <td>${r.crop ? `<img src="${media(r.crop)}" alt="" loading="lazy" data-zoom="${media(r.crop)}" data-cap="${esc(r.text)}">` : '<small>no picture</small>'}</td>
      <td class="bn">${esc(r.text) || '—'}</td><td class="mono">${esc(r.text_en)}</td>
      <td>${chip(r.status)}${r.watch_hit ? ' <span class="chip hit">Watchlist</span>' : ''}${r.night ? ' ' + nightChip(1) : ''}</td>
      <td>${when(r.created)}</td><td>${src(r)}<br><small>${esc(r.filename || '')}</small></td>
      <td><button class="no-btn" data-del="${r.id}" title="Delete this record">${icon('trash')}</button></td></tr>`).join('')}</tbody></table></div>`;
  } else {
    el.innerHTML = `<div class="rgrid">${rows.map(r => `<div class="rcard ${r.watch_hit ? 'hit' : ''}">
      <div class="img" ${r.crop ? `data-zoom="${media(r.crop)}" data-cap="${esc(r.text)}"` : ''}>${r.crop ? `<img src="${media(r.crop)}" alt="" loading="lazy">` : '<span class="none">no picture</span>'}</div>
      <div class="body"><div class="bn">${esc(r.text) || '—'}</div><div class="pen">${esc(r.text_en)}</div>
        <div class="chips">${chip(r.status)}${r.watch_hit ? '<span class="chip hit">Watchlist</span>' : ''}${nightChip(r.night)}</div>
        <div class="meta"><span>#${r.id} · ${when(r.created)} · ${src(r)}</span>
          <button class="no-btn del" data-del="${r.id}" title="Delete this record">${icon('trash')}</button></div></div></div>`).join('')}</div>`;
  }
  if (rows.length >= recLimit && total > rows.length) {
    el.insertAdjacentHTML('beforeend', `<div class="more-btn"><button class="btn" id="r-more">Show more</button></div>`);
    $('#r-more').addEventListener('click', () => { recLimit += 120; loadRecords(); });
  }
}
$('#r-list').addEventListener('click', async e => {
  const b = e.target.closest('[data-del]'); if (!b) return;
  e.stopPropagation();
  if (!confirm(`Delete record #${b.dataset.del} and its plate picture?`)) return;
  try { await api('/api/plate/' + b.dataset.del, { method: 'DELETE' }); toast('Record deleted'); loadRecords(); }
  catch (err) { toast(err.message, true); }
});
$('#r-refresh').addEventListener('click', loadRecords);
$('#r-q').addEventListener('input', () => { clearTimeout(loadRecords._t); loadRecords._t = setTimeout(loadRecords, 250); });
$('#r-status').addEventListener('change', loadRecords);
$('#r-since').addEventListener('change', loadRecords);
$('#r-export').addEventListener('click', async () => {
  const b = $('#r-export'); b.disabled = true;
  const old = b.innerHTML; b.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px"></span>Exporting…';
  try {
    const r = await api('/api/export', { method: 'POST', body: JSON.stringify(filters()) });
    const box = $('#export-done');
    box.innerHTML = `${icon('check')}<span class="grow"><b>${r.plates} plates exported</b> with ${r.images} pictures to <code>${esc(r.folder)}</code></span>
      <a class="btn small primary" href="${r.download}">${icon('download')}Download .zip</a>
      ${INFO?.local ? `<button class="btn small" id="open-export" data-name="${esc(r.name)}">${icon('folder')}Open folder</button>` : ''}`;
    box.classList.remove('hidden');
    $('#open-export')?.addEventListener('click', ev => openFolder('export', ev.currentTarget.dataset.name));
    toast('Export ready');
  } catch (e) { toast(e.message, true); }
  b.disabled = false; b.innerHTML = old;
});
async function openFolder(what, name) {
  try { await api('/api/open', { method: 'POST', body: JSON.stringify({ what, name }) }); toast('Opened in File Explorer'); }
  catch (e) { toast(e.message, true); }
}
$('#r-open').addEventListener('click', () => openFolder('records'));
$('#r-clear').addEventListener('click', async () => {
  if (!confirm('Delete ALL saved records, plate pictures and photos from this PC? Exports already made are kept.')) return;
  try { await api('/api/clear', { method: 'POST' }); toast('All records deleted'); loadRecords(); } catch (e) { toast(e.message, true); }
});
$('#r-ret').addEventListener('click', async () => {
  const d = prompt('Delete records older than how many days?', '30');
  if (d === null) return;
  if (!(Number(d) >= 0)) return toast('Type a number of days', true);
  try {
    const j = await api('/api/retention', { method: 'POST', body: JSON.stringify({ days: Number(d) }) });
    toast(`${j.deleted_plates} records older than ${d} days deleted`); loadRecords();
  } catch (e) { toast(e.message, true); }
});

// ---------------------------------------------------------------- watchlist
async function loadWatch() {
  try {
    const w = await api('/api/watchlist');
    $('#w-list').innerHTML = w.map(x => `<div class="witem">${icon('bell')}<span class="bn">${esc(x.text)}</span>
      <span class="grow">${esc(x.note) || 'no note'} · added ${new Date(x.created * 1000).toLocaleDateString()}</span>
      <button class="btn small danger" data-id="${x.id}">Remove</button></div>`).join('')
      || '<div class="empty"><b>The watchlist is empty</b>Add a plate above to be alerted when it is read.</div>';
  } catch (e) { toast(e.message, true); }
}
$('#w-list').addEventListener('click', async e => {
  const b = e.target.closest('[data-id]'); if (!b) return;
  await api('/api/watchlist/' + b.dataset.id, { method: 'DELETE' }); loadWatch();
});
async function addWatch() {
  try {
    await api('/api/watchlist', { method: 'POST', body: JSON.stringify({ text: $('#w-text').value, note: $('#w-note').value }) });
    $('#w-text').value = $('#w-note').value = ''; loadWatch(); toast('Added to the watchlist');
  } catch (e) { toast(e.message, true); }
}
$('#w-add').addEventListener('click', addWatch);
['#w-text', '#w-note'].forEach(id => $(id).addEventListener('keydown', e => { if (e.key === 'Enter') addWatch(); }));

// ---------------------------------------------------------------- about
const pc = x => (x == null ? '—' : (+x).toFixed(2) + ' %');
async function renderAbout() {
  if (!INFO) await pollEngine();
  const m = INFO?.model || {};
  const e = m.e2e || {}, c = m.crops || {}, cpu = m.cpu || {};
  $('#model-card').innerHTML = `<h2>${esc(m.name || 'BanglaLPR V3')}</h2>
    <p class="muted">${esc(m.changes || '')}</p>
    <div class="kpi">
      <div><b>${pc(e['plate_exact_%'])}</b><span>whole plate exactly right (${e.plates ?? '—'} test photos)</span></div>
      <div><b>${pc(e['char_acc_%'])}</b><span>characters right (same photos)</span></div>
      <div><b>${pc(c['plate_exact_%'])}</b><span>plate exactly right (${c.plates ?? '—'} test plate crops)</span></div>
      <div><b>${e.fps_mean ?? '—'} / ${cpu.fps_mean ?? '—'}</b><span>photos per second, GPU / CPU only</span></div>
    </div>
    <p class="note">These numbers were measured on ${esc(m.measured_on || '—')} with the ${esc((m.measured_version || '').toUpperCase())}
      engine. V3 uses the same engine and the same weights (no new training), so the accuracy is the same; V3 only adds
      saving, export and the new screens.</p>`;
}

// ---------------------------------------------------------------- start
(async function init() {
  showNight();
  const info = await pollEngine();
  if (info?.samples?.length) {
    $('#samples').innerHTML = info.samples.map((s, i) => `<button class="sample" data-i="${i}">
      <img src="/sample/${encodeURIComponent(s.file)}" loading="lazy" alt=""><span>${esc(s.text)}</span></button>`).join('');
    $('#samples').addEventListener('click', e => {
      const el = e.target.closest('.sample'); if (el) sendSample(info.samples[+el.dataset.i]);
    });
  } else $('#samples-box').classList.add('hidden');
  if (info && !info.local) $$('.local-only').forEach(x => x.classList.add('hidden'));
  if (info && !info.demo_clip) $('#demo-video').classList.add('hidden');
  refreshCount();
  const t = location.hash.slice(1);
  if (t && $('#tab-' + t)) showTab(t);
})();
