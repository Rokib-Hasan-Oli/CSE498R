// BanglaLPR showcase front end (no framework). Talks only to the local server.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const norm = s => String(s || '').normalize('NFC').replace(/[0-9]/g, d => '০১২৩৪৫৬৭৮৯'[d]).replace(/[\s\-–—.:,_]/g, '');
let INFO = null;

function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = 'show' + (err ? ' err' : '');
  clearTimeout(t._h); t._h = setTimeout(() => (t.className = ''), err ? 8000 : 3500);
}
async function api(url, opt = {}) {
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function seg(id) {
  const el = $(id);
  el.addEventListener('click', e => {
    if (e.target.tagName !== 'BUTTON') return;
    $$('button', el).forEach(b => b.classList.toggle('on', b === e.target));
  });
  return () => $('button.on', el).dataset.v;
}
const verPhoto = seg('#ver-photo'), verVideo = seg('#ver-video'), verCam = seg('#ver-cam');

// ---------------------------------------------------------------- tabs
$('#tabs').addEventListener('click', e => {
  if (e.target.tagName !== 'BUTTON') return;
  const t = e.target.dataset.tab;
  $$('#tabs button').forEach(b => b.classList.toggle('on', b === e.target));
  $$('.tab').forEach(s => s.classList.toggle('on', s.id === 'tab-' + t));
  if (t === 'history') loadHistory();
  if (t === 'watch') loadWatch();
  if (t === 'report') renderReport();
});

// ---------------------------------------------------------------- plate cards
function badge(st) { return `<span class="badge ${st}">${st}</span>`; }
function plateCard(p, i) {
  const readers = Object.entries(p.readers || {}).map(([k, v]) =>
    `<tr><td>${esc(k)}</td><td class="bn">${esc(v) || '—'}</td></tr>`).join('');
  const extra = [];
  if (p.status_reader !== undefined && p.status_reader !== null) extra.push(`<tr><td>status reader</td><td class="bn">${esc(p.status_reader) || '—'}</td></tr>`);
  if (p.char_agreement !== undefined && p.char_agreement !== null) extra.push(`<tr><td>char CNN agreement</td><td>${(100 * p.char_agreement).toFixed(0)} %</td></tr>`);
  return `<div class="pcard ${p.watch_hit ? 'hit' : ''}">
    ${p.crop ? `<img src="/media/${esc(p.crop)}" alt="plate crop">` : '<div></div>'}
    <div>
      <div class="ptext">${esc(p.text) || '<span style="color:#999">(unreadable)</span>'}</div>
      <div class="pen">${esc(p.text_en)}</div>
      <div class="badges">${badge(p.status)}${p.watch_hit ? '<span class="badge hit">WATCHLIST MATCH</span>' : ''}
        ${p.track_id == null ? `<span class="badge">plate ${i + 1}</span>` : ''}${p.det_conf != null ? `<span class="badge">det ${p.det_conf.toFixed(2)}</span>` : ''}
        ${p.valid ? '<span class="badge">BRTA layout ✓</span>' : '<span class="badge review">layout ✗</span>'}
        ${p.upscaled ? '<span class="badge">tiny plate ×2</span>' : ''}</div>
      ${readers ? `<details><summary>What each reader saw (${Object.keys(p.readers).length} ran)</summary>
        <table class="rtable">${readers}${extra.join('')}</table></details>` : ''}
      ${p.all_reads ? `<details><summary>Readings of this track (${p.all_reads.length})</summary>
        <table class="rtable">${p.all_reads.map((t, k) => `<tr><td>read ${k + 1}</td><td class="bn">${esc(t) || '—'}</td></tr>`).join('')}</table></details>` : ''}
      <div class="msg" title="the message an edge camera would send instead of the photo">${esc(p.event_message)}</div>
    </div></div>`;
}
function resultBlock(r, title, truth) {
  const tot = r.ms ? r.ms.total : null;
  const best = r.plates.find(p => p.valid) || r.plates[0];
  let tr = '';
  if (truth) {
    const ok = best && norm(best.text) === norm(truth);
    tr = `<div class="truth">True plate: <b>${esc(truth)}</b> → ${ok ? '<span class="ok">read correctly</span>' : '<span class="bad">not exact</span>'}</div>`;
  }
  const evb = r.plates.reduce((a, p) => a + (p.event_bytes || 0), 0);
  const saving = r.photo_bytes && r.plates.length
    ? `<div class="saving">Event messages: ${evb} bytes vs the photo ${(r.photo_bytes / 1024).toFixed(0)} KB → ${(100 * (1 - evb / r.photo_bytes)).toFixed(2)} % less data to send</div>` : '';
  return `<div class="result">
    <div class="result-head"><span class="tag ${r.version}">${r.version.toUpperCase()}</span><span class="fname">${esc(title)}</span>
      ${r.retry_used ? `<span class="badge likely">retry: ${esc(r.retry_used)}</span>` : ''}
      <span class="timing">detect ${r.ms.detect} ms · read ${r.ms.read} ms · total ${tot} ms (${(1000 / tot).toFixed(1)} FPS)</span></div>
    <div class="result-body">
      <div class="photo"><img src="/media/${esc(r.annotated || r.image)}" alt="annotated photo"></div>
      <div class="plates">${tr}${r.plates.length ? r.plates.map(plateCard).join('') : '<div class="noplate">No plate found in this photo.</div>'}${saving}</div>
    </div></div>`;
}

// ---------------------------------------------------------------- photo tab
async function sendPhoto(file) {
  const out = $('#photo-out');
  const holder = document.createElement('div');
  holder.innerHTML = `<p class="empty">Reading ${esc(file.name)}… (the first request loads the models, ~10 s)</p>`;
  if ($('.empty', out) && !$('.result', out)) out.innerHTML = '';
  out.prepend(holder);
  try {
    const j = await api(`/api/recognize?version=${verPhoto()}`, { method: 'POST', body: file,
      headers: { 'X-Filename': encodeURIComponent(file.name) } });
    holder.innerHTML = j.results.map(r => resultBlock(r, file.name)).join('');
  } catch (e) { holder.innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`; toast(e.message, true); }
}
async function sendSample(s) {
  const out = $('#photo-out');
  const holder = document.createElement('div');
  holder.innerHTML = `<p class="empty">Reading test photo ${esc(s.file)}…</p>`;
  if (!$('.result', out)) out.innerHTML = '';
  out.prepend(holder);
  try {
    const j = await api(`/api/sample?version=${verPhoto()}&file=${encodeURIComponent(s.file)}`, { method: 'POST' });
    holder.innerHTML = j.results.map(r => resultBlock(r, `${s.file} (${s.source})`, j.truth)).join('');
  } catch (e) { holder.innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`; toast(e.message, true); }
}
$('#file-photo').addEventListener('change', e => { [...e.target.files].forEach(sendPhoto); e.target.value = ''; });
const drop = $('#drop');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => [...e.dataTransfer.files].filter(f => f.type.startsWith('image/')).forEach(sendPhoto));

// ---------------------------------------------------------------- video tab
$('#file-video').addEventListener('change', e => {
  const f = e.target.files[0]; e.target.value = '';
  if (f) runVideoJob(() => api(`/api/video?version=${verVideo()}`, { method: 'POST', body: f,
    headers: { 'X-Filename': encodeURIComponent(f.name) } }), f.name);
});
$('#demo-video').addEventListener('click', () =>
  runVideoJob(() => api(`/api/video_demo?version=${verVideo()}`, { method: 'POST' }), 'demo_from_test_photos.mp4 (synthetic)'));
async function runVideoJob(start, name) {
  const prog = $('#video-prog'), out = $('#video-out');
  prog.classList.remove('hidden'); $('i', prog).style.width = '0%'; $('span', prog).textContent = 'uploading…';
  out.innerHTML = '';
  try {
    const { job } = await start();
    for (;;) {
      await new Promise(r => setTimeout(r, 800));
      const j = await api('/api/job/' + job);
      const pct = j.total ? Math.min(100, 100 * j.frame / j.total) : 0;
      $('i', prog).style.width = pct + '%';
      $('span', prog).textContent = `frame ${j.frame}${j.total ? ' / ' + j.total : ''}`;
      if (j.state === 'error') throw new Error(j.error);
      if (j.state === 'done') { showVideo(j.result, name); break; }
    }
  } catch (err) { toast(err.message, true); out.innerHTML = `<p class="empty">Error: ${esc(err.message)}</p>`; }
  prog.classList.add('hidden');
}
function showVideo(r, name) {
  const ev = r.events.map((p, i) => plateCard(p, i).replace('<div class="badges">',
    `<div class="badges"><span class="badge">track #${p.track_id}</span><span class="badge">${p.agree}/${p.reads} reads agree</span><span class="badge">${p.first_s}–${p.last_s} s</span>`)).join('');
  $('#video-out').innerHTML = `<div class="result"><div class="result-head"><span class="tag">${esc(name)}</span>
     <span class="timing">${r.frames} frames · ${r.processing_fps} FPS processing · ${r.tracks} tracks · ${r.reads} plate reads</span></div>
     <div class="vgrid" style="padding:14px"><div><video src="/media/${esc(r.video)}" controls></video>
       <p class="hint">Annotated video saved locally (mp4v codec; if your browser cannot play it, open
       <a href="/media/${esc(r.video)}" download>the file</a> in VLC).</p></div>
     <div class="events">${ev || '<p class="empty">No plates read.</p>'}</div></div></div>`;
}

// ---------------------------------------------------------------- camera tab
let camTimer = null, camBusy = false;
$('#cam-start').addEventListener('click', async () => {
  if (!navigator.mediaDevices?.getUserMedia) {        // phones allow the camera only on https pages (or localhost)
    toast('Live camera needs https. On the PC start the server with --phone and open the https:// address it prints. ' +
          'Or use Test photo -> Choose photo -> Camera.', true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: 'environment' } } });
    const v = $('#cam'); v.srcObject = stream; await v.play();
    $('#cam-start').disabled = true; $('#cam-stop').disabled = false;
    camTimer = setInterval(camTick, 350);
  } catch (e) { toast('Camera: ' + e.message, true); }
});
$('#cam-stop').addEventListener('click', () => {
  clearInterval(camTimer); camTimer = null;
  const v = $('#cam'); (v.srcObject?.getTracks() || []).forEach(t => t.stop()); v.srcObject = null;
  $('#cam-start').disabled = false; $('#cam-stop').disabled = true; $('#cam-stat').textContent = 'camera off';
  const c = $('#cam-overlay'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
});
async function camTick() {
  if (camBusy) return;
  const v = $('#cam'); if (!v.videoWidth) return;
  camBusy = true;
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.85));
  try {
    const t0 = performance.now();
    const r = await api(`/api/frame?version=${verCam()}`, { method: 'POST', body: blob });
    const ov = $('#cam-overlay'); ov.width = v.videoWidth; ov.height = v.videoHeight;
    const g = ov.getContext('2d'); g.lineWidth = Math.max(3, ov.width / 300); g.font = `${Math.max(18, ov.width / 45)}px IBM Plex Sans`;
    const col = { verified: '#2fbf71', likely: '#ffb020', review: '#ff4d5e' };
    r.plates.forEach(p => {
      const [x1, y1, x2, y2] = p.box; g.strokeStyle = g.fillStyle = col[p.status] || '#fff';
      g.strokeRect(x1, y1, x2 - x1, y2 - y1); g.fillText(p.text_en || '?', x1, Math.max(20, y1 - 8));
    });
    $('#cam-stat').textContent = `${r.version} · ${r.ms.total} ms model · ${(performance.now() - t0).toFixed(0)} ms round trip · ${r.plates.length} plate(s)`;
    if (r.stored.length) {
      const log = $('#cam-events'); if ($('.empty', log)) log.innerHTML = '';
      r.stored.forEach(p => log.insertAdjacentHTML('afterbegin', `<div class="ev">${p.crop ? `<img src="/media/${esc(p.crop)}">` : ''}
        <div><div class="bn">${esc(p.text)}</div><div class="pen">${esc(p.text_en)}</div>${badge(p.status)}
        ${p.watch_hit ? '<span class="badge hit">WATCHLIST</span>' : ''} <small>${new Date().toLocaleTimeString()}</small></div></div>`));
      if (r.stored.some(p => p.watch_hit)) toast('Watchlist match: ' + r.stored.find(p => p.watch_hit).text, true);
    }
  } catch (e) { $('#cam-stat').textContent = 'error: ' + e.message; }
  camBusy = false;
}

// ---------------------------------------------------------------- history tab
async function loadHistory() {
  try {
    const s = await api('/api/stats');
    const bv = Object.entries(s.by_version).map(([k, v]) => `${k.toUpperCase()} ${v.avg_ms} ms`).join(' · ') || '—';
    $('#stats').innerHTML = [
      [s.runs, 'requests stored'], [s.plates, 'plate readings'], [s.unique_plates, 'unique valid plates'],
      [(s.by_status.verified || 0), 'verified'], [(s.by_status.review || 0), 'sent to review'], [s.watch_hits, 'watchlist hits'],
      [s.payload_bytes_avg ? Math.round(s.payload_bytes_avg) + ' B' : '—', 'avg event message'], [(s.db_bytes / 1024).toFixed(0) + ' KB', 'database size'],
      [bv, 'avg time per photo']].map(([b, t]) => `<div class="stat"><b>${esc(b)}</b><span>${t}</span></div>`).join('');
    const q = new URLSearchParams({ q: $('#h-q').value, status: $('#h-status').value, version: $('#h-ver').value, limit: 300 });
    const rows = await api('/api/history?' + q);
    $('#h-table tbody').innerHTML = rows.map(r => `<tr class="${r.watch_hit ? 'hit' : ''}">
      <td class="mono">${r.id}</td><td class="mono">${new Date(r.created * 1000).toLocaleString()}</td>
      <td>${r.crop ? `<a href="/media/${esc(r.annotated || r.crop)}" target="_blank"><img src="/media/${esc(r.crop)}"></a>` : ''}</td>
      <td class="bn">${esc(r.text) || '—'}</td><td class="mono">${esc(r.text_en)}</td><td>${badge(r.status)}</td>
      <td class="mono">${esc(r.version)}</td><td class="mono">${esc(r.kind)}${r.track_id != null ? ' #' + r.track_id : ''}<br><small>${esc(r.filename || '')}</small></td>
      <td class="mono">${r.payload_bytes ?? ''}</td></tr>`).join('') || '<tr><td colspan="9" class="empty">Nothing stored yet.</td></tr>';
  } catch (e) { toast(e.message, true); }
}
$('#h-refresh').addEventListener('click', loadHistory);
$('#h-q').addEventListener('input', () => { clearTimeout(loadHistory._t); loadHistory._t = setTimeout(loadHistory, 250); });
$('#h-status').addEventListener('change', loadHistory); $('#h-ver').addEventListener('change', loadHistory);
$('#h-clear').addEventListener('click', async () => {
  if (!confirm('Delete ALL stored results, photos and crops from this PC?')) return;
  await api('/api/clear', { method: 'POST' }); toast('All results deleted'); loadHistory();
});
$('#h-ret').addEventListener('click', async () => {
  const d = prompt('Delete results older than how many days? (privacy retention)', '30');
  if (d === null) return;
  const j = await api('/api/retention', { method: 'POST', body: JSON.stringify({ days: Number(d) }) });
  toast(`${j.deleted_plates} readings older than ${d} days deleted`); loadHistory();
});

// ---------------------------------------------------------------- watchlist
async function loadWatch() {
  const w = await api('/api/watchlist');
  $('#w-list').innerHTML = w.map(x => `<div class="witem"><span class="bn">${esc(x.text)}</span>
    <span class="grow">${esc(x.note)} · added ${new Date(x.created * 1000).toLocaleString()}</span>
    <button class="btn danger" data-id="${x.id}">Remove</button></div>`).join('') || '<p class="empty">Watchlist is empty.</p>';
}
$('#w-list').addEventListener('click', async e => {
  if (!e.target.dataset.id) return;
  await fetch('/api/watchlist/' + e.target.dataset.id, { method: 'DELETE' }); loadWatch();
});
$('#w-add').addEventListener('click', async () => {
  try {
    await api('/api/watchlist', { method: 'POST', body: JSON.stringify({ text: $('#w-text').value, note: $('#w-note').value }) });
    $('#w-text').value = $('#w-note').value = ''; loadWatch(); toast('Added to watchlist');
  } catch (e) { toast(e.message, true); }
});

// ---------------------------------------------------------------- report tab
const pct = x => (x == null ? '—' : (+x).toFixed(2) + ' %');
function versionCard(v, R) {
  const c = R.configs[v], g = R.system_gpu?.[v] || {}, cpu = R.system_cpu?.[v] || {};
  const det = R.detectors?.[c.detector.file.replace('.pt', '')] || {};
  const ocr = g.ocr_test_crops || {}, e2e = g.e2e_photos || {};
  const stages = [`Detector ${c.detector.file} (#${c.detector.paper})`]
    .concat(c.retry?.length ? [`retry on miss: ${c.retry.map(r => `${r.file}${r.clahe ? ' + CLAHE' : ''}`).join(' → ')}`] : [])
    .concat([`${c.rule} over ${c.readers.length} readers`])
    .concat(c.status_reader ? [`status reader ${c.status_reader.name}`] : [])
    .concat(c.char_verifiers?.length ? [`char CNNs ${c.char_verifiers.join(' + ')}`] : []);
  const t = [
    ['Detection mAP@50 ≥ 95 % (test, 1,153 photos)', det.test?.mAP50, 95, '%'],
    ['Character accuracy ≥ 90 % (whole photos)', e2e.char_acc_pct, 90, '%'],
    ['Character accuracy ≥ 90 % (test crops)', ocr.char_acc_pct, 90, '%'],
    ['Real time ≥ 10 FPS, GPU (RTX 5060 Ti)', e2e.fps_mean, 10, 'FPS'],
    ['Real time ≥ 10 FPS, CPU only (Core Ultra 7 265K)', cpu.e2e_photos?.fps_mean, 10, 'FPS'],
  ].map(([n, val, thr, u]) => `<tr><td>${n}</td><td><b>${val == null ? '—' : (+val).toFixed(u === 'FPS' ? 1 : 2) + ' ' + u}</b></td>
     <td class="${val >= thr ? 'pass' : 'fail'}">${val == null ? '' : val >= thr ? 'PASS' : 'FAIL'}</td></tr>`).join('');
  const st = Object.entries(ocr.status || {}).map(([k, x]) => `<tr><td>${badge(k)}</td><td>${x.share_pct}% of plates</td><td>${x.exact_pct}% exact</td></tr>`).join('');
  return `<div class="card"><h2>${esc(c.name)}</h2>
    <div class="papers">Built from papers ${c.built_from_papers.map(p => '#' + p).join(', ')}${c.builds_on ? ' · builds on ' + c.builds_on.toUpperCase() : ''}</div>
    <div class="pipeline">${stages.map(s => `<span class="st">${esc(s)}</span>`).join('<span class="ar">→</span>')}</div>
    <div class="kpi"><div><b>${pct(e2e.plate_exact_pct)}</b><span>plate exact, 204 whole photos</span></div>
      <div><b>${pct(ocr.plate_exact_pct)}</b><span>plate exact, 651 test crops</span></div>
      <div><b>${pct(e2e.plate_found_pct)}</b><span>photos with a plate found</span></div></div>
    <h4>Proposal targets (measured on this PC)</h4><table class="targets small-table">${t}</table>
    <h4>Status levels on the 651 test crops</h4><table class="small-table">${st}</table>
    <h4>By source (whole photos)</h4><table class="small-table">${Object.entries(e2e.by_source || {}).map(([k, x]) =>
      `<tr><td>${esc(k)}</td><td>${x.plates} photos</td><td>${pct(x.plate_exact_pct)} exact</td><td>${pct(x.char_acc_pct)} char</td></tr>`).join('')}</table>
  </div>`;
}
function fixKeys(o) {        // JSON keys like "plate_exact_%" -> plate_exact_pct for easy access
  if (Array.isArray(o)) return o.map(fixKeys);
  if (o && typeof o === 'object') return Object.fromEntries(Object.entries(o).map(([k, v]) => [k.replace('_%', '_pct'), fixKeys(v)]));
  return o;
}
async function renderReport() {
  if (!INFO) INFO = await api('/api/info');
  const R = fixKeys(INFO.report);
  const cmp = R.system_gpu?.v2_vs_v1_ocr, cmp2 = R.system_gpu?.v2_vs_v1_e2e;
  const readers = Object.entries(R.readers || {}).sort((a, b) => b[1].test.plate_exact_pct - a[1].test.plate_exact_pct)
    .map(([k, x]) => `<tr><td class="mono">${k}</td><td>#${x.paper}</td><td>${x.kind}</td><td>${pct(x.val.plate_exact_pct)}</td>
      <td>${pct(x.test.plate_exact_pct)}</td><td>${pct(x.test.char_acc_pct)}</td><td>${x.ms_per_plate_batch1} ms</td></tr>`).join('');
  const dets = Object.entries(R.detectors || {}).map(([k, x]) => `<tr><td class="mono">${k}</td><td>#${x.paper}</td><td>${x.imgsz}</td>
      <td>${x.val.mAP50.toFixed(2)}</td><td>${x.test.mAP50.toFixed(2)}</td><td>${x.test.mAP50_95.toFixed(2)}</td><td>${x.test.R.toFixed(1)}</td><td>${x.params_M} M</td></tr>`).join('');
  $('#report').innerHTML = `<div class="rep-grid">${versionCard('v1', R)}${versionCard('v2', R)}
    <div class="card full"><h2>V2 vs V1 on the same plates</h2>
      <p>Test crops: V2 fixes <b>${cmp?.fixed}</b> plates and breaks <b>${cmp?.broken}</b> (exact McNemar p = ${cmp?.p_mcnemar}).
       Whole photos: ${cmp2?.fixed} fixed / ${cmp2?.broken} broken (p = ${cmp2?.p_mcnemar}${cmp2?.p_mcnemar > 0.05 ? ', not significant on 204 photos' : ''}).</p>
      <p class="note">No new training was done for V1 or V2: both are built only from the weights the team trained on Kaggle
       (paper kits, dataset v1.1). Every choice was made on the validation split, then the test split was scored once.
       Numbers here are re-measured on this PC by eval/system_eval.py, not copied from papers.</p></div>
    <div class="card full"><h2>All candidate readers (OCR crops: val 766, test 651)</h2>
      <table class="small-table"><tr><th>reader</th><th>paper</th><th>kind</th><th>val exact</th><th>test exact</th><th>test char acc</th><th>ms/plate (GPU)</th></tr>${readers}</table></div>
    <div class="card full"><h2>All candidate detectors (detection val 896 / test 1,153 photos)</h2>
      <table class="small-table"><tr><th>detector</th><th>paper</th><th>px</th><th>val mAP50</th><th>test mAP50</th><th>test mAP50-95</th><th>test recall</th><th>params</th></tr>${dets}</table>
      <p class="note">mAP50-95 is low for every model because one source (ridoykj) labels boxes tight to the plate edge (found in the Cascaded YOLO and YOLOv8 studies).</p></div>
  </div>`;
}

// ---------------------------------------------------------------- start
(async function init() {
  try {
    INFO = await api('/api/info');
    $('#samples').innerHTML = INFO.samples.map((s, i) => `<div class="sample" data-i="${i}">
      <img src="/sample/${encodeURIComponent(s.file)}" loading="lazy" alt="">
      <div>${esc(s.text)}<small>${esc(s.source)}</small></div></div>`).join('');
    $('#samples').addEventListener('click', e => {
      const el = e.target.closest('.sample'); if (el) sendSample(INFO.samples[+el.dataset.i]);
    });
  } catch (e) { toast('Server not reachable: ' + e.message, true); }
})();
