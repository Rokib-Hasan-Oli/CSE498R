// BanglaLPR Mobile: phone front end for the same local server (V1 / V2 run on the PC).
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let version = localStorage.getItem('lpr-version') || 'v2';

function toast(m, err) {
  const t = $('#toast'); t.textContent = m; t.className = 'toast show' + (err ? ' err' : '');
  clearTimeout(t._h); t._h = setTimeout(() => (t.className = 'toast'), err ? 6000 : 2500);
}
async function api(url, opt = {}) {
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
const chip = s => `<span class="chip ${s}">${s}</span>`;
function alertHit(p) {
  const a = $('#alert');
  a.innerHTML = `WATCHLIST MATCH<span class="bn">${esc(p.text)}</span><small>${esc(p.text_en)} · tap to close</small>`;
  a.classList.add('show'); a.onclick = () => a.classList.remove('show');
  navigator.vibrate?.([300, 120, 300, 120, 600]);
}

// ------------------------------------------------------------------ version + navigation
function setVersion(v) {
  version = v; try { localStorage.setItem('lpr-version', v); } catch (e) { /* private mode */ }
  $$('#ver button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
}
$('#ver').addEventListener('click', e => { if (e.target.dataset.v) { setVersion(e.target.dataset.v); toast(`Using ${e.target.dataset.v.toUpperCase()}`); } });
setVersion(version);
$('#nav').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#nav button').forEach(x => x.classList.toggle('on', x === b));
  $$('.view').forEach(v => v.classList.toggle('on', v.id === 'v-' + b.dataset.v));
  if (b.dataset.v === 'hist') loadHist();
  if (b.dataset.v === 'info') loadInfo();
});

// ------------------------------------------------------------------ live scan
let stream = null, busy = false, timer = null;
$('#scan-btn').addEventListener('click', async () => (stream ? stopScan() : startScan()));
async function startScan() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('The live camera needs the https address (start the PC server with --phone). You can still use Photo.', true); return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 960 } } });
  } catch (e) { toast('Camera: ' + e.message, true); return; }
  const v = $('#video'); v.srcObject = stream; await v.play();
  $('#cam-hint').hidden = true; $('#cam-stat').hidden = false;
  $('#scan-btn').textContent = 'Stop'; $('#scan-btn').classList.add('stop');
  timer = setInterval(tick, 300);
}
function stopScan() {
  clearInterval(timer); timer = null;
  stream?.getTracks().forEach(t => t.stop()); stream = null;
  $('#video').srcObject = null; $('#cam-hint').hidden = false; $('#cam-stat').hidden = true;
  $('#scan-btn').textContent = 'Start scanning'; $('#scan-btn').classList.remove('stop');
  const ov = $('#ov'); ov.getContext('2d').clearRect(0, 0, ov.width, ov.height);
}
async function tick() {
  const v = $('#video'); if (busy || !v.videoWidth) return;
  busy = true;
  const s = Math.min(1, 960 / Math.max(v.videoWidth, v.videoHeight));       // smaller frames = faster over Wi-Fi
  const c = document.createElement('canvas'); c.width = Math.round(v.videoWidth * s); c.height = Math.round(v.videoHeight * s);
  c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
  const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.82));
  try {
    const t0 = performance.now();
    const r = await api(`/api/frame?version=${version}`, { method: 'POST', body: blob });
    drawBoxes(r, c.width, c.height);
    $('#cam-stat').textContent = `${version.toUpperCase()} · model ${Math.round(r.ms.total)} ms · round trip ${Math.round(performance.now() - t0)} ms`;
    const good = r.plates.filter(p => p.valid);
    if (good.length) showLast(good[0]);
    for (const p of r.stored) { navigator.vibrate?.(60); if (p.watch_hit) alertHit(p); }
  } catch (e) { $('#cam-stat').textContent = 'error: ' + e.message; }
  busy = false;
}
function drawBoxes(r, w, h) {
  const v = $('#video'), ov = $('#ov');
  const W = ov.clientWidth, H = ov.clientHeight; ov.width = W * devicePixelRatio; ov.height = H * devicePixelRatio;
  const g = ov.getContext('2d'); g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); g.clearRect(0, 0, W, H);
  const sc = Math.max(W / w, H / h), ox = (W - w * sc) / 2, oy = (H - h * sc) / 2;     // video uses object-fit: cover
  const col = { verified: '#2fbf71', likely: '#ffb020', review: '#ff4d5e' };
  g.lineWidth = 3; g.font = '600 15px system-ui';
  r.plates.forEach(p => {
    const [x1, y1, x2, y2] = p.box.map((q, i) => (i % 2 ? q * sc + oy : q * sc + ox));
    g.strokeStyle = g.fillStyle = col[p.status] || '#fff';
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    g.fillText(p.text_en || '?', x1, Math.max(16, y1 - 6));
  });
}
function showLast(p) {
  $('#last .t').textContent = p.text; $('#last .e').textContent = p.text_en;
  $('#last-chips').innerHTML = chip(p.status) + (p.watch_hit ? '<span class="chip hit">WATCHLIST</span>' : '');
}

// ------------------------------------------------------------------ photo
async function sendPhoto(f) {
  if (!f) return;
  const out = $('#photo-out');
  out.innerHTML = `<div class="card muted">Reading with ${version.toUpperCase()}… (first time: the PC loads the model)</div>`;
  try {
    const j = await api(`/api/recognize?version=${version}`, { method: 'POST', body: f, headers: { 'X-Filename': encodeURIComponent(f.name || 'phone.jpg') } });
    const r = j.results[0];
    out.innerHTML = `<div class="card"><img class="annot" src="/media/${esc(r.annotated)}" alt="">
      <div class="muted">${r.version.toUpperCase()} · ${r.ms.total} ms on the PC · saved to history</div>
      ${r.plates.length ? r.plates.map(plateRow).join('') : '<p>No plate found. Move closer so the plate fills more of the photo.</p>'}</div>`;
    r.plates.forEach(p => p.watch_hit && alertHit(p));
  } catch (e) { out.innerHTML = `<div class="card">Error: ${esc(e.message)}</div>`; }
}
function plateRow(p) {
  return `<div class="plate ${p.watch_hit ? 'hit' : ''}">${p.crop ? `<img src="/media/${esc(p.crop)}" alt="">` : ''}
    <div><div class="t">${esc(p.text) || '(unreadable)'}</div><div class="e">${esc(p.text_en)}</div>
    ${chip(p.status)}${p.watch_hit ? '<span class="chip hit">WATCHLIST</span>' : ''}</div></div>`;
}
$('#cap').addEventListener('change', e => { sendPhoto(e.target.files[0]); e.target.value = ''; });
$('#gal').addEventListener('change', e => { sendPhoto(e.target.files[0]); e.target.value = ''; });

// ------------------------------------------------------------------ history
async function loadHist() {
  try {
    const rows = await api('/api/history?limit=60&q=' + encodeURIComponent($('#q').value));
    $('#hist').innerHTML = rows.map(r => `<div class="plate ${r.watch_hit ? 'hit' : ''}">${r.crop ? `<img src="/media/${esc(r.crop)}" alt="">` : ''}
      <div><div class="t">${esc(r.text) || '—'}</div><div class="e">${esc(r.text_en)}</div>
      ${chip(r.status)}<span class="chip">${esc(r.version)}</span><span class="chip">${new Date(r.created * 1000).toLocaleString()}</span></div></div>`).join('')
      || '<div class="muted">Nothing saved yet.</div>';
  } catch (e) { $('#hist').innerHTML = `<div class="muted">Error: ${esc(e.message)}</div>`; }
}
$('#q').addEventListener('input', () => { clearTimeout(loadHist._t); loadHist._t = setTimeout(loadHist, 250); });

// ------------------------------------------------------------------ model info
async function loadInfo() {
  try {
    const R = (await api('/api/info')).report;
    const card = v => {
      const c = R.configs[v], g = R.system_gpu?.[v]?.e2e_photos || {}, o = R.system_gpu?.[v]?.ocr_test_crops || {}, cpu = R.system_cpu?.[v]?.e2e_photos || {};
      return `<div class="card"><b>${esc(c.name)}</b><div class="muted">${c.readers.length} readers · rule "${esc(c.rule)}" · detector ${esc(c.detector.file)}</div>
        <div class="kpis"><div><b>${g['plate_exact_%']} %</b><span>plate exact, 204 test photos</span></div>
        <div><b>${g['char_acc_%']} %</b><span>character accuracy</span></div>
        <div><b>${o['plate_exact_%']} %</b><span>plate exact, 651 test crops</span></div>
        <div><b>${g.fps_mean} / ${cpu.fps_mean ?? '–'}</b><span>FPS on PC GPU / CPU</span></div></div></div>`;
    };
    $('#info').innerHTML = card('v1') + card('v2') +
      '<div class="card muted">Measured on the team PC (RTX 5060 Ti / Core Ultra 7 265K), dataset v1.1 test split. Both versions use only weights the team trained; no numbers are copied from papers.</div>';
  } catch (e) { $('#info').innerHTML = `<div class="card">Error: ${esc(e.message)}</div>`; }
}

// ------------------------------------------------------------------ server check + install support
(async () => {
  try { await api('/api/stats'); $('#dot').className = 'dot ok'; } catch (e) { $('#dot').className = 'dot bad'; toast('Cannot reach the BanglaLPR server on the PC', true); }
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => { /* untrusted certificate: app still works in the browser */ });
})();
