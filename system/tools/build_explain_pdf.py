"""Write the explanation PDF inside each model folder: V1/BanglaLPR_V1_Explained.pdf and V2/BanglaLPR_V2_Explained.pdf.

  python tools/build_explain_pdf.py            (run tools/build_packages.py first)

Every number is read from the folder's own files: results/summary.json (measured on 31 Oct 2026),
results/evaluate_gpu.json (re-measured from inside the folder), results/training_facts.json (the Kaggle runs' own
results.json / environment.json) and models/config.json. Papers are named, not numbered (team rule for course
documents); weight file names such as p18_v8n_576.pt are code names. HTML -> PDF with headless Chrome.
"""
import html
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
E = html.escape

PAPERS = {   # code -> (short name, full title, authors / venue as in the team notes)
    'p03': ('Cloud Hybrid', 'A Cloud-Based Hybrid Model for Real-Time Detection of BRTA-Approved Licence Plates Using YOLO Tiny and Haar Cascade'),
    'p20': ('Plate SR', 'Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark'),
    'p18': ('Embedded LPR', 'An Embedded Real-Time License Plate Recognition System for Complex Traffic Scenes'),
    'p11': ('Video LPR', 'Real-time Bangla License Plate Recognition System for Low Resource Video-based Applications'),
    'p02': ('Cascaded YOLO', 'Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition'),
    'p08': ('CNN + GFP-GAN', 'Bengali License Plate Recognition: Unveiling Clarity with CNN and GFP-GAN'),
    'p10': ('BLPnet', 'BLPnet: A new DNN model and Bengali OCR engine for Automatic License Plate Recognition'),
    'p27': ('YOLOv10', 'YOLOv10: Real-Time End-to-End Object Detection'),
    'p28': ('YOLOv9', 'YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information'),
    'p21': ('Advanced DL ALPR', 'Advanced deep learning techniques for automated license plate recognition'),
    'p35': ('HVI-CIDNet', 'HVI: A New Color Space for Low-light Image Enhancement'),
    'p23': ('Near-Duplicates', 'Do We Train on Test Data? The Impact of Near-Duplicates on License Plate Recognition'),
    'p06': ('YOLOv8 + EasyOCR', 'YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles'),
    'p19': ('LPLCv2', 'LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification'),
    'p34': ('SVTRv2', 'SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition'),
    'p24': ('Model Fusion', 'Leveraging Model Fusion for Improved License Plate Recognition'),
}
N = lambda code: PAPERS[code][0]
MODEL_NAMES = {
    'p18_v8n_576.pt': 'YOLOv8n plate detector, 576 px', 'p18_slot.pt': 'slot OCR (CNN, 9 fixed character slots)',
    'p11_chars_v8n.pt': 'character YOLOv8n (one box per token)', 'p02_lpnr_v12n.pt': 'character YOLO12n "LPNR", 640 px',
    'p20_crnn_hr.pt': 'CRNN-CTC sequence reader', 'p08_cnn28.pt': 'character CNN 28x28', 'p10_char_cnn32.pt': 'character CNN 32x32',
    'p06_det_v8s.pt': 'YOLOv8s plate detector, 640 px', 'p27_chars_v8n.pt': 'character YOLOv8n (one box per token)',
    'p34_svtrv2_s.onnx': 'SVTRv2-S + CTC reader (ONNX)', 'p34_svtrv2_t.onnx': 'SVTRv2-T + CTC reader (ONNX)',
}


def pval(p):
    return '< 0.0001' if p < 0.0001 else f'{p:.4f}'.rstrip('0').rstrip('.')


def pct(x, d=2):
    return '–' if x is None else f'{x:.{d}f} %'


def load(v):
    d = os.path.join(ROOT, v.upper())
    J = lambda *p: json.load(open(os.path.join(d, *p), encoding='utf-8'))
    ev = os.path.join(d, 'results', 'evaluate_gpu.json')
    return d, J('results', 'summary.json'), J('results', 'training_facts.json'), J('models', 'config.json'), \
        (json.load(open(ev, encoding='utf-8')) if os.path.exists(ev) else None)


def table(head, rows, cls=''):
    h = ''.join(f'<th>{E(str(x))}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def flow_svg(rows):
    """rows: list of lists of (label, sub, kind). Draws boxes left to right per row, arrows between them."""
    W, bw, bh, gap, rh = 760, 160, 54, 18, 86
    col = {'det': ('#e3f1ec', '#006a4e'), 'read': ('#e8eefc', '#2d4fa3'), 'fuse': ('#fff1da', '#9a5c00'),
           'out': ('#fbe6e9', '#b01e33'), 'pre': ('#f0f0f0', '#555')}
    parts, y = [], 10
    for row in rows:
        n = len(row)
        total = n * bw + (n - 1) * gap
        x = (W - total) / 2
        for i, (lab, sub, k) in enumerate(row):
            f, s = col[k]
            parts.append(f'<rect x="{x:.0f}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{f}" stroke="{s}" stroke-width="1.5"/>')
            parts.append(f'<text x="{x + bw / 2:.0f}" y="{y + 22}" text-anchor="middle" font-size="12" font-weight="700" fill="{s}">{E(lab)}</text>')
            parts.append(f'<text x="{x + bw / 2:.0f}" y="{y + 39}" text-anchor="middle" font-size="9.5" fill="#333">{E(sub)}</text>')
            if i < n - 1:
                ax = x + bw
                parts.append(f'<line x1="{ax + 2:.0f}" y1="{y + bh / 2}" x2="{ax + gap - 3:.0f}" y2="{y + bh / 2}" stroke="#666" stroke-width="1.6" marker-end="url(#a)"/>')
            x += bw + gap
        y += rh
    h = y - rh + bh + 14
    return (f'<svg viewBox="0 0 {W} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Arial">'
            '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#666"/></marker></defs>'
            + ''.join(parts) + '</svg>')


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: Arial, 'Nirmala UI', sans-serif; font-size: 10.3pt; line-height: 1.42; color: #18211c; }
.bn { font-family: 'Nirmala UI', Arial, sans-serif; }
h1 { font-size: 22pt; margin: 0 0 4px; color: #0f2b21; }
h2 { font-size: 14pt; color: #0f2b21; border-bottom: 2px solid #006a4e; padding-bottom: 3px; margin: 18px 0 8px; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12px 0 5px; color: #0f2b21; break-after: avoid; }
p { margin: 5px 0; } ul, ol { margin: 4px 0 6px 18px; padding: 0; } li { margin: 2px 0; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 10px; font-size: 9pt; break-inside: avoid; table-layout: auto; }
th { background: #eef4f0; text-align: left; padding: 4px 6px; border: 1px solid #cfd9d2; }
td { padding: 4px 6px; border: 1px solid #dde4de; vertical-align: top; }
.cover { border: 2px solid #0f2b21; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; }
.sub { color: #4d5c54; font-size: 10.5pt; }
.kpis { display: flex; gap: 8px; margin: 12px 0 4px; }
.kpis div { flex: 1; background: #eef4f0; border-radius: 8px; padding: 8px 10px; }
.kpis b { display: block; font-size: 16pt; color: #006a4e; } .kpis span { font-size: 8.5pt; color: #4d5c54; }
.pass { color: #006a4e; font-weight: 700; } .fail { color: #b01e33; font-weight: 700; }
.note { border-left: 3px solid #b8c7bd; padding: 4px 10px; color: #3d4a43; background: #f6f8f6; margin: 8px 0; font-size: 9.4pt; }
.truth { border: 1.5px solid #006a4e; background: #f1f8f4; padding: 8px 12px; border-radius: 8px; margin: 8px 0; font-size: 9.6pt; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f2f4f2; padding: 0 3px; }
pre { font-family: Consolas, monospace; font-size: 8.6pt; background: #f2f4f2; padding: 7px 9px; border-radius: 6px; white-space: pre-wrap; break-inside: avoid; }
.pb { break-before: page; }
.small { font-size: 8.6pt; color: #4d5c54; }
"""


def facts_rows(F):
    rows = []
    for w, f in F.items():
        if w.endswith('.json'):
            continue
        img = f['imgsz']
        img = 'MSR 64x64 / 48x96 / 40x112 / 32x128' if 'svtrv2' in w else ('x'.join(map(str, img)) if isinstance(img, list) else img)
        ep = f"{f['epochs_run'] or f['epochs']}" + (f" of {f['epochs']}" if f['epochs_run'] and f['epochs'] and f['epochs_run'] != f['epochs'] else '')
        mins = f"{f['train_minutes']:.1f}" if isinstance(f['train_minutes'], (int, float)) else '–'
        rows.append([f'<code>{E(w)}</code><br><span class="small">{E(MODEL_NAMES.get(w, ""))}</span>',
                     E(N(w[:3])), E(f['member']), E(str(f['run_date_utc'] or '–')[:16]) + ('*' if 'file time' in str(f['run_date_utc']) else ''),
                     E(str(f['dataset_version'])),
                     E(str(f['start_weights'] or '–')), E(ep), E(str(img)), mins])
    t = table(['Weight file / model', 'Paper study', 'Trained by', 'Kaggle run (UTC)', 'Data', 'Start weights',
               'Epochs', 'Input', 'Train min'], rows)
    if any('file time' in str(f['run_date_utc']) for f in F.values()):
        t += '<p class="small">* the BLPnet notebook writes no run date; this is the time of its files in the results zip.</p>'
    return t


def targets(S, D, ev):
    g, c, det = S['gpu'], S['cpu'], S['detectors'].get('p18_v8n_576', {})
    fps2 = f" (re-run from this folder: {ev['e2e_photos']['fps_mean']} FPS)" if ev else ''
    rows = [
        ['Plate detection mAP@50 ≥ 95 %', 'detection test split, 1,153 photos', pct(det.get('test', {}).get('mAP50')), det.get('test', {}).get('mAP50', 0) >= 95],
        ['Character recognition accuracy ≥ 90 %', '651 test plate crops', pct(g['ocr_test_crops']['char_acc_%']), g['ocr_test_crops']['char_acc_%'] >= 90],
        ['Character recognition accuracy ≥ 90 %', '204 whole test photos (detect + read)', pct(g['e2e_photos']['char_acc_%']), g['e2e_photos']['char_acc_%'] >= 90],
        ['Real time ≥ 10 FPS', 'whole photo, GPU RTX 5060 Ti' + fps2, f"{g['e2e_photos']['fps_mean']} FPS", g['e2e_photos']['fps_mean'] >= 10],
        ['Real time ≥ 10 FPS', 'whole photo, CPU only (Core Ultra 7 265K, 8 threads)', f"{c['e2e_photos']['fps_mean']} FPS", c['e2e_photos']['fps_mean'] >= 10],
        ['(stricter) whole plate exactly right', 'test crops / whole photos', f"{pct(g['ocr_test_crops']['plate_exact_%'])} / {pct(g['e2e_photos']['plate_exact_%'])}", None],
    ]
    return table(['Proposal target', 'Measured on', 'Result', ''],
                 [[E(a), E(b), f'<b>{E(r)}</b>', '' if ok is None else ('<span class="pass">PASS</span>' if ok else '<span class="fail">FAIL</span>')]
                  for a, b, r, ok in rows])


def results_block(v, S, ev):
    g, c, b = S['gpu'], S['cpu'], S['breakdown_ocr_test']
    o, e = g['ocr_test_crops'], g['e2e_photos']
    rows = [['651 test crops (reader part)', pct(o['plate_exact_%']), pct(o['char_acc_%']), pct(o['CER_%']), f"{o['ms_per_plate']} ms / plate"],
            ['204 whole photos (detect + read)', pct(e['plate_exact_%']), pct(e['char_acc_%']), pct(e['CER_%']),
             f"{e['ms_per_photo_mean']} ms / photo (GPU) · {c['e2e_photos']['ms_per_photo_mean']} ms (CPU)"]]
    out = table(['Test set', 'Plate exact', 'Character accuracy', 'CER', 'Time'], rows)
    out += f"<p>Plates found in the whole photos: <b>{pct(e['plate_found_%'])}</b>."
    if e.get('retry_used_photos'):
        out += f" The retry detector was used on {e['retry_used_photos']} of the 204 photos."
    out += '</p>'
    if ev:
        out += (f"<p class='small'>Re-measured from inside this folder with <code>evaluate.py</code>: test crops "
                f"{pct(ev['ocr_test_crops']['plate_exact_%'])}, whole photos {pct(ev['e2e_photos']['plate_exact_%'])}, "
                f"{ev['e2e_photos']['fps_mean']} FPS (speed varies a few FPS between runs; accuracy is identical).</p>")
    cond = [[k, str(x['plates']), pct(x['plate_exact_%']), pct(x['char_acc_%'])] for k, x in b.items() if k in ('clear', 'night', 'blurry')]
    src = [[k, str(x['plates']), pct(x['plate_exact_%']), pct(x['char_acc_%'])] for k, x in e['by_source'].items()]
    out += '<h3>By condition (test crops) and by source (whole photos)</h3>'
    out += table(['Subset', 'Plates', 'Plate exact', 'Char accuracy'], cond + [['— whole photos: ' + r[0]] + r[1:] for r in src])
    st = S['gpu']['ocr_test_crops']['status']
    out += '<h3>Status levels shown to the operator (test crops)</h3>'
    out += table(['Status', 'Share of plates', 'Exact among them', 'Meaning'],
                 [[k, pct(x['share_%'], 1), pct(x['exact_%']),
                   {'verified': 'two independent readers agree on a valid BRTA plate: accept automatically',
                    'likely': 'valid plate, no second reader agrees, but the character CNNs agree with ≥ 80 % of characters',
                    'review': 'broken layout or disagreement: a person should check the crop'}[k]] for k, x in st.items()])
    er = b.get('errors', {})
    out += ('<p>Remaining errors on the test crops: ' + ', '.join(f'{k} {n}' for k, n in er.items()) + '.</p>')
    return out


def readers_table(S, used):
    rows = []
    for k, x in sorted(S['readers'].items(), key=lambda kv: -kv[1]['test']['plate_exact_%']):
        rows.append([f'<code>{E(k)}</code>', E(N(k[:3])), E(x['kind']), pct(x['val']['plate_exact_%']),
                     pct(x['test']['plate_exact_%']), pct(x['test']['char_acc_%']), f"{x['ms_per_plate_batch1']} ms",
                     '<b>used</b>' if k in used else 'not used'])
    return table(['Reader', 'Paper study', 'Kind', 'Val exact', 'Test exact', 'Test char acc', 'ms/plate', 'In this version'], rows)


def detectors_table(S, used):
    rows = []
    for k, x in sorted(S['detectors'].items(), key=lambda kv: -kv[1]['val']['mAP50']):
        rows.append([f'<code>{E(k)}</code>', E(N(k[:3])), str(x['imgsz']), f"{x['val']['mAP50']:.2f}", f"{x['test']['mAP50']:.2f}",
                     f"{x['test']['mAP50_95']:.2f}", f"{x['test']['R']:.1f}", f"{x['params_M']} M", '<b>used</b>' if k in used else 'not used'])
    return table(['Detector', 'Paper study', 'px', 'Val mAP50', 'Test mAP50', 'Test mAP50-95', 'Test recall', 'Params', 'In this version'], rows)


def strategy_table(S, keys):
    rows = []
    for k in keys:
        va, te = S['detector_strategies']['val'].get(k), S['detector_strategies']['test'].get(k)
        if va:
            rows.append([E(k), pct(va['found_%']), str(va['missed_photos']), pct(te['found_%']), str(te['missed_photos']), pct(va['extra_box_%'])])
    return table(['Strategy', 'Val plates found', 'Val photos missed', 'Test plates found', 'Test photos missed', 'Val extra boxes'], rows)


USAGE = """<h2>7. How to use it</h2>
<h3>Command line (this folder alone)</h3>
<p>Install once with <code>pip install -r requirements.txt</code> (the team venv <code>D:/498R/_dataset_build/venv</code> already
has everything), then run from inside this folder:</p>
<table><thead><tr><th>Command</th><th>What it does</th></tr></thead><tbody>
<tr><td><code>python predict.py car.jpg</code></td><td>reads one photo: plate text, status, timing; annotated photo in <code>predictions/</code></td></tr>
<tr><td><code>python predict.py photos_folder/</code></td><td>every photo in a folder, all results in <code>predictions/results.json</code></td></tr>
<tr><td><code>python predict.py traffic.mp4</code></td><td>a video: one reading per vehicle + an annotated video</td></tr>
<tr><td><code>python predict.py car.jpg --cpu</code></td><td>the same without a GPU</td></tr>
<tr><td><code>python evaluate.py --data D:/498R/dataset</code></td><td>re-measures the numbers in this PDF on the test sets</td></tr>
</tbody></table>
<h3>Web app and mobile app (both versions, one server on the PC)</h3>
<pre>D:/498R/_dataset_build/venv/Scripts/python.exe D:/498R/BanglaLPR_System/webapp/server.py --phone --port 8498</pre>
<ul><li><b>Web app</b> (PC browser): <code>https://localhost:8498</code> — test photo (V1, V2 or compare), video, live camera,
stored results (SQLite, CSV export, retention), watchlist, model report.</li>
<li><b>Mobile app</b> (phone on the same Wi-Fi): open the https address the server prints, followed by <code>/app</code>,
e.g. <code>https://192.168.0.40:8498/app</code>. Accept the self-signed certificate (Advanced → Proceed), then Chrome menu →
<b>Add to Home screen</b>. Tabs: Scan (live back camera, boxes and text over the video, vibration, watchlist alert),
Photo (camera or gallery), History, Models; V1/V2 switch at the top. The models run on the PC; the phone sends frames
over Wi-Fi.</li>
<li><b>Android app, fully offline</b>: <code>BanglaLPR_{V}.apk</code> in this folder (install guide: <code>BanglaLPR_System/android/README.md</code>). The same models
(converted to ONNX) and the same reading rules, re-written in Java, run <b>on the phone</b> with ONNX Runtime; the app
has no internet permission. Scan (live camera), Photo, History (watchlist, CSV export), About. On the PC the Java
engine gave the same plate text as this folder's Python code on {SAME}; speed on a phone was not measured (the app
shows the time per frame).</li>
<li>From Python: <code>from lpr.pipeline import LPRSystem; lpr = LPRSystem('path/to/this/folder')</code>,
then <code>lpr.process(cv2.imread('car.jpg'))</code>.</li></ul>"""


def build(v):
    d, S, F, C, ev = load(v)
    sel = S['reader_selection_on_val']
    ch, single = sel['chosen_on_val'], sel['best_single_on_val']
    g = S['gpu']
    used_readers = [r['name'] for r in C['readers']] + ([C['status_reader']['name']] if C.get('status_reader') else [])
    used_dets = [C['detector']['file'][:-3]] + [r['file'][:-3] for r in C.get('retry', [])]
    title = 'BanglaLPR V1 — Bangla cascade' if v == 'v1' else 'BanglaLPR V2 — multi-reader verify'
    papers = ['p03', 'p20', 'p18', 'p11', 'p02', 'p08', 'p10'] if v == 'v1' else ['p27', 'p28', 'p21', 'p35', 'p23', 'p06', 'p19', 'p34']
    H = [f'<html><head><meta charset="utf-8"><title>{E(title)}</title><style>{CSS}</style></head><body>']
    # ------------------------------------------------------------------ cover
    lead = ('V1 is the first complete BanglaLPR model. It is assembled only from models trained in the team\'s studies of '
            'seven Bangla-plate papers: one plate detector, three plate readers combined with a BRTA-layout fallback, an '
            'independent reader that gives every plate a second opinion, and two character CNNs that flag doubtful plates.'
            if v == 'v1' else
            'V2 is V1 upgraded with eight more papers. It keeps V1\'s detector and main reader and adds what those papers '
            'measured to work: a second detector that runs only when the first finds nothing (with a low-light retry), an '
            'always-on second reader, two SVTRv2 readers that join only when the first two disagree, and a status that '
            'comes from reader agreement. Everything was chosen on validation data and tested once.')
    H.append(f'''<div class="cover"><h1>{E(title)}</h1>
<div class="sub">Real-time Bangla licence plate detection and recognition · NSU CSE 498R, Fall 2026<br>
Team: Md. Yousuf · MD. Rokib Hasan Oli · Tashfia Adiba Zaman Nizhum · Md Rakibul Hasan<br>
Built 31 Oct 2026, explained 1 Nov 2026 · dataset 498R BanglaLPR All-in-One v1.1 · folder <code>BanglaLPR_System/{v.upper()}</code></div>
<p style="margin-top:10px">{E(lead)}</p>
<div class="kpis"><div><b>{pct(g['e2e_photos']['plate_exact_%'])}</b><span>whole plate exact, 204 test photos</span></div>
<div><b>{pct(g['e2e_photos']['char_acc_%'])}</b><span>character accuracy, whole photos</span></div>
<div><b>{pct(g['ocr_test_crops']['plate_exact_%'])}</b><span>plate exact, 651 test crops</span></div>
<div><b>{g['e2e_photos']['fps_mean']} / {S['cpu']['e2e_photos']['fps_mean']}</b><span>FPS whole photo, GPU / CPU only</span></div></div></div>
<div class="truth"><b>What is real in this document.</b> No model was trained for this version. It is built only from weight
files the team trained on Kaggle (section 4 lists each run: who, when, data version, epochs). The new part is how
they are combined, and every combination was chosen on the <b>validation</b> split before the <b>test</b> split was
scored once. All numbers were measured on the team PC (RTX 5060 Ti 16 GB, Core Ultra 7 265K) and are stored in
<code>results/</code>; none are copied from the papers.</div>''')
    H.append('<h2>1. Proposal targets</h2>' + targets(S, d, ev))
    if v == 'v2':
        vs = S['v2_vs_v1']
        H.append(f"""<p><b>V2 against V1 on the same plates:</b> on the 651 test crops V2 fixes {vs['v2_vs_v1_ocr']['fixed']} plates
and breaks {vs['v2_vs_v1_ocr']['broken']} (exact McNemar p = {pval(vs['v2_vs_v1_ocr']['p_mcnemar'])}, a real gain). On the 204
whole photos it fixes {vs['v2_vs_v1_e2e']['fixed']} and breaks {vs['v2_vs_v1_e2e']['broken']} (p = {pval(vs['v2_vs_v1_e2e']['p_mcnemar'])}:
not significant on so few photos). All of the photo gain is on night photos (NightLPBD
{pct(S['gpu']['e2e_photos']['by_source']['nightlpbd']['plate_exact_%'])} vs 71.67 % for V1); day photos are unchanged.</p>""")
    # ------------------------------------------------------------------ how it works
    if v == 'v1':
        flow = flow_svg([
            [('Photo / frame', 'camera, file, video', 'pre'), ('Plate detector', 'YOLOv8n-576', 'det'), ('Crop + 8 %', 'tiny: bicubic x2', 'pre'), ('Primary reader', 'char-YOLOv8n', 'read')],
            [('Layout check', 'district-letter / 6 digits', 'fuse'), ('Fallback readers', 'LPNR YOLO12n + slot OCR', 'read'), ('Vote', 'layout-valid first', 'fuse'), ('Plate text', 'Bangla + English', 'out')],
            [('Second opinion', 'CRNN reader', 'read'), ('Character CNNs', 'two CNNs re-check boxes', 'read'), ('Status', 'verified / likely / review', 'out'), ('Store / send', 'SQLite, 170-byte event', 'out')],
        ])
        steps = f"""<ol>
<li><b>Detect.</b> The YOLOv8n plate detector from the {N('p18')} study looks at the whole photo at 576 px and returns plate boxes (confidence ≥ 0.25).</li>
<li><b>Crop.</b> Each box is cut out with 8 % padding (the {N('p02')} cascade). A crop lower than 32 px is enlarged x2 with bicubic resizing.</li>
<li><b>Read.</b> The character-YOLOv8n from the {N('p11')} study finds every token (district word, মেট্রো, series letter, digits). The BRTA decoder from the {N('p02')} study puts the words on top and the six digits below by token kind, giving e.g. <span class="bn">ঢাকা মেট্রো-গ ১৭-৪১১৯</span>.</li>
<li><b>Check the layout.</b> A real plate is one district (+ optional মেট্রো), one letter and six digits. If the reading breaks this, the <b>fallback</b> runs: the LPNR char-YOLO12n ({N('p02')}) and the slot OCR ({N('p18')}) also read the crop, and the most common layout-valid reading wins (tie → primary reader). This happened on {ch['helpers_test_%']} % of test plates.</li>
<li><b>Second opinion.</b> The CRNN reader trained in the {N('p20')} study reads every plate independently. If two readers give the same plate, the status is <b>verified</b>.</li>
<li><b>Character check.</b> The character CNNs from the {N('p08')} and {N('p10')} studies re-classify each character box. If they agree with fewer than 80 % of the characters, an unverified plate becomes <b>review</b>, otherwise <b>likely</b>.</li>
<li><b>Output.</b> Bangla text, English text (e.g. "Dhaka Metro-Ga 17-4119"), box, status, timings; stored in the local database and shown as a ~170-byte event message (the {N('p03')} edge-to-server idea).</li></ol>"""
    else:
        flow = flow_svg([
            [('Photo / frame', 'camera, file, video', 'pre'), ('Detector (V1)', 'YOLOv8n-576', 'det'), ('Nothing found?', 'YOLOv8s (EasyOCR study)', 'det'), ('Still nothing?', 'YOLOv8s on CLAHE copy', 'det')],
            [('Crop + 8 %', 'tiny: bicubic x2', 'pre'), ('Reader 1 (V1)', 'char-YOLOv8n', 'read'), ('Reader 2 (new)', 'char-YOLOv8n, YOLOv10 run', 'read'), ('Same valid plate?', '86 % stop here', 'fuse')],
            [('Helpers if they differ', 'LPNR + SVTRv2-S / -T', 'read'), ('Vote', 'layout-valid first', 'fuse'), ('Status', '2 agree = verified', 'out'), ('Store / send', 'SQLite, 170-byte event', 'out')],
        ])
        steps = f"""<ol>
<li><b>Detect (kept from V1).</b> The YOLOv8n-576 detector finds the plates.</li>
<li><b>Retry only on a miss (new).</b> If it finds nothing, the YOLOv8s detector from the {N('p06')} study looks at the same photo. If that also finds nothing, it looks once more at a CLAHE-brightened copy at confidence 0.35. The {N('p35')} study showed that enhancing <i>every</i> photo adds false boxes and does not help reading, so V2 enhances only after a miss. Used on {g['e2e_photos'].get('retry_used_photos')} of the 204 test photos.</li>
<li><b>Two readers always (new).</b> The {N('p11')} character-YOLOv8n (V1's reader) and the character-YOLOv8n trained in the {N('p27')} study both read every crop.</li>
<li><b>Verify rule (new).</b> If both give the same layout-valid plate, that is the answer ({100 - ch['helpers_test_%']:.1f} % of test plates). Otherwise three helpers also read it: the LPNR char-YOLO12n ({N('p02')}, from V1) and SVTRv2-S and SVTRv2-T ({N('p34')}). The most common layout-valid reading wins (tie → reader 1).</li>
<li><b>Status from agreement (new).</b> The {N('p19')} study found that the readers' own agreement/confidence is a better "is this right?" signal than a separate legibility classifier, so a plate is <b>verified</b> when at least two readers agree, otherwise <b>review</b>.</li>
<li><b>Honest evaluation.</b> Every choice was made on validation data and the test split was used once, on a split where no plate appears in two splits — the lesson of the {N('p23')} study, which showed that leaked test plates inflate scores.</li>
<li><b>Output</b> as in V1: Bangla + English text, box, status, timings, stored locally, ~170-byte event message.</li></ol>"""
    H.append('<h2 class="pb">2. How it works</h2>' + flow + steps)
    # ------------------------------------------------------------------ papers
    if v == 'v1':
        rows = [
            ['p18', 'Plate detector <code>p18_v8n_576.pt</code> (best val mAP50 of the six V1-paper detectors, 3.0 M parameters); slot OCR <code>p18_slot.pt</code> as second fallback reader', 'dynamic INT8 OCR (−20 points in its Kaggle run)'],
            ['p11', 'Primary reader <code>p11_chars_v8n.pt</code> (86.18 % alone); video mode: track each plate and vote until 3 readings agree', 'Haar gate (2.47 % recall in our run); 320 px detector (val mAP50 94.80)'],
            ['p02', 'First fallback reader <code>p02_lpnr_v12n.pt</code> at 640 px (87.71 % alone); the BRTA layout decoder; 8 % crop padding', 'its YOLO12s detector (val 96.80 < 96.92); vehicle stage'],
            ['p20', 'Second-opinion reader <code>p20_crnn_hr.pt</code> (CRNN, 75.73 % alone) for the status', 'its super-resolution network: worse than bicubic on small plates (section 5)'],
            ['p08', 'Character CNN <code>p08_cnn28.pt</code> re-checks every character box', 'GFP-GAN restoration (205 ms/plate, no exact gain); contour segmentation'],
            ['p10', 'Character CNN <code>p10_char_cnn32.pt</code> (rebuilt from the team\'s BLPnet notebook) as second checker', 'its YOLOv8n detector (val 96.73); Inception box regressor'],
            ['p03', 'System design: read at the edge, send only small results to a server (the web/mobile app and the event message)', 'Haar cascade (F1 4.43 % in our run); its YOLOv8n 416/640 detectors (val 96.66 / 96.22)'],
        ]
    else:
        rows = [
            ['p27', 'Always-on second reader <code>p27_chars_v8n.pt</code> (86.02 % alone)', 'YOLOv10n/s detectors: higher val mAP50 (96.87 / 97.28) but fewer val plates found (94.94 / 95.41 % vs 96.35 %); char-YOLOv10n reader (83.72 %)'],
            ['p34', 'Helper readers SVTRv2-S and SVTRv2-T (81.87 / 81.26 % alone, but they make different mistakes), exported to ONNX', '—'],
            ['p06', 'Retry detector <code>p06_det_v8s.pt</code>, used only when V1\'s detector finds nothing', 'EasyOCR (13.52 % exact)'],
            ['p35', 'Low-light rule: brighten (CLAHE) only after a miss', 'the HVI-CIDNet network itself (no gain on night plates in its run)'],
            ['p19', 'Status from reader agreement (its study: confidence/agreement beat the classifier as a gate)', 'legibility classifier (81.0 % vs 84.45 % at 80 % coverage)'],
            ['p23', 'Evaluation protocol: grouped split, choose on val, test once', 'KUET-only "fair" reader (62.37 % on the full test)'],
            ['p28', '— evaluated, no component runs in V2', 'YOLOv9s: val mAP50 97.11 but val plates found 95.41 %; as a retry detector 96.71 % vs 97.18 % for the chosen chain'],
            ['p21', '— evaluated, no component runs in V2', 'its YOLOv9s / v10s / v8s / v5su detectors score the same as the YOLOv9 / YOLOv10 runs; Tesseract (0.77 %)'],
        ]
    H.append('<h2>3. The papers and what each one contributes</h2>')
    H.append(table(['Paper (short name)', 'Used in ' + v.upper(), 'Tested, not used (why)'],
                   [[f'<b>{E(N(c))}</b><br><span class="small">{E(PAPERS[c][1])}</span>', a, b] for c, a, b in rows]))
    if v == 'v2':
        H.append(f"""<p class="note">The fusion idea (a vote that prefers readings with a valid BRTA layout) extends the team's
{N('p24')} study. {N('p28')} and {N('p21')} do not contribute a running component: their detectors were measured on
validation and lost. Adding one anyway would mean choosing by the test set, which is exactly what the {N('p23')} study
warns against. To use YOLOv9s anyway, change <code>retry[0].file</code> in <code>models/config.json</code> to
<code>p28_v9s.pt</code> (also copy that file into <code>models/weights</code>).</p>
<h3>What V2 changes compared with V1</h3>""" + table(['Part', 'V1', 'V2'], [
            ['Detector', 'YOLOv8n-576 only', 'same + YOLOv8s retry on a miss + CLAHE retry'],
            ['Readers', '3 (char-YOLOv8n, LPNR YOLO12n, slot OCR)', '5 (char-YOLOv8n x2, LPNR YOLO12n, SVTRv2-S, SVTRv2-T); slot OCR dropped'],
            ['Rule', 'fallback: helpers only when the layout breaks', 'verify: two readers always, helpers when they disagree'],
            ['Status', 'CRNN second opinion + character CNNs', 'agreement of the readers that ran'],
            ['Auto-accepted plates', '78.0 % (96.06 % exact)', f"{S['gpu']['ocr_test_crops']['status']['verified']['share_%']} % ({S['gpu']['ocr_test_crops']['status']['verified']['exact_%']} % exact)"],
            ['Plate exact, crops / photos', '90.02 % / 89.22 %', f"{pct(g['ocr_test_crops']['plate_exact_%'])} / {pct(g['e2e_photos']['plate_exact_%'])}"],
        ]))
    # ------------------------------------------------------------------ training
    H.append('<h2 class="pb">4. The trained models inside this folder and how they were trained</h2>')
    H.append('<p>All weights were trained on free Kaggle Tesla T4 x2 sessions with the team\'s paper kits (same code, same data split, seed 42). '
             'The facts below are read from each run\'s own <code>results.json</code> and <code>environment.json</code> '
             '(<code>results/training_facts.json</code>).</p>')
    H.append(facts_rows(F))
    olds = [w for w, f in F.items() if str(f.get('dataset_version', '')).startswith('1.0')]
    if olds:
        H.append(f'<p class="note">{", ".join("<code>" + E(w) + "</code>" for w in olds)} were trained on dataset v1.0 (before the '
                 'BLPD crop fix of 20 Oct, which changed 91 BLPD rows). They are scored here on the v1.1 test split like '
                 'everything else; a v1.1 re-run of that kit is listed under improvements.</p>')
    H.append('''<h3>What the data is</h3><p>498R BanglaLPR All-in-One v1.1: photos and plate crops from the team Drive and 11 public
Bangladeshi sources, de-duplicated, with a <b>grouped 70/15/15 split</b> so that the same plate or near-duplicate photo
never appears in two splits. Detection 8,574 / 896 / 1,153 photos (train / val / test); plate-text crops 5,607 / 766 /
651; whole-photo end-to-end test 204 (144 BLPD external day photos + 60 NightLPBD night photos). Validation and test use
human labels only.</p>''')
    # ------------------------------------------------------------------ selection
    H.append('<h2>5. How this version was assembled (no peeking at test)</h2>')
    H.append(f"""<p><b>Readers and rule:</b> every ordered combination of the candidate readers (up to 5) with every rule was
scored on the 766 validation crops; the best one (ties → faster) is
<b>{' + '.join(ch['readers'])}</b> with the <b>{ch['rule']}</b> rule: val {pct(ch['val']['plate_exact_%'])}, then test
{pct(ch['test']['plate_exact_%'])}. The best single reader on val ({single['readers'][0]}) scores
{pct(single['test']['plate_exact_%'])} on test; the combination fixes {sel['vs_best_single_test']['fixed']} test plates and breaks
{sel['vs_best_single_test']['broken']} (exact McNemar p {pval(sel['vs_best_single_test']['p_mcnemar'])}). Ceiling if we always picked the
right reader (oracle): {pct(sel['oracle_test_%'])}.</p>""")
    H.append(readers_table(S, used_readers))
    H.append('<p><b>Detector:</b> each candidate scored on the 896 validation photos; the system rule counts plates actually found '
             '(IoU ≥ 0.5 at confidence 0.25), because a higher mAP50 did not always mean more plates found.</p>')
    H.append(detectors_table(S, used_dets))
    if v == 'v2':
        H.append(strategy_table(S, ['p18_v8n_576', 'p06_det_v8s', 'p28_v9s', 'p27_v10s', 'p18_v8n_576 -> p28_v9s on miss',
                                    'p18_v8n_576 -> p27_v10s (+CLAHE) on miss', 'p18_v8n_576 -> p06_det_v8s on miss',
                                    'p18_v8n_576 -> p06_det_v8s (+CLAHE) on miss']))
    else:
        cv, up = S['char_verifier_val'], S['tiny_plate_upscale_val']
        H.append(f"""<p><b>Character checkers:</b> on validation, when the {N('p08')} CNN agrees with fewer than 80 % of the
characters only {pct(cv['p08']['below_0.8']['exact_%'], 1)} of those plates are right ({pct(cv['p08']['at_least_0.8']['exact_%'], 1)} otherwise);
the {N('p10')} CNN: {pct(cv['p10']['below_0.8']['exact_%'], 1)} vs {pct(cv['p10']['at_least_0.8']['exact_%'], 1)}. Their mean is the best
error signal (AUROC {cv['mean_p08_p10']['auroc']}; the {N('p08')} CNN alone {cv['p08']['auroc']}, the {N('p10')} CNN {cv['p10']['auroc']}).</p>
<p><b>Tiny plates:</b> validation crops shrunk to 24 px high read {up['h24_small']} % as they are, {up['h24_bicubic']} % after bicubic x2 and
only {up['h24_sr']} % after the {N('p20')} super-resolution network (32 px: {up['h32_small']} / {up['h32_bicubic']} / {up['h32_sr']} %). So V1
uses bicubic x2 for crops under 32 px, and the SR network is not used.</p>""")
    # ------------------------------------------------------------------ results
    H.append('<h2 class="pb">6. Results</h2>' + results_block(v, S, ev))
    # ------------------------------------------------------------------ usage
    same = ('651 / 651 test crops and 203 / 204 test photos' if v == 'v1' else '651 / 651 test crops and 204 / 204 test photos')
    H.append(USAGE.replace('{V}', v.upper()).replace('{SAME}', same))
    # ------------------------------------------------------------------ improve + next
    if v == 'v1':
        imp = [
            '<b>Add a second always-on reader (done in V2).</b> V1 accepts only 78 % of plates automatically; an independent second reader raises that to ~95 % and the "verify" rule fixed 12 more test plates (V2).',
            '<b>Retry when nothing is found (done in V2).</b> A second detector on a miss, then a CLAHE copy, finds 97.18 % of val plates instead of 96.35 %.',
            f'<b>Re-train the {N("p20")} CRNN on dataset v1.1</b> (its run used v1.0) and score it again with <code>evaluate.py</code>.',
            '<b>Fix the remaining digit errors:</b> 26 of V1\'s 65 wrong test plates have one wrong digit in a valid layout. More training crops of hard digits (night, blur) and grammar-constrained decoding for the sequence readers help here.',
            '<b>Night data:</b> night crops 80.00 % vs clear 93.75 %. Add IR-lit or night video frames, then retrain the char-YOLO readers with night augmentations (gamma, motion blur, noise).',
            '<b>Re-train the final readers on train + val</b> once all choices are fixed (+14 % data), and report that version on test only.',
        ]
    else:
        imp = [
            '<b>Record real video and use the track vote.</b> The video mode already tracks each plate and votes over its readings; a labelled Bangla multi-frame set (one string per vehicle) would measure the gain. Every top team of the ICPR 2026 low-resolution LPR competition fused frames.',
            '<b>Grammar-constrained decoding</b> for the SVTRv2 / CRNN readers (only district words, letters and 6 digits allowed), so the helpers stop voting for impossible strings.',
            f'<b>Night and blur:</b> night crops {pct(S["breakdown_ocr_test"]["night"]["plate_exact_%"])}, blurry {pct(S["breakdown_ocr_test"]["blurry"]["plate_exact_%"])} vs clear {pct(S["breakdown_ocr_test"]["clear"]["plate_exact_%"])}. Fix capture first (850 nm IR light, fast shutter), then add night training data pseudo-labelled from video tracks where 3 of 3 readings agree.',
            f'<b>Calibrated review threshold</b> (split conformal prediction on validation) so that auto-accepted plates reach a stated accuracy such as 99 %; today verified plates are {pct(S["gpu"]["ocr_test_crops"]["status"]["verified"]["exact_%"])} right.',
            f'<b>Re-train the {N("p27")} character reader on dataset v1.1</b> (its run used v1.0) and retrain all chosen readers on train + val for the final version.',
            '<b>Single-line motorcycle front plates</b> (mandatory from 15 Nov 2026): collect and label them, add a one-line layout rule (today the layout check expects words above digits) and fine-tune the readers.',
            '<b>Measure on edge hardware</b> (Raspberry Pi 5 + Hailo NPU, Jetson Orin Nano): export the detectors and char-YOLO readers (ONNX / TensorRT / Hailo), keep the sequence readers in FP16/FP32 (dynamic INT8 cost the slot OCR 20 points), time <code>evaluate.py --cpu</code> on the board.',
        ]
    nxt = [
        '<b>Make the offline Android app faster.</b> The APK already runs everything on the phone (ONNX Runtime, CPU + XNNPACK). Next: measure ms per frame on a few real phones, try the NNAPI / GPU execution providers, and a calibrated INT8 plate detector (keep the readers in FP32: dynamic INT8 cost the slot OCR 20 points).',
        '<b>Edge unit + fast data:</b> a Raspberry Pi 5 + Hailo box (~7-8 W) at a gate or toll lane sends only ~170-byte plate events over MQTT on dual-SIM 4G (about 0.3 GB a month per camera, against ~1,300 GB to stream 1080p video), with a local outbox for power cuts. Starlink only for sites without 4G, one dish per toll plaza or highway hub serving many cameras.',
        '<b>Pilot:</b> (1) NSU gate, 2-4 weeks, real video accuracy and review rate; (2) Pi + Hailo unit at a parking entry for 30 days; (3) a toll plaza or low-traffic tunnel with the operator, comparing our vehicle count with theirs.',
        '<b>Uses in Bangladesh</b> (sources in the team\'s research report): DMP AI-camera enforcement (a cheaper unit per intersection), toll backup where RFID tags fail, watchlist alerts for stolen / fake-plate vehicles, fuel-pump plate logs during rationing, port gate passes, parking, ride-share driver verification.',
        '<b>Privacy by design</b> (Personal Data Protection Act 2026): plate + time + place is personal data; keep data in Bangladesh, delete non-hit reads after 72 h (enforcement) or 30 days (parking), keyed hashes for counting, a person confirms every hit. The app already has local storage and a retention button.',
        '<b>Publishable research:</b> "A Bangla multi-frame plate benchmark with calibrated abstention, measured on a solar-capable Pi 5 + Hailo unit" — no Bangla plate paper has multi-frame fusion, calibrated review or grammar decoding yet.',
    ]
    H.append('<h2 class="pb">8. How to improve this version</h2><ol>' + ''.join(f'<li>{x}</li>' for x in imp) + '</ol>')
    H.append('<h2>9. What we can do next</h2><ul>' + ''.join(f'<li>{x}</li>' for x in nxt) + '</ul>')
    lim = ['No new training: if any weight is retrained, re-run the selection scripts in <code>BanglaLPR_System/eval/</code> so the choices stay validation-based.',
           'The whole-photo test is small (204 photos, one plate each); its BLPD day photos come from the same streets as some training photos (no same plate), so expect lower accuracy in new cities.',
           'No real traffic video exists in the dataset; the video mode was checked on a synthetic clip made from still test photos.',
           'Speed was measured on a desktop GPU and a desktop CPU, not on an edge board or a phone.',
           'Two-line rear plates only; the new single-line motorcycle front plates are not in the data.']
    H.append('<h2>10. Limitations</h2><ul>' + ''.join(f'<li>{x}</li>' for x in lim) + '</ul>')
    H.append('<h2>Papers named in this document</h2>' + table(['Short name', 'Full title'],
             [[E(PAPERS[c][0]), E(PAPERS[c][1])] for c in papers + (['p24'] if v == 'v2' else [])]))
    H.append('</body></html>')
    page = os.path.join(d, 'results', f'explained_{v}.html')
    open(page, 'w', encoding='utf-8').write('\n'.join(H))
    pdf = os.path.join(d, f'BanglaLPR_{v.upper()}_Explained.pdf')
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={pdf}',
                    'file:///' + page.replace('\\', '/')], check=True, capture_output=True, timeout=120)
    print('wrote', pdf, os.path.getsize(pdf) // 1024, 'KB')


if __name__ == '__main__':
    for v in (sys.argv[1:] or ['v1', 'v2']):
        build(v)
