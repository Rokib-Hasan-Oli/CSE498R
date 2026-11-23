"""Collect how every weight file used by V1 / V2 was trained, straight from its Kaggle run's own files.

  python tools/training_facts.py

Reads results.json (+ environment.json) inside each paper run's zip and writes tools/training_facts.json:
member, run date (UTC, from environment.json), dataset version, model, epochs, image size, batch, training minutes,
hardware, and the run's own test score. Nothing is typed by hand except the zip locations and the member names.
"""
import glob
import json
import os
import zipfile

D = 'D:/498R/docs'
G = 'D:/Github/CSE498R/weights'


def one(pattern):
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


SOURCES = {
    # weight file : (member, zip, results.json path inside, environment.json path inside)
    'p18_v8n_576.pt': ('Md Rakibul Hasan', f'{D}/Md Rakibul Hasan/18_*/After train/paper18_outputs.zip',
                       'runs/v8n_576/results.json', 'environment.json'),
    'p18_slot.pt': ('Md Rakibul Hasan', f'{D}/Md Rakibul Hasan/18_*/After train/paper18_outputs.zip',
                    'runs/slot_ocr/results.json', 'environment.json'),
    'p11_chars_v8n.pt': ('Md Rakibul Hasan', f'{D}/Md Rakibul Hasan/11_*/After Train/paper11_outputs.zip',
                         'runs/chars_v8n/results.json', 'environment.json'),
    'p02_lpnr_v12n.pt': ('MD. Rokib Hasan Oli', f'{D}/Rokib Hasna/2_*/After train/for_drive/paper02_outputs.zip',
                         'runs/lpnr_v12n/results.json', 'environment.json'),
    'p20_crnn_hr.pt': ('Md. Yousuf', f'{D}/Md Yousuf/20_*/After Train/paper20_outputs.zip',
                       'runs/crnn_hr/results.json', 'environment.json'),
    'p08_cnn28.pt': ('MD. Rokib Hasan Oli', f'{D}/Rokib Hasna/8_*/After Train/github_upload/weights/paper08/*/paper08_results.zip',
                     'runs/cnn28/results.json', 'environment.json'),
    'p10_char_cnn32.pt': ('Tashfia Adiba Zaman Nizhum', f'{D}/Tashfia Adiba Zaman Nizhum/10_*/After Train/results.zip',
                          'paper10/char_cnn_32/settings.json', 'paper10/environment.json'),
    'p06_det_v8s.pt': ('MD. Rokib Hasan Oli', f'{D}/Rokib Hasna/6_*/After Train/github_upload/weights/paper06/*/paper06_results.zip',
                       'runs/det_v8s/results.json', 'environment.json'),
    'p27_chars_v8n.pt': ('Md. Yousuf', f'{G}/paper27/Md.Yousuf/paper27_results.zip',
                         'runs/chars_v8n/results.json', 'environment.json'),
    'p34_svtrv2_s.onnx': ('MD. Rokib Hasan Oli', f'{D}/Rokib Hasna/new5/34_*/After Train/github_upload/weights/paper34/*/paper34_results.zip',
                          'runs/svtrv2_s_msr/results.json', 'environment.json'),
    'p34_svtrv2_t.onnx': ('MD. Rokib Hasan Oli', f'{D}/Rokib Hasna/new5/34_*/After Train/github_upload/weights/paper34/*/paper34_results.zip',
                          'runs/svtrv2_t_msr/results.json', 'environment.json'),
}


def find(zf, name):
    names = zf.namelist()
    if name in names:
        return name
    tail = [n for n in names if n.endswith(name)]
    return tail[0] if tail else None


def pick(d, *keys):
    for k in keys:
        cur = d
        for part in k.split('.'):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            return cur
    return None


out = {}
for w, (member, zpat, rj, ej) in SOURCES.items():
    z = one(zpat)
    with zipfile.ZipFile(z) as zf:
        r = json.loads(zf.read(find(zf, rj)).decode('utf-8'))
        e = json.loads(zf.read(find(zf, ej)).decode('utf-8')) if find(zf, ej) else {}
    with zipfile.ZipFile(z) as zf:
        info = zf.getinfo(find(zf, rj))
        zdate = '%04d-%02d-%02d %02d:%02d (file time in zip)' % info.date_time[:5]
        man = find(zf, 'dataset_manifest.json')
        man_ver = json.loads(zf.read(man).decode('utf-8')).get('version') if man else None
    s = r.get('settings', r)
    ver = e.get('dataset_version') or pick(r, 'dataset_version') or man_ver
    if not ver:      # kits built before the v1.1 rebuild (21 Oct 2026) did not write the field: those runs used v1.0
        ver = '1.0 (older kit without the dataset_version field)'
    f = {
        'member': member,
        'source_zip': os.path.relpath(z, 'D:/498R').replace('\\', '/'),
        'run_date_utc': e.get('date') or e.get('time') or e.get('started') or zdate,
        'dataset_version': ver,
        'gpu': e.get('gpu') or e.get('gpus') or e.get('device') or r.get('device'),
        'start_weights': pick(s, 'start_weights', 'model', 'weights', 'arch'),
        'epochs': pick(s, 'epochs', 'epochs_planned'),
        'epochs_run': pick(r, 'epochs_run', 'epochs_done', 'best_epoch'),
        'imgsz': pick(s, 'imgsz', 'input', 'input_HW', 'size') or (32 if 'char_cnn_32' in rj else None),
        'batch': pick(s, 'batch', 'batch_size'),
        'optimizer': pick(s, 'optimizer'),
        'train_minutes': pick(r, 'train_minutes', 'minutes', 'train_min', 'training_minutes'),
        'run_test_score': pick(r, 'ocr_test.rows_decoder.plate_exact_%', 'test_boxes.mAP50', 'test.mAP50',
                               'test_metrics.mAP50', 'test_acc_best_val_%', 'test_acc', 'test_plate_exact_%'),
    }
    out[w] = f
    print(w, f)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_facts.json'), 'w',
                    encoding='utf-8'), indent=1, ensure_ascii=False)
