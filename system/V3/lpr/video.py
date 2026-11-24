"""Video / camera stream: track each plate across frames and give ONE reading per vehicle (paper #11 idea).

ByteTrack (Ultralytics) follows every plate box; each track is read on a few frames only: reading stops once the
track is "locked" (LOCK identical layout-valid readings) or after MAX_READS reads, so a car seen for 3 s costs
3-8 reads instead of 90. The track's plate = majority of its layout-valid readings (ties -> higher mean confidence).
"""
import collections
import time

import cv2

from . import fusion, metrics, vocab
from .pipeline import crop, draw, to_english

LOCK, MAX_READS, MIN_H = 3, 8, 14


def vote(reads):
    """reads: [(text, conf, status)] -> (text, agree_count, n_reads)."""
    valid = [r for r in reads if metrics.is_valid_plate(r[0])] or reads
    groups = collections.defaultdict(list)
    for t, c, s in valid:
        groups[vocab.norm_text(t)].append((t, c))
    best = max(groups.values(), key=lambda g: (len(g), sum(c for _, c in g) / len(g)))
    return best[0][0], len(best), len(reads)


def process_video(system, path, out_path=None, max_frames=None, progress=None, night=False):
    """night: night mode of the app (detector confidence and reading rule from system.night_cfg)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    writer = None
    tracks = {}
    det = system.detector
    i, t0, reads_done = 0, time.perf_counter(), 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        with system._lock:
            r = det.model.track(frame, imgsz=det.imgsz, conf=system.night_cfg['det_conf'] if night else det.conf,
                                device=det.device, persist=True,
                                tracker='bytetrack.yaml', verbose=False)[0]
        shown = []
        if r.boxes is not None and r.boxes.id is not None:
            todo = []
            for b, c, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.id.cpu().numpy()):
                tid = int(tid)
                tr = tracks.setdefault(tid, {'reads': [], 'first': i, 'locked': False, 'best': None, 'best_h': 0})
                tr['last'] = i
                cimg = crop(frame, b)
                if cimg.size and cimg.shape[0] > tr['best_h']:
                    tr['best'], tr['best_h'] = cimg, cimg.shape[0]
                if not tr['locked'] and len(tr['reads']) < MAX_READS and cimg.size and cimg.shape[0] >= MIN_H:
                    todo.append((tid, cimg))
                shown.append((tid, b, float(c)))
            if todo:
                with system._lock:
                    res = system.read_crops([c for _, c in todo], night=night)
                reads_done += len(todo)
                for (tid, _), rd in zip(todo, res):
                    tr = tracks[tid]
                    tr['reads'].append((rd['text'], rd['conf'], rd['status']))
                    txt, agree, _ = vote(tr['reads'])
                    tr['locked'] = metrics.is_valid_plate(txt) and agree >= LOCK
        if out_path:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            plates = []
            for tid, b, c in shown:
                tr = tracks[tid]
                txt = vote(tr['reads'])[0] if tr['reads'] else ''
                plates.append({'box': b.tolist(), 'text_en': f'#{tid} ' + to_english(txt),
                               'status': 'verified' if tr['locked'] else 'likely' if txt else 'review'})
            writer.write(draw(frame, {'plates': plates}))
        i += 1
        if progress and i % 10 == 0:
            progress(i, total)
    cap.release()
    if writer:
        writer.release()
    secs = time.perf_counter() - t0
    events = []
    for tid, tr in sorted(tracks.items()):
        if not tr['reads']:
            continue
        text, agree, n = vote(tr['reads'])
        st = 'verified' if metrics.is_valid_plate(text) and agree >= 2 else 'likely' if metrics.is_valid_plate(text) else 'review'
        events.append({'track_id': tid, 'text': text, 'text_en': to_english(text), 'status': st, 'agree': agree,
                       'reads': n, 'frames': tr['last'] - tr['first'] + 1, 'first_s': round(tr['first'] / fps, 2),
                       'last_s': round(tr['last'] / fps, 2), '_crop': tr['best'],
                       'all_reads': [x[0] for x in tr['reads']]})
    return {'frames': i, 'fps_video': round(fps, 2), 'seconds': round(secs, 2), 'night_mode': bool(night),
            'processing_fps': round(i / max(secs, 1e-6), 2), 'reads': reads_done, 'tracks': len(tracks),
            'events': events}
