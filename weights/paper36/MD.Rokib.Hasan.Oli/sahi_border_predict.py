"""Plate boxes with SAHI + full image and the cut-off-box filter of the paper #36 study (val-best setting).

  pip install ultralytics==8.4.170 sahi==0.12.8
  python sahi_border_predict.py best_det_v8s_sf.pt photo.jpg [more photos]      (prints conf x1 y1 x2 y2 per plate)

640 x 640 slices with 25 % overlap + one pass on the whole photo; a slice box that touches (within 2 px) an edge of
its slice that is not an edge of the photo is dropped (its plate is whole in a neighbouring slice or in the full
photo); the rest is merged with NMS at IoU 0.5 (the paper's rule). Same boxes as `sahi_common.merge(rec, 'SAHI_FI',
'BORDER')` in report/MD.Rokib.Hasan.Oli/paper36/source/.
"""
import sys

import numpy as np


def load(weights, device='cuda:0' if __import__('torch').cuda.is_available() else 'cpu', conf=0.25):
    from sahi import AutoDetectionModel
    return AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path=weights, confidence_threshold=conf,
                                              device=device, image_size=640)


def predict(model, image, size=640, overlap=0.25, iou=0.5, tol=2):
    """image: path or RGB array. Returns an (n, 5) array [conf, x1, y1, x2, y2], best first."""
    from sahi.postprocess.combine import nms
    from sahi.predict import get_prediction
    from sahi.slicing import slice_image
    from sahi.utils.cv import read_image_as_pil
    arr = np.asarray(read_image_as_pil(image))
    h, w = arr.shape[:2]
    rows = [[o.score.value, *o.bbox.to_xyxy()] for o in get_prediction(arr, model).object_prediction_list]
    sr = slice_image(image=arr, slice_height=size, slice_width=size, overlap_height_ratio=overlap,
                     overlap_width_ratio=overlap, auto_slice_resolution=True)
    sl = []
    for img, (sx, sy) in zip(sr.images, sr.starting_pixels):
        x1s, y1s, x2s, y2s = sx, sy, sx + img.shape[1], sy + img.shape[0]
        model.perform_batch_inference([np.ascontiguousarray(img)])
        model.convert_original_predictions(shift_amount=[[sx, sy]], full_shape=[[h, w]])
        for o in model.object_prediction_list_per_image[0]:
            b = o.get_shifted_object_prediction().bbox.to_xyxy()
            cut = ((b[0] - x1s <= tol and x1s > 0) or (b[1] - y1s <= tol and y1s > 0) or
                   (x2s - b[2] <= tol and x2s < w) or (y2s - b[3] <= tol and y2s < h))
            if not cut:
                sl.append([o.score.value, *b])
    allb = np.array(sl + rows if len(sr) > 1 else sl, dtype=float).reshape(-1, 5)
    if len(allb) > 1:
        a = np.concatenate([allb[:, 1:5], allb[:, :1], np.zeros((len(allb), 1))], 1)
        allb = allb[nms(a, match_metric='IOU', match_threshold=iou)]
    return allb[np.argsort(-allb[:, 0], kind='stable')]


if __name__ == '__main__':
    m = load(sys.argv[1])
    for f in sys.argv[2:]:
        for b in predict(m, f):
            print(f, ' '.join(f'{v:.3f}' if i == 0 else f'{v:.1f}' for i, v in enumerate(b)))
