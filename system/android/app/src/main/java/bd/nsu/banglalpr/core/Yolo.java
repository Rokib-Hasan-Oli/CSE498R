package bd.nsu.banglalpr.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Ultralytics YOLO on ONNX, with the same pre / post-processing as model.predict:
 *  - letterbox: scale to imgsz (shortest side), pad with grey 114; "auto" = pad only to a multiple of 32 (plate
 *    detectors, as predict does for one photo), otherwise a full square (char readers, the kit's rect=False)
 *  - output (1, 4 + nc, N): cx, cy, w, h, class scores -> score > conf -> class-aware NMS IoU 0.7, max 300 boxes
 */
public final class Yolo {
    public static final class Box {
        public float x1, y1, x2, y2, conf;
        public int cls;
        Box(float x1, float y1, float x2, float y2, float conf, int cls) {
            this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2; this.conf = conf; this.cls = cls;
        }
    }

    final Net net;
    final int imgsz;
    final boolean auto;

    public Yolo(Net.Source src, String name, int imgsz, boolean auto) throws Exception {
        net = new Net(src, name); this.imgsz = imgsz; this.auto = auto;
    }

    public List<Box> predict(Img img, float conf) throws Exception {
        double r = Math.min((double) imgsz / img.h, (double) imgsz / img.w);
        int nw = (int) Math.rint(img.w * r), nh = (int) Math.rint(img.h * r);     // Python round() = half-even
        double dw = imgsz - nw, dh = imgsz - nh;
        if (auto) { dw = dw % 32; dh = dh % 32; }
        dw /= 2; dh /= 2;
        Img im = (nw != img.w || nh != img.h) ? img.resizeLinear(nw, nh) : img;
        int top = (int) Math.rint(dh - 0.1), bottom = (int) Math.rint(dh + 0.1);
        int left = (int) Math.rint(dw - 0.1), right = (int) Math.rint(dw + 0.1);
        im = im.pad(top, bottom, left, right, 114);
        int H = im.h, W = im.w, hw = H * W;
        float[] x = new float[3 * hw];
        for (int i = 0; i < hw; i++)
            for (int c = 0; c < 3; c++) x[c * hw + i] = (im.px[3 * i + c] & 255) / 255f;
        long[][] os = new long[1][];
        float[] y = net.run(x, new long[]{1, 3, H, W}, os);
        int rows = (int) os[0][1], n = (int) os[0][2], nc = rows - 4;
        List<Box> cand = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            int best = 0;
            float bs = -1;
            for (int c = 0; c < nc; c++) { float v = y[(4 + c) * n + j]; if (v > bs) { bs = v; best = c; } }
            if (bs <= conf) continue;
            float cx = y[j], cy = y[n + j], bw = y[2 * n + j], bh = y[3 * n + j];
            cand.add(new Box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, bs, best));
        }
        List<Box> keep = nms(cand, 0.7f, 300);
        for (Box b : keep) {                                  // back to the original photo (scale_boxes)
            b.x1 = clamp((float) ((b.x1 - left) / r), img.w); b.x2 = clamp((float) ((b.x2 - left) / r), img.w);
            b.y1 = clamp((float) ((b.y1 - top) / r), img.h); b.y2 = clamp((float) ((b.y2 - top) / r), img.h);
        }
        return keep;
    }

    private static float clamp(float v, int hi) { return Math.max(0, Math.min(hi, v)); }

    /** class-aware NMS like torchvision.batched_nms (boxes of different classes never suppress each other) */
    static List<Box> nms(List<Box> b, float iou, int maxDet) {
        Box[] s = b.toArray(new Box[0]);
        Arrays.sort(s, (p, q) -> Float.compare(q.conf, p.conf));
        boolean[] dead = new boolean[s.length];
        List<Box> out = new ArrayList<>();
        for (int i = 0; i < s.length && out.size() < maxDet; i++) {
            if (dead[i]) continue;
            out.add(s[i]);
            for (int j = i + 1; j < s.length; j++)
                if (!dead[j] && s[j].cls == s[i].cls && iouOf(s[i], s[j]) > iou) dead[j] = true;
        }
        return out;
    }

    static float iouOf(Box a, Box b) {
        float ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
        float iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
        float in = ix * iy;
        return in / Math.max(1e-9f, (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - in);
    }
}
