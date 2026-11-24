package bd.nsu.banglalpr.core;

import java.util.ArrayList;
import java.util.List;

/** The plate readers, mirroring lpr/readers.py: char-YOLO, CRNN (CTC), slot OCR, SVTRv2, and the character CNNs. */
public final class Readers {
    private Readers() {}

    public static final class Reading {
        public final String text; public final float conf; public final boolean valid;
        public final List<Vocab.Tok> boxes;
        Reading(String text, float conf, List<Vocab.Tok> boxes) {
            this.text = text; this.conf = conf; this.valid = Vocab.isValidPlate(text); this.boxes = boxes;
        }
    }

    public interface Reader { String name(); Reading read(Img crop) throws Exception; }

    /** character detector + BRTA decoder (rec.decode: boxes conf >= 0.25 after NMS at conf 0.10) */
    public static final class CharYolo implements Reader {
        final Yolo y; final String name;
        public CharYolo(Net.Source s, String name, String file, int imgsz) throws Exception {
            y = new Yolo(s, file, imgsz, false); this.name = name;
        }
        public String name() { return name; }
        public Reading read(Img crop) throws Exception {
            List<Vocab.Tok> toks = new ArrayList<>();
            float sum = 0;
            for (Yolo.Box b : y.predict(crop, 0.10f)) {
                if (b.conf < 0.25f) continue;
                toks.add(new Vocab.Tok(Vocab.CLASSES[b.cls], b.x1, b.y1, b.x2, b.y2, b.conf));
                sum += b.conf;
            }
            String[] t = Vocab.plateText(toks);
            return new Reading(t[0], toks.isEmpty() ? 0 : sum / toks.size(), toks);
        }
    }

    /** seq.prep: grey, two-line plates (w/h < 2.6) cut and placed side by side, aspect-kept INTER_AREA, pad 127 */
    static float[] seqPrep(Img img, int H, int W, boolean stack) {
        byte[] g = img.gray();
        int w = img.w, h = img.h;
        if (stack && (double) w / Math.max(1, h) < 2.6) {
            int th = h / 2, bh = h - th;
            byte[] bot = new byte[w * bh];
            System.arraycopy(g, th * w, bot, 0, w * bh);
            byte[] botR = Img.grayLinear(bot, w, bh, w, th);     // cv2.resize(bot, (w, top.shape[0])): INTER_LINEAR
            byte[] s2 = new byte[2 * w * th];
            for (int y = 0; y < th; y++) {
                System.arraycopy(g, y * w, s2, y * 2 * w, w);
                System.arraycopy(botR, y * w, s2, y * 2 * w + w, w);
            }
            g = s2; w = 2 * w; h = th;
        }
        double s = Math.min((double) H / h, (double) W / w);
        int nh = Math.max(1, (int) (h * s)), nw = Math.max(1, (int) (w * s));
        byte[] r = Img.grayResize(g, w, h, nw, nh);
        float[] x = new float[H * W];
        java.util.Arrays.fill(x, (127 / 255f - 0.5f) / 0.5f);
        int y0 = (H - nh) / 2, x0 = (W - nw) / 2;
        for (int y = 0; y < nh; y++)
            for (int xx = 0; xx < nw; xx++) x[(y + y0) * W + xx + x0] = ((r[y * nw + xx] & 255) / 255f - 0.5f) / 0.5f;
        return x;
    }

    static float[][] softmaxRows(float[] logits, int T, int C) {
        float[][] p = new float[T][C];
        for (int t = 0; t < T; t++) {
            float mx = -Float.MAX_VALUE;
            for (int c = 0; c < C; c++) mx = Math.max(mx, logits[t * C + c]);
            float s = 0;
            for (int c = 0; c < C; c++) { p[t][c] = (float) Math.exp(logits[t * C + c] - mx); s += p[t][c]; }
            for (int c = 0; c < C; c++) p[t][c] /= s;
        }
        return p;
    }

    /** CRNN-CTC: greedy decode, confidence = mean max-prob of kept tokens */
    public static final class Ctc implements Reader {
        final Net n; final String name; final int H, W; final boolean stack;
        public Ctc(Net.Source s, String name, String file, int H, int W, boolean stack) throws Exception {
            n = new Net(s, file); this.name = name; this.H = H; this.W = W; this.stack = stack;
        }
        public String name() { return name; }
        public Reading read(Img crop) throws Exception {
            long[][] os = new long[1][];
            float[] y = n.run(seqPrep(crop, H, W, stack), new long[]{1, 1, H, W}, os);
            int T = (int) os[0][1], C = (int) os[0][2];
            float[][] p = softmaxRows(y, T, C);
            List<String> toks = new ArrayList<>();
            float cs = 0;
            int prev = 0, kept = 0;
            for (int t = 0; t < T; t++) {
                int b = argmax(p[t]);
                if (b != 0 && b != prev && b <= Vocab.CLASSES.length) { toks.add(Vocab.CLASSES[b - 1]); cs += p[t][b]; kept++; }
                prev = b;
            }
            return new Reading(Vocab.tokensToText(toks), kept == 0 ? 0 : cs / kept, null);
        }
    }

    /** slot OCR: 9 fixed slots, stop at the first empty slot (class 0) */
    public static final class Slot implements Reader {
        final Net n; final String name;
        public Slot(Net.Source s, String name, String file) throws Exception { n = new Net(s, file); this.name = name; }
        public String name() { return name; }
        public Reading read(Img crop) throws Exception {
            long[][] os = new long[1][];
            float[] y = n.run(seqPrep(crop, 64, 128, false), new long[]{1, 1, 64, 128}, os);
            int L = (int) os[0][1], C = (int) os[0][2];
            float[][] p = softmaxRows(y, L, C);
            List<String> toks = new ArrayList<>();
            float cs = 0;
            for (int i = 0; i < L; i++) {
                int b = argmax(p[i]);
                if (b == 0) break;
                toks.add(Vocab.CLASSES[b - 1]); cs += p[i][b];
            }
            return new Reading(Vocab.tokensToText(toks), toks.isEmpty() ? 0 : cs / toks.size(), null);
        }
    }

    /** SVTRv2 (MSR): bucket by aspect ratio, Pillow bicubic resize, normalise to [-1, 1]; output is softmax */
    public static final class Svtr implements Reader {
        static final int[][] MSR = {{64, 64}, {48, 96}, {40, 112}, {32, 128}};
        final Net n; final String name;
        public Svtr(Net.Source s, String name, String file) throws Exception { n = new Net(s, file); this.name = name; }
        public String name() { return name; }
        public Reading read(Img crop) throws Exception {
            double r = (double) crop.w / Math.max(1, crop.h);
            int[] hw = MSR[r < 1.5 ? 0 : r < 2.5 ? 1 : r < 3.5 ? 2 : 3];
            int H = hw[0], W = hw[1];
            Img im = crop.resizePil(W, H, true);
            float[] x = new float[3 * H * W];
            for (int i = 0; i < H * W; i++)
                for (int c = 0; c < 3; c++) x[c * H * W + i] = ((im.px[3 * i + c] & 255) / 255f - 0.5f) / 0.5f;
            long[][] os = new long[1][];
            float[] y = n.run(x, new long[]{1, 3, H, W}, os);
            int T = (int) os[0][1], C = (int) os[0][2];
            List<String> toks = new ArrayList<>();
            double prod = 1;
            int prev = 0;
            for (int t = 0; t < T; t++) {
                int b = 0;
                float bv = -1;
                for (int c = 0; c < C; c++) if (y[t * C + c] > bv) { bv = y[t * C + c]; b = c; }
                if (b != 0 && b != prev) { toks.add(Vocab.CLASSES[b - 1]); prod *= bv; }
                prev = b;
            }
            return new Reading(Vocab.tokensToText(toks), toks.isEmpty() ? 0 : (float) prod, null);
        }
    }

    static int argmax(float[] a) {
        int b = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
        return b;
    }

    /** character CNN agreement with the char-YOLO boxes (CharVerifier): p08 28x28 + 4 % pad, p10 32x32 */
    public static final class CharCnn {
        final Net n; final boolean p08;
        public CharCnn(Net.Source s, boolean p08) throws Exception {
            n = new Net(s, p08 ? "p08_cnn28.onnx" : "p10_char_cnn32.onnx"); this.p08 = p08;
        }
        public float agreement(Img crop, List<Vocab.Tok> boxes) throws Exception {
            if (boxes == null || boxes.isEmpty()) return 0;
            int S = p08 ? 28 : 32;
            double pad = p08 ? 0.04 : 0.0;
            List<float[]> xs = new ArrayList<>();
            List<String> want = new ArrayList<>();
            for (Vocab.Tok t : boxes) {
                double pw = pad * (t.x2 - t.x1), ph = pad * (t.y2 - t.y1);
                Img c = crop.crop((int) (t.x1 - pw), (int) (t.y1 - ph), (int) Math.ceil(t.x2 + pw), (int) Math.ceil(t.y2 + ph));
                if (c.w == 0 || c.h == 0) continue;
                Img r = p08 ? c.resizeLinear(S, S) : c.resizePil(S, S, true);
                float[] x = new float[3 * S * S];
                for (int i = 0; i < S * S; i++) for (int ch = 0; ch < 3; ch++) x[ch * S * S + i] = (r.px[3 * i + ch] & 255) / 255f;
                xs.add(x); want.add(t.t);
            }
            if (xs.isEmpty()) return 0;
            float[] all = new float[xs.size() * 3 * S * S];
            for (int i = 0; i < xs.size(); i++) System.arraycopy(xs.get(i), 0, all, i * 3 * S * S, 3 * S * S);
            long[][] os = new long[1][];
            float[] y = n.run(all, new long[]{xs.size(), 3, S, S}, os);
            int C = (int) os[0][os[0].length - 1];
            if (y.length != xs.size() * C) {         // an execution provider ignored the batch: one crop at a time
                y = new float[xs.size() * C];
                for (int i = 0; i < xs.size(); i++)
                    System.arraycopy(n.run(xs.get(i), new long[]{1, 3, S, S}, null), 0, y, i * C, C);
            }
            String[] cls = p08 ? Vocab.CLASSES : Vocab.P10_CLASSES;
            int ok = 0;
            for (int i = 0; i < xs.size(); i++) {
                int b = 0;
                for (int c = 1; c < C; c++) if (y[i * C + c] > y[i * C + b]) b = c;
                if (cls[b].equals(want.get(i))) ok++;
            }
            return (float) ok / xs.size();
        }
    }
}
