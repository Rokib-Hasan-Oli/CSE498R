package bd.nsu.banglalpr.core;

/**
 * An 8-bit RGB image (interleaved R,G,B) plus the exact resize / colour steps the Python system uses, so the phone
 * feeds the models the same pixels:
 *  - resizeLinear: OpenCV INTER_LINEAR (fixed-point, half-pixel centres)      -> YOLO letterbox, character CNN 28x28
 *  - resizePil:    Pillow resize (BICUBIC / BILINEAR with antialias support) -> SVTRv2 input, BLPnet CNN 32x32
 *  - grayAreaResize: OpenCV BGR2GRAY + INTER_AREA                            -> CRNN / slot OCR input
 *  - resizeCubic:  OpenCV INTER_CUBIC (x2 for tiny plates)
 *  - clahe:        OpenCV CLAHE(clip 3, 8x8 tiles) on the L channel of Lab   -> V2 low-light retry
 * Pure Java, no Android classes: the same code runs in the app and in the desktop parity test.
 */
public final class Img {
    public final int w, h;
    public final byte[] px;            // length w * h * 3, RGB

    public Img(int w, int h, byte[] px) {
        this.w = w; this.h = h; this.px = px;
    }

    public static Img fromArgb(int[] argb, int w, int h) {
        byte[] p = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++) {
            int c = argb[i];
            p[3 * i] = (byte) (c >> 16); p[3 * i + 1] = (byte) (c >> 8); p[3 * i + 2] = (byte) c;
        }
        return new Img(w, h, p);
    }

    public int[] toArgb() {
        int[] a = new int[w * h];
        for (int i = 0; i < w * h; i++)
            a[i] = 0xff000000 | ((px[3 * i] & 255) << 16) | ((px[3 * i + 1] & 255) << 8) | (px[3 * i + 2] & 255);
        return a;
    }

    int at(int x, int y, int c) { return px[(y * w + x) * 3 + c] & 255; }

    /** pixels [x1, x2) x [y1, y2), clipped to the image */
    public Img crop(int x1, int y1, int x2, int y2) {
        x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(w, x2); y2 = Math.min(h, y2);
        int nw = Math.max(0, x2 - x1), nh = Math.max(0, y2 - y1);
        byte[] p = new byte[nw * nh * 3];
        for (int y = 0; y < nh; y++) System.arraycopy(px, ((y + y1) * w + x1) * 3, p, y * nw * 3, nw * 3);
        return new Img(nw, nh, p);
    }

    public Img pad(int top, int bottom, int left, int right, int value) {
        int nw = w + left + right, nh = h + top + bottom;
        byte[] p = new byte[nw * nh * 3];
        java.util.Arrays.fill(p, (byte) value);
        for (int y = 0; y < h; y++) System.arraycopy(px, y * w * 3, p, ((y + top) * nw + left) * 3, w * 3);
        return new Img(nw, nh, p);
    }

    // ------------------------------------------------------------------------------------------ OpenCV INTER_LINEAR
    private static void linTab(int ssize, int dsize, int[] i0, int[] i1, short[] a0, short[] a1) {
        double scale = (double) ssize / dsize;
        for (int d = 0; d < dsize; d++) {
            double f = (d + 0.5) * scale - 0.5;
            int s = (int) Math.floor(f);
            f -= s;
            if (s < 0) { f = 0; s = 0; }
            if (s >= ssize - 1) { f = 0; s = ssize - 1; }
            i0[d] = s; i1[d] = Math.min(s + 1, ssize - 1);
            a0[d] = (short) Math.rint((1 - f) * 2048); a1[d] = (short) Math.rint(f * 2048);
        }
    }

    public Img resizeLinear(int nw, int nh) {
        if (nw == w && nh == h) return this;
        int[] xi0 = new int[nw], xi1 = new int[nw], yi0 = new int[nh], yi1 = new int[nh];
        short[] xa0 = new short[nw], xa1 = new short[nw], ya0 = new short[nh], ya1 = new short[nh];
        linTab(w, nw, xi0, xi1, xa0, xa1);
        linTab(h, nh, yi0, yi1, ya0, ya1);
        int[][] rows = new int[h][];
        byte[] out = new byte[nw * nh * 3];
        for (int y = 0; y < nh; y++) {
            int[] r0 = hrow(rows, yi0[y], nw, xi0, xi1, xa0, xa1), r1 = hrow(rows, yi1[y], nw, xi0, xi1, xa0, xa1);
            int b0 = ya0[y], b1 = ya1[y];
            for (int i = 0; i < nw * 3; i++) {
                int v = (((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16) + 2) >> 2;
                out[y * nw * 3 + i] = (byte) Math.max(0, Math.min(255, v));
            }
        }
        return new Img(nw, nh, out);
    }

    private int[] hrow(int[][] cache, int sy, int nw, int[] i0, int[] i1, short[] a0, short[] a1) {
        if (cache[sy] != null) return cache[sy];
        int[] r = new int[nw * 3];
        int base = sy * w * 3;
        for (int x = 0; x < nw; x++)
            for (int c = 0; c < 3; c++)
                r[x * 3 + c] = (px[base + i0[x] * 3 + c] & 255) * a0[x] + (px[base + i1[x] * 3 + c] & 255) * a1[x];
        cache[sy] = r;
        return r;
    }

    // ------------------------------------------------------------------------------------------ Pillow resize
    private interface Filter { double f(double x); }
    private static final Filter BICUBIC = x -> {
        double a = -0.5;
        x = Math.abs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0;
    };
    private static final Filter BILINEAR = x -> { x = Math.abs(x); return x < 1 ? 1 - x : 0; };
    private static final int PREC = 22;

    /** Pillow precompute_coeffs + normalize_coeffs_8bpc: fixed-point weights per output pixel */
    private static int[][] pilCoeffs(int inSize, int outSize, Filter flt, double support0, int[] bounds) {
        double scale = (double) inSize / outSize, fscale = Math.max(scale, 1.0), support = support0 * fscale;
        int[][] k = new int[outSize][];
        for (int xx = 0; xx < outSize; xx++) {
            double center = (xx + 0.5) * scale, ss = 1.0 / fscale;
            int xmin = Math.max((int) (center - support + 0.5), 0);
            int xmax = Math.min((int) (center + support + 0.5), inSize) - xmin;
            double[] w = new double[xmax];
            double ww = 0;
            for (int x = 0; x < xmax; x++) { w[x] = flt.f((x + xmin - center + 0.5) * ss); ww += w[x]; }
            int[] kk = new int[xmax];
            for (int x = 0; x < xmax; x++) {
                double v = ww != 0 ? w[x] / ww : 0;
                kk[x] = (int) (v < 0 ? v * (1 << PREC) - 0.5 : v * (1 << PREC) + 0.5);
            }
            k[xx] = kk;
            bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
        }
        return k;
    }

    private static int clip8(long v) {
        long r = v >> PREC;
        return (int) (r < 0 ? 0 : r > 255 ? 255 : r);
    }

    public Img resizePil(int nw, int nh, boolean bicubic) {
        Filter flt = bicubic ? BICUBIC : BILINEAR;
        double sup = bicubic ? 2.0 : 1.0;
        Img cur = this;
        if (nw != w) {                                           // horizontal pass
            int[] b = new int[2 * nw];
            int[][] k = pilCoeffs(w, nw, flt, sup, b);
            byte[] o = new byte[nw * h * 3];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < nw; x++)
                    for (int c = 0; c < 3; c++) {
                        long s = 1L << (PREC - 1);
                        int[] kk = k[x];
                        for (int i = 0; i < kk.length; i++) s += (long) (px[(y * w + b[2 * x] + i) * 3 + c] & 255) * kk[i];
                        o[(y * nw + x) * 3 + c] = (byte) clip8(s);
                    }
            cur = new Img(nw, h, o);
        }
        if (nh != h) {                                           // vertical pass
            int[] b = new int[2 * nh];
            int[][] k = pilCoeffs(h, nh, flt, sup, b);
            byte[] o = new byte[cur.w * nh * 3];
            for (int y = 0; y < nh; y++)
                for (int x = 0; x < cur.w; x++)
                    for (int c = 0; c < 3; c++) {
                        long s = 1L << (PREC - 1);
                        int[] kk = k[y];
                        for (int i = 0; i < kk.length; i++) s += (long) (cur.px[((b[2 * y] + i) * cur.w + x) * 3 + c] & 255) * kk[i];
                        o[(y * cur.w + x) * 3 + c] = (byte) clip8(s);
                    }
            cur = new Img(cur.w, nh, o);
        }
        return cur;
    }

    // ------------------------------------------------------------------------------------------ OpenCV INTER_CUBIC
    public Img resizeCubic(int nw, int nh) {
        byte[] o = new byte[nw * nh * 3];
        double sx = (double) w / nw, sy = (double) h / nh;
        for (int y = 0; y < nh; y++) {
            double fy = (y + 0.5) * sy - 0.5;
            int iy = (int) Math.floor(fy);
            double ty = fy - iy;
            double[] wy = cubicW(ty);
            for (int x = 0; x < nw; x++) {
                double fx = (x + 0.5) * sx - 0.5;
                int ix = (int) Math.floor(fx);
                double[] wx = cubicW(fx - ix);
                for (int c = 0; c < 3; c++) {
                    double s = 0;
                    for (int j = 0; j < 4; j++) {
                        int yy = Math.min(h - 1, Math.max(0, iy - 1 + j));
                        for (int i = 0; i < 4; i++) {
                            int xx = Math.min(w - 1, Math.max(0, ix - 1 + i));
                            s += wy[j] * wx[i] * at(xx, yy, c);
                        }
                    }
                    o[(y * nw + x) * 3 + c] = (byte) Math.max(0, Math.min(255, Math.rint(s)));
                }
            }
        }
        return new Img(nw, nh, o);
    }

    private static double[] cubicW(double t) {
        double a = -0.75;
        double[] r = new double[4];
        double[] d = {1 + t, t, 1 - t, 2 - t};
        for (int i = 0; i < 4; i++) {
            double x = d[i];
            r[i] = x <= 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
        }
        return r;
    }

    // ------------------------------------------------------------------------------------------ grey + INTER_AREA
    /** OpenCV COLOR_BGR2GRAY on our RGB pixels (fixed point, same coefficients) */
    public byte[] gray() {
        byte[] g = new byte[w * h];
        for (int i = 0; i < w * h; i++) {
            int r = px[3 * i] & 255, gg = px[3 * i + 1] & 255, b = px[3 * i + 2] & 255;
            g[i] = (byte) ((r * 4899 + gg * 9617 + b * 1868 + 8192) >> 14);
        }
        return g;
    }

    /** OpenCV INTER_LINEAR for a grey image */
    public static byte[] grayLinear(byte[] g, int w, int h, int nw, int nh) {
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = g[i];
        Img r = new Img(w, h, rgb).resizeLinear(nw, nh);
        byte[] o = new byte[nw * nh];
        for (int i = 0; i < nw * nh; i++) o[i] = r.px[3 * i];
        return o;
    }

    /** OpenCV INTER_AREA for a grey image (exact area weights when shrinking; linear when enlarging) */
    public static byte[] grayResize(byte[] g, int w, int h, int nw, int nh) {
        if (nw > w || nh > h) return grayLinear(g, w, h, nw, nh);   // enlarging: OpenCV area ~ linear
        double[][] tx = areaTab(w, nw), ty = areaTab(h, nh);
        double[] tmp = new double[nw * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < nw; x++) {
                double s = 0;
                for (int i = 0; i < tx[x].length; i += 2) s += (g[y * w + (int) tx[x][i]] & 255) * tx[x][i + 1];
                tmp[y * nw + x] = s;
            }
        byte[] o = new byte[nw * nh];
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++) {
                double s = 0;
                for (int i = 0; i < ty[y].length; i += 2) s += tmp[(int) ty[y][i] * nw + x] * ty[y][i + 1];
                o[y * nw + x] = (byte) Math.max(0, Math.min(255, Math.rint(s)));
            }
        return o;
    }

    private static double[][] areaTab(int ssize, int dsize) {
        double scale = (double) ssize / dsize;
        double[][] t = new double[dsize][];
        for (int d = 0; d < dsize; d++) {
            java.util.ArrayList<Double> l = new java.util.ArrayList<>();
            double fsx1 = d * scale, fsx2 = fsx1 + scale, cell = Math.min(scale, ssize - fsx1);
            int sx1 = (int) Math.ceil(fsx1), sx2 = (int) Math.floor(fsx2);
            sx2 = Math.min(sx2, ssize - 1);
            sx1 = Math.min(sx1, sx2);
            if (sx1 - fsx1 > 1e-3) { l.add((double) (sx1 - 1)); l.add((sx1 - fsx1) / cell); }
            for (int s = sx1; s < sx2; s++) { l.add((double) s); l.add(1.0 / cell); }
            if (fsx2 - sx2 > 1e-3) { l.add((double) sx2); l.add(Math.min(Math.min(fsx2 - sx2, 1.0), cell) / cell); }
            double[] a = new double[l.size()];
            for (int i = 0; i < a.length; i++) a[i] = l.get(i);
            t[d] = a;
        }
        return t;
    }

    // ------------------------------------------------------------------------------------------ CLAHE on Lab L
    private static double srgb2lin(double c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
    private static double lin2srgb(double c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; }
    private static double fLab(double t) { return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16.0 / 116; }
    private static double fInv(double t) { double t3 = t * t * t; return t3 > 0.008856 ? t3 : (t - 16.0 / 116) / 7.787; }

    public Img clahe() {
        int n = w * h;
        double[] L = new double[n], A = new double[n], B = new double[n];
        byte[] l8 = new byte[n];
        for (int i = 0; i < n; i++) {
            double r = srgb2lin((px[3 * i] & 255) / 255.0), g = srgb2lin((px[3 * i + 1] & 255) / 255.0), b = srgb2lin((px[3 * i + 2] & 255) / 255.0);
            double X = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456;
            double Y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
            double Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
            double fx = fLab(X), fy = fLab(Y), fz = fLab(Z);
            L[i] = Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y;
            A[i] = 500 * (fx - fy); B[i] = 200 * (fy - fz);
            l8[i] = (byte) Math.max(0, Math.min(255, Math.rint(L[i] * 255 / 100)));
        }
        byte[] eq = claheGray(l8, w, h, 3.0, 8, 8);
        byte[] o = new byte[n * 3];
        for (int i = 0; i < n; i++) {
            double Ln = (eq[i] & 255) * 100.0 / 255;
            double fy = (Ln + 16) / 116, fx = fy + A[i] / 500, fz = fy - B[i] / 200;
            double X = fInv(fx) * 0.950456, Y = Ln > 7.9996 ? fy * fy * fy : Ln / 903.3, Z = fInv(fz) * 1.088754;
            double r = 3.240479 * X - 1.537150 * Y - 0.498535 * Z;
            double g = -0.969256 * X + 1.875992 * Y + 0.041556 * Z;
            double b = 0.055648 * X - 0.204043 * Y + 1.057311 * Z;
            o[3 * i] = (byte) Math.max(0, Math.min(255, Math.rint(lin2srgb(Math.max(0, r)) * 255)));
            o[3 * i + 1] = (byte) Math.max(0, Math.min(255, Math.rint(lin2srgb(Math.max(0, g)) * 255)));
            o[3 * i + 2] = (byte) Math.max(0, Math.min(255, Math.rint(lin2srgb(Math.max(0, b)) * 255)));
        }
        return new Img(w, h, o);
    }

    /** OpenCV CLAHE: tiles, clipped histograms with even redistribution, bilinear blend of tile LUTs */
    static byte[] claheGray(byte[] src, int w, int h, double clipLimit, int tx, int ty) {
        int pw = (w + tx - 1) / tx * tx, ph = (h + ty - 1) / ty * ty;   // pad with reflect-101 to whole tiles
        int tw = pw / tx, th = ph / ty, area = tw * th;
        int clip = Math.max((int) (clipLimit * area / 256), 1);
        int[][][] lut = new int[ty][tx][256];
        for (int j = 0; j < ty; j++)
            for (int i = 0; i < tx; i++) {
                int[] hist = new int[256];
                for (int y = j * th; y < (j + 1) * th; y++)
                    for (int x = i * tw; x < (i + 1) * tw; x++)
                        hist[src[refl(y, h) * w + refl(x, w)] & 255]++;
                int excess = 0;
                for (int k = 0; k < 256; k++) if (hist[k] > clip) { excess += hist[k] - clip; hist[k] = clip; }
                int batch = excess / 256, resid = excess - batch * 256;
                for (int k = 0; k < 256; k++) hist[k] += batch;
                if (resid > 0) {
                    int step = Math.max(256 / resid, 1);
                    for (int k = 0; k < 256 && resid > 0; k += step) { hist[k]++; resid--; }
                }
                double scale = 255.0 / area;
                int sum = 0;
                for (int k = 0; k < 256; k++) { sum += hist[k]; lut[j][i][k] = (int) Math.max(0, Math.min(255, Math.rint(sum * scale))); }
            }
        byte[] o = new byte[w * h];
        double itw = 1.0 / tw, ith = 1.0 / th;
        for (int y = 0; y < h; y++) {
            double tyf = y * ith - 0.5;
            int ty1 = (int) Math.floor(tyf), ty2 = ty1 + 1;
            double ya = tyf - ty1, ya1 = 1 - ya;
            ty1 = Math.max(ty1, 0); ty2 = Math.min(ty2, ty - 1);
            for (int x = 0; x < w; x++) {
                double txf = x * itw - 0.5;
                int tx1 = (int) Math.floor(txf), tx2 = tx1 + 1;
                double xa = txf - tx1, xa1 = 1 - xa;
                tx1 = Math.max(tx1, 0); tx2 = Math.min(tx2, tx - 1);
                int v = src[y * w + x] & 255;
                double r = (lut[ty1][tx1][v] * xa1 + lut[ty1][tx2][v] * xa) * ya1 + (lut[ty2][tx1][v] * xa1 + lut[ty2][tx2][v] * xa) * ya;
                o[y * w + x] = (byte) Math.max(0, Math.min(255, Math.rint(r)));
            }
        }
        return o;
    }

    private static int refl(int p, int n) {
        if (n == 1) return 0;
        while (p < 0 || p >= n) p = p < 0 ? -p : 2 * n - 2 - p;
        return p;
    }
}
