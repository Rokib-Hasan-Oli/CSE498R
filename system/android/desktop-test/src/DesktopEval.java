import bd.nsu.banglalpr.core.Img;
import bd.nsu.banglalpr.core.Net;
import bd.nsu.banglalpr.core.Pipeline;
import bd.nsu.banglalpr.core.Vocab;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parity test of the Android engine on the PC: the SAME Java classes the APK uses, run with desktop ONNX Runtime
 * on the 498R v1.1 test sets, compared plate by plate with the Python system's predictions.
 *
 *   java -cp <classes>;onnxruntime.jar DesktopEval v1|v2 <onnx dir> <dataset dir> <python results dir> [max] [night]
 *   night: V3 night mode (readCrop / process with night = true) against V3/tools/night_preds.py's files
 */
public class DesktopEval {
    static List<String[]> csv(String path) throws Exception {
        List<String[]> rows = new ArrayList<>();
        for (String line : Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)) {
            List<String> f = new ArrayList<>();
            StringBuilder b = new StringBuilder();
            boolean q = false;
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                if (c == '"') { if (q && i + 1 < line.length() && line.charAt(i + 1) == '"') { b.append('"'); i++; } else q = !q; }
                else if (c == ',' && !q) { f.add(b.toString()); b.setLength(0); }
                else b.append(c);
            }
            f.add(b.toString());
            rows.add(f.toArray(new String[0]));
        }
        return rows;
    }

    static Img load(String path) throws Exception {
        BufferedImage bi = ImageIO.read(new File(path));
        int w = bi.getWidth(), h = bi.getHeight();
        return Img.fromArgb(bi.getRGB(0, 0, w, h, null, 0, w), w, h);
    }

    static int lev(String a, String b) {
        int[] prev = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) prev[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            int[] cur = new int[b.length() + 1];
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++)
                cur[j] = Math.min(Math.min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1));
            prev = cur;
        }
        return prev[b.length()];
    }

    public static void main(String[] a) throws Exception {
        String v = a[0], onnx = a[1], data = a[2], py = a[3];
        int max = a.length > 4 ? Integer.parseInt(a[4]) : 1_000_000;
        boolean night = a.length > 5 && a[5].equals("night");
        String tag = v + (night ? "_night" : "");
        Net.threads = 8;
        Net.Source src = name -> Files.readAllBytes(Paths.get(onnx, name));
        long t0 = System.nanoTime();
        Pipeline p = new Pipeline(v, src);
        System.out.printf("loaded %s in %.1f s%s%n", v, (System.nanoTime() - t0) / 1e9, night ? " (night mode)" : "");

        // ---- A. reader part on the OCR test crops, vs the Python predictions of the same version
        Map<String, String> pyOcr = new HashMap<>();
        List<String[]> pr = csv(py + "/ocr_test_preds_" + tag + ".csv");
        for (String[] r : pr.subList(1, pr.size())) pyOcr.put(r[0], r[2]);
        List<String[]> ocr = csv(data + "/ocr/test.csv");
        int n = 0, exact = 0, same = 0, ce = 0, cl = 0;
        Map<String, int[]> st = new LinkedHashMap<>();
        long tr = 0;
        for (String[] r : ocr.subList(1, Math.min(ocr.size(), max + 1))) {
            Img im = load(data + "/ocr/images/test/" + r[0]);
            long s = System.nanoTime();
            Pipeline.Plate pl = p.readCrop(im, night);
            tr += System.nanoTime() - s;
            String pn = Vocab.normText(pl.text), tn = Vocab.normText(r[1]);
            n++; exact += pn.equals(tn) ? 1 : 0;
            same += pn.equals(Vocab.normText(pyOcr.getOrDefault(r[0], ""))) ? 1 : 0;
            ce += lev(pn, tn); cl += Math.max(1, tn.length());
            int[] c = st.computeIfAbsent(pl.status, k -> new int[2]);
            c[0]++; c[1] += pn.equals(tn) ? 1 : 0;
        }
        System.out.printf("A. %d test crops: plate exact %.2f %%, char acc %.2f %%, same text as Python %.2f %%, %.1f ms/plate (PC CPU)%n",
                n, 100.0 * exact / n, 100.0 * (1 - (double) ce / cl), 100.0 * same / n, tr / 1e6 / n);
        for (Map.Entry<String, int[]> e : st.entrySet())
            System.out.printf("   status %-8s %5.1f %% of plates, %.2f %% exact%n", e.getKey(), 100.0 * e.getValue()[0] / n, 100.0 * e.getValue()[1] / e.getValue()[0]);

        // ---- B. whole photos
        Map<String, String> pyE = new HashMap<>();
        List<String[]> pe = csv(py + "/e2e_preds_" + tag + "_gpu.csv");
        for (String[] r : pe.subList(1, pe.size())) pyE.put(r[0], r[2]);
        List<String[]> e2e = csv(data + "/e2e_test/e2e_test.csv");
        n = 0; exact = 0; same = 0; ce = 0; cl = 0;
        int found = 0;
        long tt = 0;
        for (String[] r : e2e.subList(1, Math.min(e2e.size(), max + 1))) {
            Img im = load(data + "/e2e_test/images/" + r[0]);
            long s = System.nanoTime();
            Pipeline.Result res = p.process(im, night);
            tt += System.nanoTime() - s;
            Pipeline.Plate best = null;
            for (Pipeline.Plate q : res.plates) if (q.valid) { best = q; break; }
            if (best == null && !res.plates.isEmpty()) best = res.plates.get(0);
            String pn = Vocab.normText(best == null ? "" : best.text), tn = Vocab.normText(r[1]);
            n++; found += res.plates.isEmpty() ? 0 : 1; exact += pn.equals(tn) ? 1 : 0;
            same += pn.equals(Vocab.normText(pyE.getOrDefault(r[0], ""))) ? 1 : 0;
            ce += lev(pn, tn); cl += Math.max(1, tn.length());
        }
        System.out.printf("B. %d whole photos: found %.2f %%, plate exact %.2f %%, char acc %.2f %%, same text as Python %.2f %%, %.1f ms/photo = %.1f FPS (PC CPU, %d threads)%n",
                n, 100.0 * found / n, 100.0 * exact / n, 100.0 * (1 - (double) ce / cl), 100.0 * same / n, tt / 1e6 / n, 1e9 * n / tt, Net.threads);
    }
}
