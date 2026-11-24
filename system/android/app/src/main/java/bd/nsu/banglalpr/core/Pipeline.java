package bd.nsu.banglalpr.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * BanglaLPR V1 / V2 on the phone: the same stages and choices as lpr/pipeline.py + lpr/fusion.py and
 * V1/models/config.json, V2/models/config.json (see the explanation PDFs).
 */
public final class Pipeline implements AutoCloseable {
    public static final class Plate {
        public float x1, y1, x2, y2, detConf;
        public String text = "", english = "", status = "review";
        public boolean valid, upscaled, night;
        public float conf;
        public Float charAgreement;
        public Map<String, String> readers = new LinkedHashMap<>();
        public Img crop;
    }

    public static final class Result {
        public final List<Plate> plates = new ArrayList<>();
        public String retry;
        public boolean night;
        public double msDetect, msRead, msTotal;
    }

    public final String version;
    final Yolo det;
    final List<Object[]> retry = new ArrayList<>();          // {Yolo, Boolean clahe, Float conf, String name}
    final List<Readers.Reader> readers = new ArrayList<>();
    final String rule;
    final Readers.Reader statusReader;
    final List<Readers.CharCnn> charCnns = new ArrayList<>();
    /** night mode (V3/models/config.json "night_mode", chosen on val by V3/tools/night_mode_eval.py): the main detector's
     *  confidence and the reading rule. "vote10" = every reader on the crop and on a CLAHE copy, MV-BM+format over all
     *  ten; the status counts only the readings of the crop (a CLAHE copy is not an independent opinion). */
    public float nightDetConf = 0.25f;
    public String nightRead = "vote10";

    public Pipeline(String version, Net.Source src) throws Exception {
        this.version = version;
        det = new Yolo(src, "p18_v8n_576.onnx", 576, true);                  // both versions: Embedded LPR detector
        readers.add(new Readers.CharYolo(src, "p11_chars_v8n", "p11_chars_v8n.onnx", 320));   // primary (Video LPR)
        if (version.equals("v1")) {
            rule = "fallback";
            readers.add(new Readers.CharYolo(src, "p02_lpnr_v12n", "p02_lpnr_v12n.onnx", 640)); // Cascaded YOLO
            readers.add(new Readers.Slot(src, "p18_slot", "p18_slot.onnx"));                     // Embedded LPR
            statusReader = new Readers.Ctc(src, "p20_crnn_hr", "p20_crnn_hr.onnx", 48, 256, true); // Plate SR CRNN
            charCnns.add(new Readers.CharCnn(src, true));                                        // CNN + GFP-GAN
            charCnns.add(new Readers.CharCnn(src, false));                                       // BLPnet
        } else {
            rule = "verify";
            readers.add(new Readers.CharYolo(src, "p27_chars_v8n", "p27_chars_v8n.onnx", 320)); // YOLOv10 run
            readers.add(new Readers.CharYolo(src, "p02_lpnr_v12n", "p02_lpnr_v12n.onnx", 640)); // from V1
            readers.add(new Readers.Svtr(src, "p34_svtrv2_s", "p34_svtrv2_s.onnx"));            // SVTRv2
            readers.add(new Readers.Svtr(src, "p34_svtrv2_t", "p34_svtrv2_t.onnx"));
            statusReader = null;
            Yolo v8s = new Yolo(src, "p06_det_v8s.onnx", 640, true);                           // YOLOv8 + EasyOCR
            retry.add(new Object[]{v8s, false, 0.25f, "p06_det_v8s.pt"});
            retry.add(new Object[]{v8s, true, 0.35f, "p06_det_v8s.pt + CLAHE"});                // HVI-CIDNet study
        }
    }

    // ------------------------------------------------------------------------------------------ fusion
    static boolean same(String a, String b) { return a != null && !a.isEmpty() && b != null && Vocab.normText(a).equals(Vocab.normText(b)); }

    /** metrics.fuse(..., 'MV-BM+format', best): majority of layout-valid readings, tie -> best (primary) model */
    static String fuse(List<String[]> preds, String best) {
        List<String[]> v = new ArrayList<>();
        for (String[] p : preds) if (Vocab.isValidPlate(p[1])) v.add(p);
        if (v.isEmpty()) v = preds;
        LinkedHashMap<String, List<String[]>> g = new LinkedHashMap<>();
        for (String[] p : v) g.computeIfAbsent(Vocab.normText(p[1]), k -> new ArrayList<>()).add(p);
        int top = 0;
        for (List<String[]> l : g.values()) top = Math.max(top, l.size());
        List<List<String[]>> tied = new ArrayList<>();
        for (List<String[]> l : g.values()) if (l.size() == top) tied.add(l);
        if (tied.size() == 1) return tied.get(0).get(0)[1];
        for (List<String[]> l : tied) for (String[] p : l) if (p[0].equals(best)) return l.get(0)[1];
        List<String[]> bestL = tied.get(0);
        double bm = -1;
        for (List<String[]> l : tied) {
            double m = 0;
            for (String[] p : l) m += Double.parseDouble(p[2]);
            m /= l.size();
            if (m > bm) { bm = m; bestL = l; }
        }
        return bestL.get(0)[1];
    }

    static boolean needsHelpers(String rule, String primary, String second) {
        if (rule.equals("vote")) return true;
        if (rule.equals("fallback")) return !Vocab.isValidPlate(primary);
        if (rule.equals("verify")) return !(Vocab.isValidPlate(primary) && same(primary, second));
        return false;
    }

    static String status(String text, List<String> opinions, Float charAgree, float charMin) {
        if (!Vocab.isValidPlate(text)) return "review";
        int agree = 0;
        for (String o : opinions) if (same(text, o)) agree++;
        if (opinions.size() >= 2 && agree >= 2) return "verified";
        if (charAgree != null) return charAgree >= charMin ? "likely" : "review";
        return opinions.size() < 2 ? "likely" : "review";
    }

    // ------------------------------------------------------------------------------------------ reading
    public void readInto(Plate p, Img crop) throws Exception { readInto(p, crop, false); }

    public void readInto(Plate p, Img crop, boolean night) throws Exception {
        Readers.Reading[] runs = new Readers.Reading[readers.size()];
        runs[0] = readers.get(0).read(crop);
        int k0 = rule.equals("verify") ? 2 : 1;
        if (k0 == 2) runs[1] = readers.get(1).read(crop);
        if (readers.size() > k0 && needsHelpers(rule, runs[0].text, k0 == 2 ? runs[1].text : null))
            for (int j = k0; j < readers.size(); j++) runs[j] = readers.get(j).read(crop);
        List<String[]> have = new ArrayList<>();
        List<String> opinions = new ArrayList<>();
        for (int j = 0; j < runs.length; j++)
            if (runs[j] != null) {
                have.add(new String[]{readers.get(j).name(), runs[j].text, String.valueOf(runs[j].conf)});
                opinions.add(runs[j].text);
                p.readers.put(readers.get(j).name(), runs[j].text);
            }
        String text = have.size() == 1 ? runs[0].text : fuse(have, readers.get(0).name());
        if (statusReader != null) {
            String st = statusReader.read(crop).text;
            opinions.add(st);
            p.readers.put(statusReader.name() + " (status)", st);
        }
        if (!charCnns.isEmpty() && runs[0].boxes != null && !runs[0].boxes.isEmpty()) {
            float s = 0;
            for (Readers.CharCnn c : charCnns) s += c.agreement(crop, runs[0].boxes);
            p.charAgreement = s / charCnns.size();
        }
        p.text = text;
        p.english = Vocab.toEnglish(text);
        p.valid = Vocab.isValidPlate(text);
        p.conf = runs[0].conf;
        p.status = status(text, opinions, p.charAgreement, 0.8f);
        if (night && rule.equals("verify") && nightRead.equals("vote10")) {
            List<String[]> all = new ArrayList<>();
            List<String> own = new ArrayList<>();
            for (int j = 0; j < readers.size(); j++) {
                if (runs[j] == null) runs[j] = readers.get(j).read(crop);
                all.add(new String[]{readers.get(j).name(), runs[j].text, String.valueOf(runs[j].conf)});
                own.add(runs[j].text);
                p.readers.put(readers.get(j).name(), runs[j].text);
            }
            Img cl = crop.clahe();
            for (Readers.Reader r : readers) {
                Readers.Reading rc = r.read(cl);
                all.add(new String[]{r.name() + "+clahe", rc.text, String.valueOf(rc.conf)});
                p.readers.put(r.name() + "+clahe", rc.text);
            }
            p.text = fuse(all, readers.get(0).name());
            p.english = Vocab.toEnglish(p.text);
            p.valid = Vocab.isValidPlate(p.text);
            p.status = status(p.text, own, null, 0.8f);
            p.night = true;
        }
    }

    /** whole photo / camera frame -> plates (at most 5, best detector score first) */
    public Result process(Img img) throws Exception { return process(img, false); }

    /** night: the app's night mode (clock or the user's switch) */
    public Result process(Img img, boolean night) throws Exception {
        Result r = new Result();
        r.night = night;
        long t0 = System.nanoTime();
        List<Yolo.Box> boxes = det.predict(img, night ? nightDetConf : 0.25f);
        for (Object[] rt : retry) {
            if (!boxes.isEmpty()) break;
            Img in = (Boolean) rt[1] ? img.clahe() : img;
            boxes = ((Yolo) rt[0]).predict(in, (Float) rt[2]);
            r.retry = (String) rt[3];
        }
        long t1 = System.nanoTime();
        int n = 0;
        for (Yolo.Box b : boxes) {
            if (n >= 5) break;
            float pw = (b.x2 - b.x1) * 0.08f, ph = (b.y2 - b.y1) * 0.08f;     // crop + 8 % (Cascaded YOLO)
            Img c = img.crop((int) Math.max(0, b.x1 - pw), (int) Math.max(0, b.y1 - ph),
                             (int) Math.min(img.w, b.x2 + pw), (int) Math.min(img.h, b.y2 + ph));
            if (c.w == 0 || c.h == 0) continue;
            n++;
            Plate p = new Plate();
            p.x1 = b.x1; p.y1 = b.y1; p.x2 = b.x2; p.y2 = b.y2; p.detConf = b.conf;
            if (c.h < 32) { c = c.resizeCubic(c.w * 2, c.h * 2); p.upscaled = true; }   // tiny plate: x2 bicubic
            p.crop = c;
            readInto(p, c, night);
            r.plates.add(p);
        }
        long t2 = System.nanoTime();
        r.msDetect = (t1 - t0) / 1e6; r.msRead = (t2 - t1) / 1e6; r.msTotal = (t2 - t0) / 1e6;
        return r;
    }

    /** reader part only (a plate crop) */
    public Plate readCrop(Img crop) throws Exception { return readCrop(crop, false); }

    public Plate readCrop(Img crop, boolean night) throws Exception {
        Plate p = new Plate();
        p.crop = crop;
        readInto(p, crop, night);
        return p;
    }

    public static Map<String, String> paperNames() {
        Map<String, String> m = new HashMap<>();
        m.put("p18", "Embedded LPR"); m.put("p11", "Video LPR"); m.put("p02", "Cascaded YOLO"); m.put("p20", "Plate SR");
        m.put("p08", "CNN + GFP-GAN"); m.put("p10", "BLPnet"); m.put("p27", "YOLOv10"); m.put("p34", "SVTRv2");
        m.put("p06", "YOLOv8 + EasyOCR"); return m;
    }

    @Override public void close() { /* sessions are freed with the process */ }
}
