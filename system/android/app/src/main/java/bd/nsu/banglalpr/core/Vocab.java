package bd.nsu.banglalpr.core;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** BRTA plate grammar: the same rules as lpr/vocab.py and lpr/metrics.py (token kinds, layout check, text form). */
public final class Vocab {
    private Vocab() {}

    public static final String[] CLASSES = nfcAll(VocabData.CLASSES);
    public static final String[] P10_CLASSES = nfcAll(VocabData.P10_CLASSES);
    private static final Map<String, String> KIND = map(VocabData.KIND_KEYS, VocabData.KIND_VALUES);
    private static final Map<String, String> VARIANTS = map(VocabData.VARIANT_KEYS, VocabData.VARIANT_VALUES);
    private static final List<String[]> ENGLISH = new ArrayList<>();
    static final String METRO = nfc(VocabData.METRO);
    static final String DIGITS = VocabData.DIGITS;

    static {
        for (int i = 0; i < VocabData.ENGLISH_KEYS.length; i++)
            ENGLISH.add(new String[]{nfc(VocabData.ENGLISH_KEYS[i]), VocabData.ENGLISH_VALUES[i]});
        ENGLISH.sort((a, b) -> b[0].length() - a[0].length());      // longest key first, like the Python sort
    }

    public static String nfc(String s) { return Normalizer.normalize(s == null ? "" : s, Normalizer.Form.NFC); }

    private static String[] nfcAll(String[] a) {
        String[] r = new String[a.length];
        for (int i = 0; i < a.length; i++) r[i] = nfc(a[i]);
        return r;
    }

    private static Map<String, String> map(String[] k, String[] v) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < k.length; i++) m.put(nfc(k[i]), v[i]);
        return m;
    }

    public static String kind(String token) { return KIND.get(nfc(token)); }

    /** a detected token: text + box */
    public static final class Tok {
        public final String t; public final float x1, y1, x2, y2, conf;
        public Tok(String t, float x1, float y1, float x2, float y2, float conf) {
            this.t = nfc(t); this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2; this.conf = conf;
        }
        float cx() { return (x1 + x2) / 2; }
        float cy() { return (y1 + y2) / 2; }
    }

    /** vocab.plate_text: words on top by kind (not by y), 6 digits as NN-NNNN; valid = exact BRTA layout */
    public static String[] plateText(List<Tok> toks) {
        List<Tok> words = new ArrayList<>(), metro = new ArrayList<>(), letters = new ArrayList<>(), digits = new ArrayList<>();
        for (Tok t : toks) {
            String k = kind(t.t);
            if ("district".equals(k)) words.add(t);
            else if ("metro".equals(k)) metro.add(t);
            else if ("letter".equals(k)) letters.add(t);
            else if ("digit".equals(k)) digits.add(t);
        }
        java.util.Comparator<Tok> byX = (a, b) -> Float.compare(a.cx(), b.cx());
        words.sort(byX); letters.sort(byX); digits.sort(byX);
        StringBuilder top = new StringBuilder();
        for (int i = 0; i < words.size(); i++) { if (i > 0) top.append(' '); top.append(words.get(i).t); }
        if (!metro.isEmpty()) { if (top.length() > 0) top.append(' '); top.append(METRO); }
        if (!letters.isEmpty()) {
            if (top.length() > 0) top.append('-');
            for (Tok t : letters) top.append(t.t);
        }
        StringBuilder ds = new StringBuilder();
        for (Tok t : digits) ds.append(t.t);
        String bottom = ds.length() == 6 ? ds.substring(0, 2) + "-" + ds.substring(2) : ds.toString();
        String text = (top + " " + bottom).trim();
        boolean valid = false;
        if (words.size() == 1 && metro.size() <= 1 && letters.size() == 1 && digits.size() == 6) {
            float up = 0, dn = 0;
            List<Tok> u = new ArrayList<>(words); u.addAll(metro); u.addAll(letters);
            for (Tok t : u) up += t.cy();
            for (Tok t : digits) dn += t.cy();
            valid = up / u.size() < dn / digits.size();
        }
        return new String[]{text, valid ? "1" : "0"};
    }

    /** vocab.norm_text: NFC, Bangla digits, no spaces / hyphens / dots */
    public static String normText(String s) {
        s = nfc(s);
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= '0' && c <= '9') c = DIGITS.charAt(c - '0');
            if (" -–—.:,_".indexOf(c) >= 0) continue;
            b.append(c);
        }
        return b.toString();
    }

    /** vocab.text_tokens: "ঢাকা মেট্রো-গ ১২-৩৪৫৬" -> [ঢাকা, মেট্রো, গ, ১,২,৩,৪,৫,৬] */
    public static List<String> textTokens(String text) {
        String t = nfc(text);
        List<String> out = new ArrayList<>();
        int sp = t.lastIndexOf(' ');
        String head = sp >= 0 ? t.substring(0, sp) : "", digits = sp >= 0 ? t.substring(sp + 1) : t;
        String words, letter;
        int dash = head.lastIndexOf('-');
        if (dash >= 0) { words = head.substring(0, dash); letter = head.substring(dash + 1); }
        else { words = head; letter = ""; }
        for (String wd : words.trim().split("\\s+")) if (!wd.isEmpty()) out.add(wd);
        if (!letter.isEmpty()) out.add(letter);
        for (int i = 0; i < digits.length(); i++) if (DIGITS.indexOf(digits.charAt(i)) >= 0) out.add(String.valueOf(digits.charAt(i)));
        return out;
    }

    /** vocab.tokens_to_text for a predicted token sequence (any order) */
    public static String tokensToText(List<String> toks) {
        StringBuilder top = new StringBuilder(), ds = new StringBuilder();
        List<String> metro = new ArrayList<>(), letters = new ArrayList<>();
        for (String t : toks) {
            String k = kind(t);
            if ("district".equals(k)) { if (top.length() > 0) top.append(' '); top.append(nfc(t)); }
            else if ("metro".equals(k)) metro.add(t);
            else if ("letter".equals(k)) letters.add(nfc(t));
            else if ("digit".equals(k)) ds.append(t);
        }
        if (!metro.isEmpty()) { if (top.length() > 0) top.append(' '); top.append(METRO); }
        if (!letters.isEmpty()) { if (top.length() > 0) top.append('-'); for (String l : letters) top.append(l); }
        String bottom = ds.length() == 6 ? ds.substring(0, 2) + "-" + ds.substring(2) : ds.toString();
        return (top + " " + bottom).trim();
    }

    /** metrics.is_valid_plate: one district, at most one মেট্রো, one letter, six digits, nothing unknown */
    public static boolean isValidPlate(String text) {
        if (text == null || text.isEmpty()) return false;
        List<String> tk = textTokens(text);
        int d = 0, m = 0, l = 0, g = 0;
        for (String t : tk) {
            String k = kind(t);
            if (k == null) return false;
            switch (k) {
                case "district": d++; break;
                case "metro": m++; break;
                case "letter": l++; break;
                default: g++;
            }
        }
        return d == 1 && l == 1 && g == 6 && m <= 1;
    }

    /** Bangla plate -> English, e.g. "Dhaka Metro-Ga 12-3456" (pipeline.to_english) */
    public static String toEnglish(String text) {
        if (text == null || text.isEmpty()) return "";
        StringBuilder b = new StringBuilder();
        String t = nfc(text);
        for (int i = 0; i < t.length(); i++) {
            int di = DIGITS.indexOf(t.charAt(i));
            b.append(di >= 0 ? (char) ('0' + di) : t.charAt(i));
        }
        String s = b.toString();
        for (String[] kv : ENGLISH) if (!(kv[0].length() == 1 && Character.isDigit(kv[0].charAt(0)))) s = s.replace(kv[0], kv[1]);
        return s;
    }

    public static List<String> unmodifiable(List<String> l) { return Collections.unmodifiableList(l); }
}
