package bd.nsu.banglalpr.app;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * V3 records on the phone: every plate = one plate picture + one row in plates.csv (same columns as the PC app).
 *
 * records/plates.csv                         newest last; "Plate image" = path of the picture, relative to the CSV
 * records/plate_images/<date>/<no>_<plate>.jpg
 * records/exports/export_<time>.zip          results.csv + plate_images/ + results.html (also copied to Download/BanglaLPR)
 *
 * The records folder is the app's external files folder (Android/data/bd.nsu.banglalpr.v3/files/records), which a PC
 * sees over USB. SQLite (records.db) only indexes the rows for search; the CSV is rewritten after deletions so both match.
 */
final class Records extends SQLiteOpenHelper {
    static final String[] COLS = {"No", "Date", "Time", "Plate (Bangla)", "Plate (English)", "Status", "BRTA layout OK",
            "Read confidence", "Detector confidence", "Source", "File name", "Plate image", "Plate image (full path)",
            "Vehicle photo", "Watchlist match", "Model", "Night mode"};

    static final class Row {
        long id, ts; String text, english, status, source, filename, crop; boolean valid, night; float conf, detConf; int watch;
    }

    final File folder, csv;

    Records(Context c) {
        super(c, "records.db", null, 2);              // 2: night column (night mode, 22 Nov 2026)
        File ext = c.getExternalFilesDir(null);
        folder = new File(ext != null ? ext : c.getFilesDir(), "records");
        new File(folder, "plate_images").mkdirs();
        csv = new File(folder, "plates.csv");
        if (!csv.exists() || !csvHeaderCurrent()) rewriteCsv();
    }

    boolean csvHeaderCurrent() {
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(new java.io.FileInputStream(csv), StandardCharsets.UTF_8))) {
            String h = r.readLine();
            return h != null && h.replace("﻿", "").equals(String.join(",", COLS));
        } catch (Exception e) { return false; }
    }

    @Override public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE plates (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, text TEXT, norm TEXT, english TEXT, " +
                "status TEXT, valid INTEGER, conf REAL, det_conf REAL, source TEXT, filename TEXT, crop TEXT, night INTEGER DEFAULT 0)");
        db.execSQL("CREATE INDEX plates_norm ON plates(norm)");
        db.execSQL("CREATE TABLE watch (norm TEXT PRIMARY KEY, text TEXT, note TEXT, ts INTEGER)");
    }

    @Override public void onUpgrade(SQLiteDatabase db, int o, int n) {
        if (o < 2) db.execSQL("ALTER TABLE plates ADD COLUMN night INTEGER DEFAULT 0");
    }

    static String slug(String en) {
        String s = (en == null ? "" : en).replaceAll("[^A-Za-z0-9]+", "-").replaceAll("^-+|-+$", "").toUpperCase(Locale.US);
        if (s.isEmpty()) s = "UNREADABLE";
        return s.length() > 40 ? s.substring(0, 40) : s;
    }

    static String day(long ts) { return new SimpleDateFormat("yyyy-MM-dd", Locale.US).format(new Date(ts)); }

    File file(String rel) { return new File(folder, rel); }

    // ------------------------------------------------------------------------------------------ writes
    synchronized Row add(String text, String norm, String english, String status, boolean valid, float conf, float detConf,
                         String source, String filename, Bitmap crop, boolean night) {
        long ts = System.currentTimeMillis();
        ContentValues v = new ContentValues();
        v.put("ts", ts); v.put("text", text); v.put("norm", norm); v.put("english", english); v.put("status", status);
        v.put("valid", valid ? 1 : 0); v.put("conf", conf); v.put("det_conf", detConf); v.put("source", source);
        v.put("filename", filename); v.put("night", night ? 1 : 0);
        SQLiteDatabase db = getWritableDatabase();
        long id = db.insert("plates", null, v);
        String rel = null;
        if (crop != null) {
            rel = "plate_images/" + day(ts) + "/" + String.format(Locale.US, "%06d", id) + "_" + slug(english) + ".jpg";
            File f = file(rel);
            f.getParentFile().mkdirs();
            try (FileOutputStream o = new FileOutputStream(f)) {
                crop.compress(Bitmap.CompressFormat.JPEG, 95, o);
                ContentValues u = new ContentValues();
                u.put("crop", rel);
                db.update("plates", u, "id = ?", new String[]{String.valueOf(id)});
            } catch (Exception e) { rel = null; }
        }
        Row r = new Row();
        r.id = id; r.ts = ts; r.text = text; r.english = english; r.status = status; r.valid = valid; r.conf = conf;
        r.detConf = detConf; r.source = source; r.filename = filename; r.crop = rel; r.watch = watched(norm) ? 1 : 0; r.night = night;
        try (OutputStream o = new FileOutputStream(csv, true)) { o.write(csvLine(r, null).getBytes(StandardCharsets.UTF_8)); }
        catch (Exception ignored) { }
        return r;
    }

    synchronized void delete(long id) {
        for (Row r : query("", "", 0, "p.id = " + id)) if (r.crop != null) file(r.crop).delete();
        getWritableDatabase().delete("plates", "id = ?", new String[]{String.valueOf(id)});
        rewriteCsv();
    }

    synchronized void clear() {
        getWritableDatabase().delete("plates", null, null);
        deleteTree(new File(folder, "plate_images"));
        new File(folder, "plate_images").mkdirs();
        rewriteCsv();
    }

    static void deleteTree(File f) {
        File[] fs = f.listFiles();
        if (fs != null) for (File x : fs) deleteTree(x);
        f.delete();
    }

    // ------------------------------------------------------------------------------------------ reads
    List<Row> list(String q, String status, int limit) { return query(q, status, limit, null); }

    List<Row> query(String q, String status, int limit, String extra) {
        List<Row> out = new ArrayList<>();
        StringBuilder sql = new StringBuilder("SELECT p.id, p.ts, p.text, p.english, p.status, p.valid, p.conf, p.det_conf, p.source, " +
                "p.filename, p.crop, (SELECT COUNT(*) FROM watch w WHERE w.norm = p.norm), p.night FROM plates p WHERE 1=1");
        List<String> args = new ArrayList<>();
        if (q != null && !q.isEmpty()) {
            sql.append(" AND (p.norm LIKE ? OR p.english LIKE ?)");
            args.add("%" + bd.nsu.banglalpr.core.Vocab.normText(q) + "%"); args.add("%" + q + "%");
        }
        if (status != null && !status.isEmpty()) { sql.append(" AND p.status = ?"); args.add(status); }
        if (extra != null) sql.append(" AND ").append(extra);
        sql.append(" ORDER BY p.id DESC");
        if (limit > 0) sql.append(" LIMIT ").append(limit);
        try (Cursor c = getReadableDatabase().rawQuery(sql.toString(), args.toArray(new String[0]))) {
            while (c.moveToNext()) {
                Row r = new Row();
                r.id = c.getLong(0); r.ts = c.getLong(1); r.text = c.getString(2); r.english = c.getString(3);
                r.status = c.getString(4); r.valid = c.getInt(5) == 1; r.conf = c.getFloat(6); r.detConf = c.getFloat(7);
                r.source = c.getString(8); r.filename = c.getString(9); r.crop = c.getString(10); r.watch = c.getInt(11);
                r.night = c.getInt(12) == 1;
                out.add(r);
            }
        }
        return out;
    }

    int count(String where) {
        try (Cursor c = getReadableDatabase().rawQuery("SELECT COUNT(*) FROM plates" + (where == null ? "" : " WHERE " + where), null)) {
            c.moveToFirst();
            return c.getInt(0);
        }
    }

    int countToday() {
        java.util.Calendar k = java.util.Calendar.getInstance();
        k.set(java.util.Calendar.HOUR_OF_DAY, 0); k.set(java.util.Calendar.MINUTE, 0); k.set(java.util.Calendar.SECOND, 0); k.set(java.util.Calendar.MILLISECOND, 0);
        return count("ts >= " + k.getTimeInMillis());
    }

    // ------------------------------------------------------------------------------------------ watchlist
    boolean watched(String norm) {
        if (norm == null || norm.isEmpty()) return false;
        try (Cursor c = getReadableDatabase().rawQuery("SELECT 1 FROM watch WHERE norm = ?", new String[]{norm})) { return c.moveToFirst(); }
    }

    synchronized void watchAdd(String norm, String text, String note) {
        ContentValues v = new ContentValues();
        v.put("norm", norm); v.put("text", text); v.put("note", note); v.put("ts", System.currentTimeMillis());
        getWritableDatabase().insertWithOnConflict("watch", null, v, SQLiteDatabase.CONFLICT_REPLACE);
        rewriteCsv();
    }

    List<String[]> watchList() {
        List<String[]> out = new ArrayList<>();
        try (Cursor c = getReadableDatabase().rawQuery("SELECT norm, text, note FROM watch ORDER BY ts DESC", null)) {
            while (c.moveToNext()) out.add(new String[]{c.getString(0), c.getString(1), c.getString(2)});
        }
        return out;
    }

    synchronized void watchDel(String norm) {
        getWritableDatabase().delete("watch", "norm = ?", new String[]{norm});
        rewriteCsv();
    }

    // ------------------------------------------------------------------------------------------ CSV
    static String statusWord(String s) {
        if ("verified".equals(s)) return "Verified (readers agree)";
        if ("likely".equals(s)) return "Likely (check if important)";
        if ("review".equals(s)) return "Check manually";
        return s == null ? "" : s;
    }

    static String q(String s) {
        if (s == null) return "";
        return (s.contains(",") || s.contains("\"") || s.contains("\n")) ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
    }

    /** one CSV line; imgRel overrides the image path (exports), null = the records-folder path */
    String csvLine(Row r, String imgRel) {
        Date d = new Date(r.ts);
        String rel = imgRel != null ? imgRel : (r.crop == null ? "" : r.crop);
        String full = r.crop == null ? "" : (imgRel != null ? "" : file(r.crop).getAbsolutePath());
        String[] v = {String.valueOf(r.id), new SimpleDateFormat("yyyy-MM-dd", Locale.US).format(d),
                new SimpleDateFormat("HH:mm:ss", Locale.US).format(d), r.text, r.english, statusWord(r.status),
                r.valid ? "yes" : "no", String.format(Locale.US, "%.3f", r.conf),
                r.detConf > 0 ? String.format(Locale.US, "%.3f", r.detConf) : "", r.source, r.filename, rel, full, "",
                r.watch > 0 ? "YES" : "", "V3", r.night ? "on" : "off"};
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < v.length; i++) b.append(i == 0 ? "" : ",").append(q(v[i]));
        return b.append("\r\n").toString();
    }

    static String header() { return "﻿" + String.join(",", COLS) + "\r\n"; }      // BOM: Excel shows Bangla

    synchronized void rewriteCsv() {
        List<Row> rows = query("", "", 0, null);
        StringBuilder b = new StringBuilder(header());
        for (int i = rows.size() - 1; i >= 0; i--) b.append(csvLine(rows.get(i), null));
        File tmp = new File(folder, "plates.csv.tmp");
        try (OutputStream o = new FileOutputStream(tmp)) { o.write(b.toString().getBytes(StandardCharsets.UTF_8)); }
        catch (Exception e) { return; }
        if (!tmp.renameTo(csv)) { csv.delete(); tmp.renameTo(csv); }
    }

    // ------------------------------------------------------------------------------------------ export
    static final class Export { File zip; String publicPath; int plates, images; }

    /** zip of results.csv + plate_images/ + results.html; also copied to Download/BanglaLPR on Android 10+ */
    Export export(Context ctx, String qq, String status) throws Exception {
        List<Row> rows = list(qq, status, 0);
        String name = "BanglaLPR_V3_export_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File dir = new File(folder, "exports");
        dir.mkdirs();
        File zip = new File(dir, name + ".zip");
        Export ex = new Export();
        ex.zip = zip; ex.plates = rows.size();
        StringBuilder csvB = new StringBuilder(header()), html = new StringBuilder();
        try (ZipOutputStream z = new ZipOutputStream(new FileOutputStream(zip))) {
            for (int i = rows.size() - 1; i >= 0; i--) {
                Row r = rows.get(i);
                String img = null;
                if (r.crop != null && file(r.crop).exists()) {
                    img = "plate_images/" + file(r.crop).getName();
                    z.putNextEntry(new ZipEntry(name + "/" + img));
                    try (InputStream in = new FileInputStream(file(r.crop))) { copy(in, z); }
                    z.closeEntry();
                    ex.images++;
                }
                csvB.append(csvLine(r, img == null ? "" : img));
                html.append(htmlRow(r, img));
            }
            put(z, name + "/results.csv", csvB.toString());
            put(z, name + "/results.html", sheet(html.toString(), rows.size()));
        }
        ex.publicPath = toDownloads(ctx, zip);
        return ex;
    }

    static void copy(InputStream in, OutputStream o) throws java.io.IOException {
        byte[] buf = new byte[1 << 16];
        int n;
        while ((n = in.read(buf)) > 0) o.write(buf, 0, n);
    }

    static void put(ZipOutputStream z, String name, String s) throws java.io.IOException {
        z.putNextEntry(new ZipEntry(name));
        z.write(s.getBytes(StandardCharsets.UTF_8));
        z.closeEntry();
    }

    /** a copy in the phone's public Download/BanglaLPR folder (Files app); null when not possible (Android 9 or older) */
    static String toDownloads(Context ctx, File zip) {
        if (Build.VERSION.SDK_INT < 29) return null;
        try {
            ContentValues v = new ContentValues();
            v.put(MediaStore.MediaColumns.DISPLAY_NAME, zip.getName());
            v.put(MediaStore.MediaColumns.MIME_TYPE, "application/zip");
            v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/BanglaLPR");
            Uri u = ctx.getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
            if (u == null) return null;
            try (OutputStream o = ctx.getContentResolver().openOutputStream(u); InputStream in = new FileInputStream(zip)) { copy(in, o); }
            return "Download/BanglaLPR/" + zip.getName();
        } catch (Exception e) { return null; }
    }

    static String esc(String s) {
        return s == null ? "" : s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;");
    }

    static String htmlRow(Row r, String img) {
        Date d = new Date(r.ts);
        return "<tr" + (r.watch > 0 ? " class=hit" : "") + "><td>" + r.id + "</td><td>" +
                (img == null ? "<span class=none>no image</span>" : "<img src=\"" + esc(img) + "\">") + "</td><td class=bn>" +
                esc(r.text) + "</td><td class=mono>" + esc(r.english) + "</td><td><span class=\"st " + esc(r.status) + "\">" +
                esc(statusWord(r.status)) + "</span>" + (r.watch > 0 ? "<b class=w>WATCHLIST</b>" : "") + "</td><td>" +
                new SimpleDateFormat("yyyy-MM-dd'<br><small>'HH:mm:ss'</small>'", Locale.US).format(d) + "</td><td>" +
                esc(r.source) + "<br><small>" + esc(r.filename) + "</small></td></tr>\n";
    }

    static String sheet(String rows, int n) {
        return "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width, initial-scale=1\">" +
                "<title>BanglaLPR V3 results</title><style>body{font:14px/1.45 system-ui,sans-serif;color:#17201b;margin:20px}" +
                "h1{font-size:20px;margin:0 0 4px}p{color:#5d6b63}table{border-collapse:collapse;width:100%}" +
                "th,td{border-bottom:1px solid #dfe5df;padding:8px;text-align:left;vertical-align:middle}" +
                "th{font-size:12px;text-transform:uppercase;color:#5d6b63;background:#f6f8f6}td img{height:56px;border-radius:4px;border:1px solid #ccd}" +
                ".bn{font-size:19px;font-weight:700;white-space:nowrap}.mono{font-family:monospace;white-space:nowrap}small{color:#5d6b63}" +
                ".st{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;background:#eef1ee;white-space:nowrap}" +
                ".st.verified{background:#e3f1ec;color:#006a4e}.st.likely{background:#fff1da;color:#8a5200}.st.review{background:#fbe6e9;color:#b01c30}" +
                "b.w{margin-left:6px;background:#d3243a;color:#fff;font-size:11px;padding:1px 6px;border-radius:4px}tr.hit td{background:#fbe6e9}.none{color:#999}" +
                "</style></head><body><h1>BanglaLPR V3 results</h1><p>" + n + " plates &middot; exported " +
                new SimpleDateFormat("d MMM yyyy, HH:mm", Locale.US).format(new Date()) +
                " on the phone &middot; same rows as results.csv</p><table><thead><tr><th>No</th><th>Plate image</th><th>Plate (Bangla)</th>" +
                "<th>English</th><th>Status</th><th>Date</th><th>Source</th></tr></thead><tbody>" + rows + "</tbody></table></body></html>";
    }
}
