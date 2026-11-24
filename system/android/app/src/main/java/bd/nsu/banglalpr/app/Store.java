package bd.nsu.banglalpr.app;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.graphics.Bitmap;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

/** Results stored on the phone only: SQLite (plates.db) + plate crop JPEGs in the app's private files folder. */
final class Store extends SQLiteOpenHelper {
    static final class Row {
        long id, ts; String version, text, english, status, source, crop; float conf; int watch;
    }

    private final File cropDir;

    Store(Context c) {
        super(c, "plates.db", null, 1);
        cropDir = new File(c.getFilesDir(), "crops");
        cropDir.mkdirs();
    }

    @Override public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE plates (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, version TEXT, text TEXT, " +
                "norm TEXT, english TEXT, status TEXT, conf REAL, source TEXT, crop TEXT)");
        db.execSQL("CREATE INDEX plates_norm ON plates(norm)");
        db.execSQL("CREATE TABLE watch (norm TEXT PRIMARY KEY, text TEXT, note TEXT, ts INTEGER)");
    }

    @Override public void onUpgrade(SQLiteDatabase db, int o, int n) { }

    long add(String version, String text, String norm, String english, String status, float conf, String source, Bitmap crop) {
        String path = null;
        if (crop != null) {
            File f = new File(cropDir, "crop_" + System.currentTimeMillis() + "_" + (int) (Math.random() * 1e6) + ".jpg");
            try (FileOutputStream o = new FileOutputStream(f)) { crop.compress(Bitmap.CompressFormat.JPEG, 90, o); path = f.getAbsolutePath(); }
            catch (Exception ignored) { }
        }
        ContentValues v = new ContentValues();
        v.put("ts", System.currentTimeMillis()); v.put("version", version); v.put("text", text); v.put("norm", norm);
        v.put("english", english); v.put("status", status); v.put("conf", conf); v.put("source", source); v.put("crop", path);
        return getWritableDatabase().insert("plates", null, v);
    }

    List<Row> list(String q, int limit) {
        List<Row> out = new ArrayList<>();
        String like = "%" + q + "%";
        try (Cursor c = getReadableDatabase().rawQuery(
                "SELECT p.id, p.ts, p.version, p.text, p.english, p.status, p.conf, p.source, p.crop, " +
                "(SELECT COUNT(*) FROM watch w WHERE w.norm = p.norm) FROM plates p " +
                "WHERE p.norm LIKE ? OR p.english LIKE ? ORDER BY p.id DESC LIMIT " + limit, new String[]{like, like})) {
            while (c.moveToNext()) {
                Row r = new Row();
                r.id = c.getLong(0); r.ts = c.getLong(1); r.version = c.getString(2); r.text = c.getString(3);
                r.english = c.getString(4); r.status = c.getString(5); r.conf = c.getFloat(6); r.source = c.getString(7);
                r.crop = c.getString(8); r.watch = c.getInt(9);
                out.add(r);
            }
        }
        return out;
    }

    int count() {
        try (Cursor c = getReadableDatabase().rawQuery("SELECT COUNT(*) FROM plates", null)) { c.moveToFirst(); return c.getInt(0); }
    }

    boolean watched(String norm) {
        try (Cursor c = getReadableDatabase().rawQuery("SELECT 1 FROM watch WHERE norm = ?", new String[]{norm})) { return c.moveToFirst(); }
    }

    void watchAdd(String norm, String text, String note) {
        ContentValues v = new ContentValues();
        v.put("norm", norm); v.put("text", text); v.put("note", note); v.put("ts", System.currentTimeMillis());
        getWritableDatabase().insertWithOnConflict("watch", null, v, SQLiteDatabase.CONFLICT_REPLACE);
    }

    List<String[]> watchList() {
        List<String[]> out = new ArrayList<>();
        try (Cursor c = getReadableDatabase().rawQuery("SELECT norm, text, note FROM watch ORDER BY ts DESC", null)) {
            while (c.moveToNext()) out.add(new String[]{c.getString(0), c.getString(1), c.getString(2)});
        }
        return out;
    }

    void watchDel(String norm) { getWritableDatabase().delete("watch", "norm = ?", new String[]{norm}); }

    void clear() {
        getWritableDatabase().delete("plates", null, null);
        File[] fs = cropDir.listFiles();
        if (fs != null) for (File f : fs) f.delete();
    }

    /** CSV of every stored plate (for sharing / the report) */
    File exportCsv(File dir) throws Exception {
        dir.mkdirs();
        File f = new File(dir, "banglalpr_results.csv");
        StringBuilder b = new StringBuilder("﻿id,time,version,plate,english,status,confidence,source\n");
        java.text.SimpleDateFormat fmt = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US);
        for (Row r : list("", 100000))
            b.append(r.id).append(',').append(fmt.format(new java.util.Date(r.ts))).append(',').append(r.version).append(',')
             .append(r.text).append(',').append(r.english).append(',').append(r.status).append(',')
             .append(String.format(java.util.Locale.US, "%.3f", r.conf)).append(',').append(r.source).append('\n');
        try (FileOutputStream o = new FileOutputStream(f)) { o.write(b.toString().getBytes("UTF-8")); }
        return f;
    }
}
