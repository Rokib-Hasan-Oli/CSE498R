package bd.nsu.banglalpr.app;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Size;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import android.util.Log;

import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import bd.nsu.banglalpr.BuildConfig;
import bd.nsu.banglalpr.core.Img;
import bd.nsu.banglalpr.core.Net;
import bd.nsu.banglalpr.core.Pipeline;
import bd.nsu.banglalpr.core.Vocab;

/**
 * BanglaLPR on the phone, fully offline. Tabs: Scan (live camera), Photo (camera / gallery), History (stored on the
 * phone, watchlist, CSV), About. The model (V1 or V2, chosen at build time) runs with ONNX Runtime on this device.
 */
public class MainActivity extends ComponentActivity {
    static final int GREEN = Color.rgb(0, 106, 78), DARK = Color.rgb(15, 43, 33), BG = Color.rgb(242, 245, 242),
            MUTED = Color.rgb(95, 109, 101), RED = Color.rgb(211, 36, 58);
    final String version = BuildConfig.MODEL;
    final ExecutorService worker = Executors.newSingleThreadExecutor();
    volatile Pipeline lpr;
    Store store;
    FrameLayout content;
    final Map<String, View> panels = new HashMap<>();
    final List<Button> tabs = new ArrayList<>();

    // scan
    PreviewView preview; Overlay overlay; TextView lastText, lastEn, lastInfo, camStat; Button scanBtn;
    ProcessCameraProvider camProvider; volatile boolean scanning; volatile boolean busy;
    final Map<String, Long> recent = new HashMap<>();
    // photo
    ImageView photoView; LinearLayout photoPlates; TextView photoInfo; Uri photoUri;
    // history
    LinearLayout histList, watchListView; EditText histQ; TextView histCount;

    ActivityResultLauncher<String> camPerm, pickImage;
    ActivityResultLauncher<Uri> takePicture;

    int dp(float v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        store = new Store(this);
        camPerm = registerForActivityResult(new ActivityResultContracts.RequestPermission(), ok -> {
            if (ok) startCamera(); else toast("Camera permission is needed for live scanning");
        });
        pickImage = registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> { if (uri != null) runPhoto(uri, "gallery"); });
        takePicture = registerForActivityResult(new ActivityResultContracts.TakePicture(), ok -> { if (ok && photoUri != null) runPhoto(photoUri, "photo"); });
        setContentView(buildUi());
        showTab("scan");
        handleShare(getIntent());
        lastInfo.setText("Loading the " + version.toUpperCase(Locale.US) + " model on this phone…");
        worker.execute(() -> {
            try {
                Net.threads = Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors()));
                Net.xnnpack = false;     // XNNPACK mishandled dynamic batches on a test device; plain CPU = the engine checked on the PC
                long t0 = System.nanoTime();
                Pipeline p = new Pipeline(version, name -> {
                    try (InputStream in = getAssets().open(name)) { return readAll(in); }
                });
                lpr = p;
                runOnUiThread(() -> { if (pendingShare != null) { Uri u = pendingShare; pendingShare = null; showTab("photo"); runPhoto(u, "shared"); } });
                double s = (System.nanoTime() - t0) / 1e9;
                runOnUiThread(() -> lastInfo.setText(String.format(Locale.US, "%s ready in %.1f s · runs on this phone, no internet", version.toUpperCase(Locale.US), s)));
            } catch (Throwable e) {
                Log.e("BanglaLPR", "model load", e);
                runOnUiThread(() -> lastInfo.setText("Model failed to load: " + err(e)));
            }
        });
    }

    Uri pendingShare;

    /** a photo shared to the app (Share -> BanglaLPR) is read like a gallery photo */
    void handleShare(Intent i) {
        if (i == null || !Intent.ACTION_SEND.equals(i.getAction())) return;
        Uri u = i.getParcelableExtra(Intent.EXTRA_STREAM);
        if (u == null) return;
        if (lpr == null) { pendingShare = u; return; }
        showTab("photo");
        runPhoto(u, "shared");
    }

    @Override protected void onNewIntent(Intent i) { super.onNewIntent(i); handleShare(i); }

    /** error text with the first lines of our own code, so a screenshot is enough to find the bug */
    static String err(Throwable e) {
        StringBuilder b = new StringBuilder(String.valueOf(e));
        int n = 0;
        for (StackTraceElement st : e.getStackTrace())
            if (st.getClassName().startsWith("bd.nsu") && n++ < 4) b.append("\n  at ").append(st.getClassName().replace("bd.nsu.banglalpr.", "")).append('.').append(st.getMethodName()).append(':').append(st.getLineNumber());
        return b.toString();
    }

    static byte[] readAll(InputStream in) throws java.io.IOException {
        java.io.ByteArrayOutputStream o = new java.io.ByteArrayOutputStream();
        byte[] buf = new byte[1 << 16];
        int n;
        while ((n = in.read(buf)) > 0) o.write(buf, 0, n);
        return o.toByteArray();
    }

    // ------------------------------------------------------------------------------------------ UI helpers
    TextView text(String s, float sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s); t.setTextSize(sp); t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    GradientDrawable round(int fill, int stroke, float r) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(fill); g.setCornerRadius(dp(r));
        if (stroke != 0) g.setStroke(dp(1.5f), stroke);
        return g;
    }

    Button button(String s, boolean primary) {
        Button bt = new Button(this);
        bt.setText(s); bt.setAllCaps(false); bt.setTextSize(16); bt.setTypeface(Typeface.DEFAULT_BOLD);
        bt.setTextColor(primary ? Color.WHITE : GREEN);
        bt.setBackground(round(primary ? GREEN : Color.WHITE, primary ? 0 : GREEN, 14));
        bt.setPadding(dp(12), dp(12), dp(12), dp(12));
        return bt;
    }

    LinearLayout card() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setBackground(round(Color.WHITE, 0, 14));
        l.setPadding(dp(14), dp(12), dp(14), dp(12));
        return l;
    }

    LinearLayout.LayoutParams lp(int w, int h, float weight, int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(w, h, weight);
        p.topMargin = dp(top);
        return p;
    }

    void toast(String s) { runOnUiThread(() -> Toast.makeText(this, s, Toast.LENGTH_LONG).show()); }

    static String chip(String status) { return status == null ? "" : status.toUpperCase(Locale.US); }

    View buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.VERTICAL);
        bar.setBackgroundColor(DARK);
        bar.setPadding(dp(16), dp(14), dp(16), dp(12));
        bar.addView(text("BanglaLPR " + version.toUpperCase(Locale.US), 21, Color.WHITE, true));
        bar.addView(text(version.equals("v1") ? "Bangla cascade · on-device · offline" : "Multi-reader verify · on-device · offline", 13, Color.rgb(169, 195, 182), false));
        root.addView(bar);
        content = new FrameLayout(this);
        root.addView(content, new LinearLayout.LayoutParams(-1, 0, 1));
        panels.put("scan", buildScan()); panels.put("photo", buildPhoto()); panels.put("history", buildHistory()); panels.put("about", buildAbout());
        for (View v : panels.values()) content.addView(v, new FrameLayout.LayoutParams(-1, -1));
        LinearLayout nav = new LinearLayout(this);
        nav.setBackgroundColor(Color.WHITE);
        String[][] items = {{"scan", "Scan"}, {"photo", "Photo"}, {"history", "History"}, {"about", "About"}};
        for (String[] it : items) {
            Button t = new Button(this);
            t.setText(it[1]); t.setAllCaps(false); t.setTextSize(14); t.setBackgroundColor(Color.WHITE); t.setTag(it[0]);
            t.setOnClickListener(v -> showTab((String) v.getTag()));
            tabs.add(t);
            nav.addView(t, new LinearLayout.LayoutParams(0, dp(56), 1));
        }
        root.addView(nav);
        return root;
    }

    void showTab(String key) {
        for (Map.Entry<String, View> e : panels.entrySet()) e.getValue().setVisibility(e.getKey().equals(key) ? View.VISIBLE : View.GONE);
        for (Button t : tabs) {
            boolean on = t.getTag().equals(key);
            t.setTextColor(on ? GREEN : MUTED); t.setTypeface(on ? Typeface.DEFAULT_BOLD : Typeface.DEFAULT);
        }
        if (key.equals("history")) loadHistory();
        if (!key.equals("scan") && scanning) stopCamera();
    }

    // ------------------------------------------------------------------------------------------ scan
    View buildScan() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(dp(12), dp(12), dp(12), dp(12));
        FrameLayout cam = new FrameLayout(this);
        cam.setBackground(round(Color.rgb(11, 21, 16), 0, 14));
        cam.setClipToOutline(true);
        preview = new PreviewView(this);
        preview.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        overlay = new Overlay(this);
        camStat = text("Tap Start and point the back camera at a number plate.", 12, Color.WHITE, false);
        camStat.setBackgroundColor(Color.argb(140, 0, 0, 0));
        camStat.setPadding(dp(8), dp(4), dp(8), dp(4));
        cam.addView(preview, new FrameLayout.LayoutParams(-1, -1));
        cam.addView(overlay, new FrameLayout.LayoutParams(-1, -1));
        FrameLayout.LayoutParams sp = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.START);
        sp.setMargins(dp(8), dp(8), dp(8), dp(8));
        cam.addView(camStat, sp);
        l.addView(cam, new LinearLayout.LayoutParams(-1, 0, 1));
        scanBtn = button("Start scanning", true);
        scanBtn.setOnClickListener(v -> {
            if (scanning) stopCamera();
            else if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera();
            else camPerm.launch(Manifest.permission.CAMERA);
        });
        l.addView(scanBtn, lp(-1, -2, 0, 10));
        LinearLayout c = card();
        c.setGravity(Gravity.CENTER_HORIZONTAL);
        c.addView(text("Last plate", 12, MUTED, false));
        lastText = text("—", 26, Color.BLACK, true);
        lastText.setGravity(Gravity.CENTER);
        lastEn = text("", 13, MUTED, false);
        lastInfo = text("", 12, MUTED, false);
        lastInfo.setGravity(Gravity.CENTER);
        c.addView(lastText); c.addView(lastEn); c.addView(lastInfo);
        l.addView(c, lp(-1, -2, 0, 10));
        return l;
    }

    void startCamera() {
        if (lpr == null) { toast("The model is still loading…"); return; }
        ListenableFuture<ProcessCameraProvider> f = ProcessCameraProvider.getInstance(this);
        f.addListener(() -> {
            try {
                camProvider = f.get();
                ResolutionSelector rs = new ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(new ResolutionStrategy(new Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .build();
                Preview pv = new Preview.Builder().setResolutionSelector(rs).build();
                pv.setSurfaceProvider(preview.getSurfaceProvider());
                ImageAnalysis an = new ImageAnalysis.Builder().setResolutionSelector(rs)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build();
                an.setAnalyzer(worker, this::analyze);
                camProvider.unbindAll();
                camProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, pv, an);
                scanning = true;
                scanBtn.setText("Stop");
                scanBtn.setBackground(round(RED, 0, 14));
            } catch (Exception e) { toast("Camera: " + e.getMessage()); }
        }, ContextCompat.getMainExecutor(this));
    }

    void stopCamera() {
        scanning = false;
        if (camProvider != null) camProvider.unbindAll();
        overlay.show(new ArrayList<>(), 1, 1);
        scanBtn.setText("Start scanning");
        scanBtn.setBackground(round(GREEN, 0, 14));
    }

    void analyze(ImageProxy proxy) {
        try {
            if (!scanning || lpr == null) return;
            Bitmap bm = proxy.toBitmap();
            int rot = proxy.getImageInfo().getRotationDegrees();
            if (rot != 0) {
                Matrix m = new Matrix();
                m.postRotate(rot);
                bm = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), m, true);
            }
            int w = bm.getWidth(), h = bm.getHeight();
            int[] px = new int[w * h];
            bm.getPixels(px, 0, w, 0, 0, w, h);
            Pipeline.Result r = lpr.process(Img.fromArgb(px, w, h));
            List<Overlay.Item> items = new ArrayList<>();
            Pipeline.Plate show = null;
            for (Pipeline.Plate p : r.plates) {
                Overlay.Item it = new Overlay.Item();
                it.x1 = p.x1; it.y1 = p.y1; it.x2 = p.x2; it.y2 = p.y2; it.status = p.status;
                it.label = p.text.isEmpty() ? "?" : p.text;
                items.add(it);
                if (show == null && p.valid) show = p;
            }
            overlay.show(items, w, h);
            final Pipeline.Plate sp = show;
            final double ms = r.msTotal;
            runOnUiThread(() -> {
                camStat.setText(String.format(Locale.US, "%s · %.0f ms per frame (%.1f FPS) · %d plate(s)", version.toUpperCase(Locale.US), ms, 1000 / Math.max(1, ms), r.plates.size()));
                if (sp != null) showLast(sp);
            });
            long now = System.currentTimeMillis();
            for (Pipeline.Plate p : r.plates) {
                if (!p.valid) continue;
                String n = Vocab.normText(p.text);
                Long last = recent.get(n);
                if (last != null && now - last < 10_000) continue;            // same plate: store once per 10 s
                recent.put(n, now);
                store.add(version, p.text, n, p.english, p.status, p.conf, "camera", toBitmap(p.crop));
                boolean hit = store.watched(n);
                vibrate(hit ? 600 : 60);
                if (hit) toast("WATCHLIST MATCH: " + p.text);
            }
        } catch (Throwable e) {
            Log.e("BanglaLPR", "frame", e);
            runOnUiThread(() -> camStat.setText("error: " + err(e)));
        } finally {
            proxy.close();
        }
    }

    void showLast(Pipeline.Plate p) {
        lastText.setText(p.text);
        lastEn.setText(p.english + "   ·   " + chip(p.status));
        lastText.setTextColor("verified".equals(p.status) ? GREEN : "likely".equals(p.status) ? Color.rgb(184, 110, 0) : RED);
    }

    void vibrate(long ms) {
        try {
            Vibrator v = (Vibrator) getSystemService(VIBRATOR_SERVICE);
            if (v == null) return;
            if (Build.VERSION.SDK_INT >= 26) v.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE));
            else v.vibrate(ms);
        } catch (Throwable ignored) { }
    }

    static Bitmap toBitmap(Img im) {
        if (im == null || im.w == 0) return null;
        return Bitmap.createBitmap(im.toArgb(), im.w, im.h, Bitmap.Config.ARGB_8888);
    }

    // ------------------------------------------------------------------------------------------ photo
    View buildPhoto() {
        ScrollView sv = new ScrollView(this);
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(dp(12), dp(12), dp(12), dp(12));
        LinearLayout row = new LinearLayout(this);
        Button take = button("Take photo", true), gal = button("Gallery", false);
        take.setOnClickListener(v -> {
            try {
                File dir = new File(getCacheDir(), "photos");
                dir.mkdirs();
                photoUri = FileProvider.getUriForFile(this, getPackageName() + ".files", new File(dir, "shot.jpg"));
                takePicture.launch(photoUri);
            } catch (Exception e) { toast("Camera app: " + e.getMessage()); }
        });
        gal.setOnClickListener(v -> pickImage.launch("image/*"));
        LinearLayout.LayoutParams a = new LinearLayout.LayoutParams(0, -2, 1), bb = new LinearLayout.LayoutParams(0, -2, 1);
        bb.leftMargin = dp(10);
        row.addView(take, a); row.addView(gal, bb);
        l.addView(row);
        photoInfo = text("Photograph a vehicle's number plate, or pick a photo. It is read on this phone and saved to History.", 13, MUTED, false);
        l.addView(photoInfo, lp(-1, -2, 0, 12));
        photoView = new ImageView(this);
        photoView.setAdjustViewBounds(true);
        l.addView(photoView, lp(-1, -2, 0, 10));
        photoPlates = new LinearLayout(this);
        photoPlates.setOrientation(LinearLayout.VERTICAL);
        l.addView(photoPlates, lp(-1, -2, 0, 10));
        sv.addView(l);
        return sv;
    }

    Bitmap decode(Uri uri) throws Exception {
        Bitmap bm;
        if (Build.VERSION.SDK_INT >= 28) {
            bm = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), uri), (d, info, src) -> {
                d.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                int big = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
                if (big > 1600) { float s = 1600f / big; d.setTargetSize(Math.round(info.getSize().getWidth() * s), Math.round(info.getSize().getHeight() * s)); }
            });
        } else {
            try (InputStream in = getContentResolver().openInputStream(uri)) { bm = BitmapFactory.decodeStream(in); }
            int big = Math.max(bm.getWidth(), bm.getHeight());
            if (big > 1600) bm = Bitmap.createScaledBitmap(bm, bm.getWidth() * 1600 / big, bm.getHeight() * 1600 / big, true);
        }
        return bm.copy(Bitmap.Config.ARGB_8888, true);
    }

    void runPhoto(Uri uri, String source) {
        if (lpr == null) { toast("The model is still loading…"); return; }
        photoInfo.setText("Reading with " + version.toUpperCase(Locale.US) + " on this phone…");
        photoPlates.removeAllViews();
        worker.execute(() -> {
            try {
                Bitmap bm = decode(uri);
                int w = bm.getWidth(), h = bm.getHeight();
                int[] px = new int[w * h];
                bm.getPixels(px, 0, w, 0, 0, w, h);
                Pipeline.Result r = lpr.process(Img.fromArgb(px, w, h));
                Canvas c = new Canvas(bm);
                Paint box = new Paint(Paint.ANTI_ALIAS_FLAG), txt = new Paint(Paint.ANTI_ALIAS_FLAG), bg = new Paint();
                box.setStyle(Paint.Style.STROKE);
                box.setStrokeWidth(Math.max(3, w / 300f));
                txt.setColor(Color.WHITE); txt.setTextSize(Math.max(24, w / 28f)); txt.setFakeBoldText(true);
                List<boolean[]> hits = new ArrayList<>();
                for (Pipeline.Plate p : r.plates) {
                    int col = Overlay.color(p.status);
                    box.setColor(col); bg.setColor(col);
                    c.drawRect(p.x1, p.y1, p.x2, p.y2, box);
                    String lab = p.text.isEmpty() ? "?" : p.text;
                    float ty = Math.max(txt.getTextSize(), p.y1 - 8);
                    c.drawRect(p.x1, ty - txt.getTextSize(), p.x1 + txt.measureText(lab) + 12, ty + 8, bg);
                    c.drawText(lab, p.x1 + 6, ty, txt);
                    String n = Vocab.normText(p.text);
                    store.add(version, p.text, n, p.english, p.status, p.conf, source, toBitmap(p.crop));
                    hits.add(new boolean[]{store.watched(n)});
                }
                runOnUiThread(() -> {
                    photoView.setImageBitmap(bm);
                    photoInfo.setText(String.format(Locale.US, "%s · %.0f ms on this phone (detect %.0f + read %.0f)%s · saved to History",
                            version.toUpperCase(Locale.US), r.msTotal, r.msDetect, r.msRead, r.retry != null ? " · retry: " + r.retry : ""));
                    if (r.plates.isEmpty()) photoPlates.addView(text("No plate found. Move closer so the plate fills more of the photo.", 15, Color.BLACK, false));
                    for (int i = 0; i < r.plates.size(); i++) photoPlates.addView(plateCard(r.plates.get(i), hits.get(i)[0]), lp(-1, -2, 0, 8));
                    for (boolean[] hh : hits) if (hh[0]) { vibrate(600); toast("WATCHLIST MATCH"); }
                });
            } catch (Throwable e) {
                Log.e("BanglaLPR", "photo", e);
                runOnUiThread(() -> photoInfo.setText("Error: " + err(e)));
            }
        });
    }

    View plateCard(Pipeline.Plate p, boolean hit) {
        LinearLayout c = card();
        if (hit) c.setBackground(round(Color.rgb(251, 229, 232), RED, 14));
        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.CENTER_VERTICAL);
        ImageView iv = new ImageView(this);
        iv.setImageBitmap(toBitmap(p.crop));
        iv.setAdjustViewBounds(true);
        row.addView(iv, new LinearLayout.LayoutParams(dp(110), -2));
        LinearLayout t = new LinearLayout(this);
        t.setOrientation(LinearLayout.VERTICAL);
        t.setPadding(dp(12), 0, 0, 0);
        TextView big = text(p.text.isEmpty() ? "(unreadable)" : p.text, 22, Color.BLACK, true);
        t.addView(big);
        t.addView(text(p.english, 13, MUTED, false));
        TextView st = text(chip(p.status) + (hit ? "   ·   WATCHLIST MATCH" : ""), 13, Overlay.color(p.status), true);
        t.addView(st);
        row.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
        c.addView(row);
        StringBuilder sb = new StringBuilder("What each reader saw:");
        for (Map.Entry<String, String> e : p.readers.entrySet()) sb.append("\n  ").append(e.getKey()).append(": ").append(e.getValue().isEmpty() ? "—" : e.getValue());
        if (p.charAgreement != null) sb.append(String.format(Locale.US, "\n  character CNN agreement: %.0f %%", 100 * p.charAgreement));
        c.addView(text(sb.toString(), 12, MUTED, false), lp(-1, -2, 0, 8));
        return c;
    }

    // ------------------------------------------------------------------------------------------ history + watchlist
    View buildHistory() {
        ScrollView sv = new ScrollView(this);
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(dp(12), dp(12), dp(12), dp(12));
        histQ = new EditText(this);
        histQ.setHint("Search plate (Bangla or English)");
        histQ.setSingleLine(true);
        histQ.addTextChangedListener(new TextWatcher() {
            public void beforeTextChanged(CharSequence s, int a, int b, int c) { }
            public void onTextChanged(CharSequence s, int a, int b, int c) { }
            public void afterTextChanged(Editable s) { loadHistory(); }
        });
        l.addView(histQ);
        histCount = text("", 12, MUTED, false);
        l.addView(histCount);
        LinearLayout row = new LinearLayout(this);
        Button exp = button("Export CSV", false), clr = button("Delete all", false);
        clr.setTextColor(RED);
        exp.setOnClickListener(v -> exportCsv());
        clr.setOnClickListener(v -> new android.app.AlertDialog.Builder(this).setMessage("Delete all stored plates and crops on this phone?")
                .setPositiveButton("Delete", (d, w) -> { store.clear(); loadHistory(); }).setNegativeButton("Cancel", null).show());
        LinearLayout.LayoutParams a = new LinearLayout.LayoutParams(0, -2, 1), b = new LinearLayout.LayoutParams(0, -2, 1);
        b.leftMargin = dp(10);
        row.addView(exp, a); row.addView(clr, b);
        l.addView(row, lp(-1, -2, 0, 8));
        histList = new LinearLayout(this);
        histList.setOrientation(LinearLayout.VERTICAL);
        l.addView(histList, lp(-1, -2, 0, 8));
        // watchlist
        l.addView(text("Watchlist (stolen / wanted plates: a match vibrates and is shown in red)", 14, Color.BLACK, true), lp(-1, -2, 0, 18));
        EditText wt = new EditText(this);
        wt.setHint("Plate text, e.g. ঢাকা মেট্রো-গ ১২-৩৪৫৬");
        l.addView(wt);
        Button add = button("Add to watchlist", false);
        add.setOnClickListener(v -> {
            String s = wt.getText().toString().trim();
            if (Vocab.normText(s).isEmpty()) return;
            store.watchAdd(Vocab.normText(s), s, "");
            wt.setText("");
            loadHistory();
        });
        l.addView(add, lp(-1, -2, 0, 6));
        watchListView = new LinearLayout(this);
        watchListView.setOrientation(LinearLayout.VERTICAL);
        l.addView(watchListView, lp(-1, -2, 0, 6));
        sv.addView(l);
        return sv;
    }

    void loadHistory() {
        if (histList == null) return;
        histList.removeAllViews();
        List<Store.Row> rows = store.list(Vocab.normText(histQ.getText().toString()), 200);
        histCount.setText(store.count() + " plates stored on this phone (showing up to 200)");
        java.text.DateFormat fmt = java.text.DateFormat.getDateTimeInstance(java.text.DateFormat.SHORT, java.text.DateFormat.SHORT);
        for (Store.Row r : rows) {
            LinearLayout c = card();
            if (r.watch > 0) c.setBackground(round(Color.rgb(251, 229, 232), RED, 14));
            LinearLayout row = new LinearLayout(this);
            row.setGravity(Gravity.CENTER_VERTICAL);
            if (r.crop != null) {
                ImageView iv = new ImageView(this);
                iv.setImageBitmap(BitmapFactory.decodeFile(r.crop));
                iv.setAdjustViewBounds(true);
                row.addView(iv, new LinearLayout.LayoutParams(dp(90), -2));
            }
            LinearLayout t = new LinearLayout(this);
            t.setOrientation(LinearLayout.VERTICAL);
            t.setPadding(dp(10), 0, 0, 0);
            t.addView(text(r.text == null || r.text.isEmpty() ? "—" : r.text, 18, Color.BLACK, true));
            t.addView(text(r.english, 12, MUTED, false));
            t.addView(text(chip(r.status) + "  ·  " + r.version + "  ·  " + r.source + "  ·  " + fmt.format(new java.util.Date(r.ts))
                    + (r.watch > 0 ? "  ·  WATCHLIST" : ""), 11, Overlay.color(r.status), true));
            row.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
            c.addView(row);
            histList.addView(c, lp(-1, -2, 0, 6));
        }
        watchListView.removeAllViews();
        for (String[] wl : store.watchList()) {
            LinearLayout row = new LinearLayout(this);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.addView(text(wl[1], 16, RED, true), new LinearLayout.LayoutParams(0, -2, 1));
            Button del = new Button(this);
            del.setText("Remove"); del.setAllCaps(false);
            del.setOnClickListener(v -> { store.watchDel(wl[0]); loadHistory(); });
            row.addView(del);
            watchListView.addView(row);
        }
    }

    void exportCsv() {
        try {
            File f = store.exportCsv(new File(getFilesDir(), "exports"));
            Uri u = FileProvider.getUriForFile(this, getPackageName() + ".files", f);
            Intent i = new Intent(Intent.ACTION_SEND).setType("text/csv").putExtra(Intent.EXTRA_STREAM, u).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(i, "Share results CSV"));
        } catch (Exception e) { toast("Export failed: " + e.getMessage()); }
    }

    // ------------------------------------------------------------------------------------------ about
    View buildAbout() {
        ScrollView sv = new ScrollView(this);
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setPadding(dp(12), dp(12), dp(12), dp(12));
        boolean v1 = version.equals("v1");
        String body = v1
                ? "BanglaLPR V1 — Bangla cascade\n\n"
                + "• Plate detector: YOLOv8n-576 (Embedded LPR study)\n"
                + "• Main reader: character YOLOv8n (Video LPR study)\n"
                + "• Fallback when the BRTA layout breaks: LPNR YOLO12n (Cascaded YOLO) + slot OCR (Embedded LPR)\n"
                + "• Second opinion: CRNN (Plate SR study); character check: two CNNs (CNN + GFP-GAN, BLPnet)\n\n"
                + "Measured on the team PC (dataset v1.1 test): 89.22 % of 204 whole test photos read exactly, character accuracy 94.74 %; "
                + "90.02 % of 651 test plate crops."
                : "BanglaLPR V2 — V1 upgraded with eight more papers\n\n"
                + "• Plate detector: YOLOv8n-576 (from V1); if nothing is found: YOLOv8s (YOLOv8 + EasyOCR study), then on a CLAHE-brightened copy (HVI-CIDNet study)\n"
                + "• Two readers always: character YOLOv8n (Video LPR) + character YOLOv8n (YOLOv10 run)\n"
                + "• When they disagree: LPNR YOLO12n + SVTRv2-S + SVTRv2-T vote (SVTRv2 study)\n"
                + "• Status from reader agreement (LPLCv2 study); chosen on validation, tested once (Near-Duplicates study)\n\n"
                + "Measured on the team PC (dataset v1.1 test): 92.65 % of 204 whole test photos read exactly, character accuracy 96.79 %; "
                + "91.71 % of 651 test plate crops.";
        body += "\n\nStatus: VERIFIED = two readers agree on a valid plate; LIKELY = valid plate, partial agreement; REVIEW = a person should check.\n\n"
                + "Everything runs on this phone with ONNX Runtime. The app has no internet permission: photos, plates and history never leave the phone. "
                + "Speed on a phone depends on its processor; the Scan screen shows the time per frame.\n\n"
                + "NSU CSE 498R · Md. Yousuf · MD. Rokib Hasan Oli · Tashfia Adiba Zaman Nizhum · Md Rakibul Hasan";
        LinearLayout c = card();
        c.addView(text(body, 14, Color.BLACK, false));
        l.addView(c);
        sv.addView(l);
        return sv;
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        scanning = false;
        worker.shutdown();
    }
}
