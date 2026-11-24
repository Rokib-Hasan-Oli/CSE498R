package bd.nsu.banglalpr.app;

import android.Manifest;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

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
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import bd.nsu.banglalpr.core.Img;
import bd.nsu.banglalpr.core.Net;
import bd.nsu.banglalpr.core.Pipeline;
import bd.nsu.banglalpr.core.Vocab;

/**
 * BanglaLPR V3 on the phone, fully offline. Same reading engine and models as V2 (checked on the PC: same text as the
 * Python system). New in V3: every plate is saved as a plate picture + a row in plates.csv (Records), one-tap export
 * (zip with CSV + pictures + a printable sheet, copied to Download/BanglaLPR) and a simpler, friendlier screen.
 */
public class V3Activity extends ComponentActivity {
    static final int GREEN = Color.rgb(0, 106, 78), GREEN_SOFT = Color.rgb(226, 242, 234), DARK = Color.rgb(13, 59, 44),
            BG = Color.rgb(243, 246, 244), INK = Color.rgb(20, 32, 25), MUTED = Color.rgb(91, 106, 98),
            LINE = Color.rgb(225, 231, 227), RED = Color.rgb(211, 36, 58), RED_SOFT = Color.rgb(253, 232, 235),
            AMBER = Color.rgb(154, 91, 0), AMBER_SOFT = Color.rgb(255, 242, 217), WHITE = Color.WHITE;
    static final String ENGINE = "v2";          // V3 = the V2 engine (no new training) + records + new screens

    final ExecutorService worker = Executors.newSingleThreadExecutor();
    volatile Pipeline lpr;
    Records rec;
    FrameLayout content;
    final Map<String, View> panels = new LinkedHashMap<>();
    final Map<String, LinearLayout> navItems = new HashMap<>();
    TextView engineState;
    // night mode (22 Nov 2026): Auto = 18:00-06:00 by the phone clock, or On / Off; the photo alone cannot tell reliably
    TextView nightBtn; String nightPref = "auto";
    volatile androidx.camera.core.Camera camera; Button torchBtn; boolean torchOn;

    // photo
    ImageView photoView; LinearLayout photoPlates; TextView photoInfo; Uri photoUri; View photoCard;
    // scan
    PreviewView preview; Overlay overlay; TextView camStat, lastText, lastEn, sessionCount; ImageView lastCrop; LinearLayout lastChips;
    Button scanBtn; View camHint;
    ProcessCameraProvider camProvider; volatile boolean scanning;
    final Map<String, Long> recent = new HashMap<>();
    int sessionSaved = 0;
    // records
    LinearLayout recList, watchListView, statsRow, filterRow, exportBox; EditText recQ; String recStatus = "";

    ActivityResultLauncher<String> camPerm, pickImage;
    ActivityResultLauncher<Uri> takePicture;
    Uri pendingShare;

    int dp(float v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        if (Build.VERSION.SDK_INT >= 21) { getWindow().setStatusBarColor(DARK); getWindow().setNavigationBarColor(WHITE); }
        rec = new Records(this);
        nightPref = getSharedPreferences("lpr", MODE_PRIVATE).getString("night", "auto");
        camPerm = registerForActivityResult(new ActivityResultContracts.RequestPermission(), ok -> {
            if (ok) startCamera(); else toast("Allow the camera to use live scanning");
        });
        pickImage = registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> { if (uri != null) runPhoto(uri, "gallery", "gallery photo"); });
        takePicture = registerForActivityResult(new ActivityResultContracts.TakePicture(), ok -> { if (ok && photoUri != null) runPhoto(photoUri, "photo", "camera photo"); });
        setContentView(buildUi());
        showTab("photo");
        handleShare(getIntent());
        worker.execute(() -> {
            try {
                Net.threads = Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors()));
                Net.xnnpack = false;     // plain CPU = the engine checked on the PC (XNNPACK broke dynamic batches on a test device)
                long t0 = System.nanoTime();
                Pipeline p = new Pipeline(ENGINE, name -> { try (InputStream in = getAssets().open(name)) { return MainActivity.readAll(in); } });
                lpr = p;
                double s = (System.nanoTime() - t0) / 1e9;
                runOnUiThread(() -> {
                    setEngine(true, String.format(Locale.US, "Ready (%.1f s)", s));
                    if (pendingShare != null) { Uri u = pendingShare; pendingShare = null; showTab("photo"); runPhoto(u, "shared", "shared photo"); }
                });
            } catch (Throwable e) {
                Log.e("BanglaLPR", "model load", e);
                runOnUiThread(() -> { setEngine(false, "Model error"); photoInfo.setText("The model failed to load: " + MainActivity.err(e)); });
            }
        });
    }

    boolean nightOn() {
        if (nightPref.equals("on")) return true;
        if (nightPref.equals("off")) return false;
        int h = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY);
        return h >= 18 || h < 6;
    }

    void showNight() {
        if (nightBtn == null) return;
        boolean on = nightOn();
        String mode = nightPref.equals("auto") ? "Auto" : nightPref.equals("on") ? "On" : "Off";
        nightBtn.setText("☾ Night: " + mode + (nightPref.equals("auto") ? (on ? " (on)" : " (off)") : ""));
        nightBtn.setTextColor(on ? Color.rgb(207, 224, 255) : Color.rgb(167, 196, 182));
        nightBtn.setBackground(round(on ? Color.rgb(31, 42, 68) : Color.argb(30, 255, 255, 255), 0, 99));
        if (torchBtn != null) torchBtn.setVisibility(scanning && camera != null && camera.getCameraInfo().hasFlashUnit() ? View.VISIBLE : View.GONE);
    }

    void cycleNight() {
        nightPref = nightPref.equals("auto") ? "on" : nightPref.equals("on") ? "off" : "auto";
        getSharedPreferences("lpr", MODE_PRIVATE).edit().putString("night", nightPref).apply();
        showNight();
        toast(nightOn() ? "Night mode on: plates are read with the night settings" : "Night mode off");
    }

    void setTorch(boolean on) {
        if (camera == null || !camera.getCameraInfo().hasFlashUnit()) return;
        camera.getCameraControl().enableTorch(on);
        torchOn = on;
        if (torchBtn != null) torchBtn.setText(on ? "Light off" : "Light on");
    }

    TextView nightChip() { return chip("☾ Night mode", Color.rgb(207, 224, 255), Color.rgb(31, 42, 68)); }

    void handleShare(Intent i) {
        if (i == null || !Intent.ACTION_SEND.equals(i.getAction())) return;
        Uri u = i.getParcelableExtra(Intent.EXTRA_STREAM);
        if (u == null) return;
        if (lpr == null) { pendingShare = u; return; }
        showTab("photo");
        runPhoto(u, "shared", "shared photo");
    }

    @Override protected void onNewIntent(Intent i) { super.onNewIntent(i); handleShare(i); }

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
        if (stroke != 0) g.setStroke(dp(1), stroke);
        return g;
    }

    Button button(String s, int style) {          // 0 primary, 1 outline, 2 text, 3 danger outline
        Button bt = new Button(this);
        bt.setText(s); bt.setAllCaps(false); bt.setTextSize(15); bt.setTypeface(Typeface.DEFAULT_BOLD);
        bt.setStateListAnimator(null);
        bt.setMinHeight(dp(48)); bt.setMinimumHeight(dp(48));
        if (style == 0) { bt.setTextColor(WHITE); bt.setBackground(round(GREEN, 0, 12)); }
        else if (style == 1) { bt.setTextColor(GREEN); bt.setBackground(round(WHITE, Color.rgb(196, 210, 202), 12)); }
        else if (style == 3) { bt.setTextColor(RED); bt.setBackground(round(WHITE, Color.rgb(240, 196, 202), 12)); }
        else { bt.setTextColor(MUTED); bt.setBackground(null); }
        bt.setPadding(dp(14), dp(10), dp(14), dp(10));
        return bt;
    }

    LinearLayout card() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setBackground(round(WHITE, LINE, 16));
        l.setPadding(dp(16), dp(14), dp(16), dp(14));
        return l;
    }

    LinearLayout row() {
        LinearLayout r = new LinearLayout(this);
        r.setOrientation(LinearLayout.HORIZONTAL);
        r.setGravity(Gravity.CENTER_VERTICAL);
        return r;
    }

    LinearLayout.LayoutParams lp(int w, int h, float weight, int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(w, h, weight);
        p.topMargin = dp(top);
        return p;
    }

    LinearLayout.LayoutParams side(int left) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, -2, 1);
        p.leftMargin = dp(left);
        return p;
    }

    void toast(String s) { runOnUiThread(() -> Toast.makeText(this, s, Toast.LENGTH_LONG).show()); }

    static String statusName(String s) {
        return "verified".equals(s) ? "Verified" : "likely".equals(s) ? "Likely" : "review".equals(s) ? "Check manually" : (s == null ? "" : s);
    }

    TextView chip(String label, int fg, int bg) {
        TextView t = text(label, 12, fg, true);
        t.setBackground(round(bg, 0, 99));
        t.setPadding(dp(10), dp(3), dp(10), dp(4));
        return t;
    }

    TextView statusChip(String s) {
        if ("verified".equals(s)) return chip("✓ Verified", GREEN, GREEN_SOFT);
        if ("likely".equals(s)) return chip("Likely", AMBER, AMBER_SOFT);
        return chip("Check manually", RED, RED_SOFT);
    }

    LinearLayout chips(View... vs) {
        LinearLayout r = row();
        for (View v : vs) {
            if (v == null) continue;
            LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-2, -2);
            p.rightMargin = dp(6);
            r.addView(v, p);
        }
        return r;
    }

    TextView sectionTitle(String s) {
        TextView t = text(s, 17, INK, true);
        t.setPadding(0, 0, 0, dp(2));
        return t;
    }

    void copy(String s) {
        ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (cm != null) { cm.setPrimaryClip(ClipData.newPlainText("plate", s)); toast("Copied: " + s); }
    }

    void setEngine(boolean ok, String s) {
        engineState.setText("● " + s);
        engineState.setTextColor(ok ? Color.rgb(60, 207, 145) : Color.rgb(255, 191, 77));
    }

    /** simple line icons drawn in code (no image files needed) */
    final class Icon extends View {
        final String kind; int color = MUTED;
        final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        Icon(String kind) { super(V3Activity.this); this.kind = kind; p.setStyle(Paint.Style.STROKE); p.setStrokeCap(Paint.Cap.ROUND); p.setStrokeJoin(Paint.Join.ROUND); }
        void setColor(int c) { color = c; invalidate(); }
        @Override protected void onDraw(Canvas c) {
            float s = Math.min(getWidth(), getHeight()) / 24f, ox = (getWidth() - 24 * s) / 2, oy = (getHeight() - 24 * s) / 2;
            c.save(); c.translate(ox, oy); c.scale(s, s);
            p.setColor(color); p.setStrokeWidth(1.9f); p.setStyle(Paint.Style.STROKE);
            Path path = new Path();
            switch (kind) {
                case "photo":
                    path.moveTo(4, 7); path.lineTo(7, 7); path.lineTo(9, 4.5f); path.lineTo(15, 4.5f); path.lineTo(17, 7); path.lineTo(20, 7);
                    path.lineTo(20, 19); path.lineTo(4, 19); path.close();
                    c.drawPath(path, p); c.drawCircle(12, 13, 3.6f, p); break;
                case "scan":
                    float[][] cs = {{4, 8, 4, 4, 8, 4}, {16, 4, 20, 4, 20, 8}, {20, 16, 20, 20, 16, 20}, {8, 20, 4, 20, 4, 16}};
                    for (float[] k : cs) { Path q = new Path(); q.moveTo(k[0], k[1]); q.lineTo(k[2], k[3]); q.lineTo(k[4], k[5]); c.drawPath(q, p); }
                    c.drawRect(7.5f, 9.5f, 16.5f, 14.5f, p); break;
                case "records":
                    c.drawRoundRect(new RectF(4, 4, 20, 20), 2.5f, 2.5f, p);
                    c.drawLine(8, 9, 16, 9, p); c.drawLine(8, 12.5f, 16, 12.5f, p); c.drawLine(8, 16, 13, 16, p); break;
                default:
                    c.drawCircle(12, 12, 8.5f, p); c.drawLine(12, 11, 12, 16.5f, p);
                    p.setStyle(Paint.Style.FILL); c.drawCircle(12, 7.8f, 1.2f, p); break;
            }
            c.restore();
        }
    }

    // ------------------------------------------------------------------------------------------ layout
    View buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        // header
        LinearLayout bar = row();
        bar.setBackgroundColor(DARK);
        bar.setPadding(dp(18), dp(14), dp(14), dp(14));
        LinearLayout titles = new LinearLayout(this);
        titles.setOrientation(LinearLayout.VERTICAL);
        LinearLayout tr = row();
        tr.addView(text("BanglaLPR", 21, WHITE, true));
        TextView v3 = chip("V3", Color.rgb(6, 33, 24), Color.rgb(60, 207, 145));
        LinearLayout.LayoutParams vp = new LinearLayout.LayoutParams(-2, -2);
        vp.leftMargin = dp(8);
        tr.addView(v3, vp);
        titles.addView(tr);
        titles.addView(text("Bangla number plate reader · offline", 12.5f, Color.rgb(167, 196, 182), false));
        bar.addView(titles, new LinearLayout.LayoutParams(0, -2, 1));
        engineState = text("● Loading model…", 12.5f, Color.rgb(255, 191, 77), true);
        engineState.setBackground(round(Color.argb(30, 255, 255, 255), 0, 99));
        engineState.setPadding(dp(10), dp(5), dp(10), dp(5));
        nightBtn = text("", 12.5f, WHITE, true);
        nightBtn.setPadding(dp(10), dp(5), dp(10), dp(5));
        nightBtn.setOnClickListener(v -> cycleNight());
        LinearLayout.LayoutParams np = new LinearLayout.LayoutParams(-2, -2);
        np.rightMargin = dp(6);
        bar.addView(nightBtn, np);
        showNight();
        bar.addView(engineState);
        root.addView(bar);

        content = new FrameLayout(this);
        root.addView(content, new LinearLayout.LayoutParams(-1, 0, 1));
        panels.put("photo", buildPhoto()); panels.put("scan", buildScan()); panels.put("records", buildRecords()); panels.put("help", buildHelp());
        for (View v : panels.values()) content.addView(v, new FrameLayout.LayoutParams(-1, -1));

        // bottom navigation
        View divider = new View(this);
        divider.setBackgroundColor(LINE);
        root.addView(divider, new LinearLayout.LayoutParams(-1, dp(1)));
        LinearLayout nav = row();
        nav.setBackgroundColor(WHITE);
        String[][] items = {{"photo", "Photo"}, {"scan", "Live scan"}, {"records", "Records"}, {"help", "Help"}};
        for (String[] it : items) {
            LinearLayout b = new LinearLayout(this);
            b.setOrientation(LinearLayout.VERTICAL);
            b.setGravity(Gravity.CENTER);
            b.setPadding(0, dp(6), 0, dp(6));
            Icon ic = new Icon(it[0]);
            b.addView(ic, new LinearLayout.LayoutParams(dp(26), dp(26)));
            TextView lab = text(it[1], 12, MUTED, false);
            lab.setGravity(Gravity.CENTER);
            b.addView(lab);
            b.setTag(it[0]);
            b.setOnClickListener(v -> showTab((String) v.getTag()));
            navItems.put(it[0], b);
            nav.addView(b, new LinearLayout.LayoutParams(0, dp(62), 1));
        }
        root.addView(nav);
        return root;
    }

    void showTab(String key) {
        for (Map.Entry<String, View> e : panels.entrySet()) e.getValue().setVisibility(e.getKey().equals(key) ? View.VISIBLE : View.GONE);
        for (Map.Entry<String, LinearLayout> e : navItems.entrySet()) {
            boolean on = e.getKey().equals(key);
            LinearLayout b = e.getValue();
            ((Icon) b.getChildAt(0)).setColor(on ? GREEN : MUTED);
            TextView t = (TextView) b.getChildAt(1);
            t.setTextColor(on ? GREEN : MUTED);
            t.setTypeface(on ? Typeface.DEFAULT_BOLD : Typeface.DEFAULT);
            b.setBackground(on ? round(GREEN_SOFT, 0, 14) : null);
        }
        if (key.equals("records")) loadRecords();
        if (!key.equals("scan") && scanning) stopCamera();
    }

    ScrollView scroller(LinearLayout l) {
        ScrollView sv = new ScrollView(this);
        sv.setFillViewport(true);
        l.setPadding(dp(14), dp(14), dp(14), dp(24));
        sv.addView(l);
        return sv;
    }

    LinearLayout vertical() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    // ------------------------------------------------------------------------------------------ photo
    View buildPhoto() {
        LinearLayout l = vertical();
        LinearLayout hero = card();
        hero.setGravity(Gravity.CENTER_HORIZONTAL);
        hero.setPadding(dp(18), dp(22), dp(18), dp(18));
        Icon ic = new Icon("photo");
        ic.setColor(GREEN);
        ic.setBackground(round(GREEN_SOFT, 0, 16));
        hero.addView(ic, new LinearLayout.LayoutParams(dp(58), dp(58)));
        TextView h = text("Read a number plate", 20, INK, true);
        h.setGravity(Gravity.CENTER);
        hero.addView(h, lp(-2, -2, 0, 12));
        TextView sub = text("Take a photo of a vehicle or pick one from your gallery. The plate is read on this phone.", 14, MUTED, false);
        sub.setGravity(Gravity.CENTER);
        hero.addView(sub, lp(-1, -2, 0, 4));
        Button take = button("Take a photo", 0), gal = button("Choose from gallery", 1);
        take.setOnClickListener(v -> {
            try {
                File dir = new File(getCacheDir(), "photos");
                dir.mkdirs();
                photoUri = FileProvider.getUriForFile(this, getPackageName() + ".files", new File(dir, "shot.jpg"));
                takePicture.launch(photoUri);
            } catch (Exception e) { toast("Camera app: " + e.getMessage()); }
        });
        gal.setOnClickListener(v -> pickImage.launch("image/*"));
        hero.addView(take, lp(-1, -2, 0, 16));
        hero.addView(gal, lp(-1, -2, 0, 8));
        TextView note = text("Each plate found is saved: plate picture + one row in plates.csv (see Records).", 12.5f, GREEN, false);
        note.setGravity(Gravity.CENTER);
        hero.addView(note, lp(-1, -2, 0, 12));
        l.addView(hero);

        photoInfo = text("", 13, MUTED, false);
        photoInfo.setVisibility(View.GONE);
        l.addView(photoInfo, lp(-1, -2, 0, 12));
        LinearLayout pc = card();
        pc.setPadding(dp(6), dp(6), dp(6), dp(6));
        photoView = new ImageView(this);
        photoView.setAdjustViewBounds(true);
        photoView.setClipToOutline(true);
        photoView.setBackground(round(Color.rgb(11, 21, 16), 0, 12));
        pc.addView(photoView, new LinearLayout.LayoutParams(-1, -2));
        pc.setVisibility(View.GONE);
        photoCard = pc;
        l.addView(pc, lp(-1, -2, 0, 10));
        photoPlates = vertical();
        l.addView(photoPlates, lp(-1, -2, 0, 4));
        return scroller(l);
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

    void runPhoto(Uri uri, String source, String filename) {
        if (lpr == null) { toast("The model is still loading, try again in a few seconds"); return; }
        photoInfo.setVisibility(View.VISIBLE);
        photoInfo.setText("Reading the photo on this phone…");
        photoPlates.removeAllViews();
        final boolean night = nightOn();
        worker.execute(() -> {
            try {
                Bitmap bm = decode(uri);
                int w = bm.getWidth(), h = bm.getHeight();
                int[] px = new int[w * h];
                bm.getPixels(px, 0, w, 0, 0, w, h);
                Pipeline.Result r = lpr.process(Img.fromArgb(px, w, h), night);
                Canvas c = new Canvas(bm);
                Paint box = new Paint(Paint.ANTI_ALIAS_FLAG), txt = new Paint(Paint.ANTI_ALIAS_FLAG), bg = new Paint();
                box.setStyle(Paint.Style.STROKE);
                box.setStrokeWidth(Math.max(3, w / 300f));
                txt.setColor(WHITE); txt.setTextSize(Math.max(24, w / 28f)); txt.setFakeBoldText(true);
                List<Records.Row> saved = new ArrayList<>();
                for (Pipeline.Plate p : r.plates) {
                    int col = Overlay.color(p.status);
                    box.setColor(col); bg.setColor(col);
                    c.drawRect(p.x1, p.y1, p.x2, p.y2, box);
                    String lab = p.text.isEmpty() ? "?" : p.text;
                    float ty = Math.max(txt.getTextSize(), p.y1 - 8);
                    c.drawRect(p.x1, ty - txt.getTextSize(), p.x1 + txt.measureText(lab) + 12, ty + 8, bg);
                    c.drawText(lab, p.x1 + 6, ty, txt);
                    saved.add(rec.add(p.text, Vocab.normText(p.text), p.english, p.status, p.valid, p.conf, p.detConf, source, filename, MainActivity.toBitmap(p.crop), r.night));
                }
                runOnUiThread(() -> {
                    photoCard.setVisibility(View.VISIBLE);
                    photoView.setImageBitmap(bm);
                    photoInfo.setText(r.plates.isEmpty() ? String.format(Locale.US, "Read in %.0f ms", r.msTotal)
                            : String.format(Locale.US, "%d plate%s found in %.0f ms · saved to Records", r.plates.size(), r.plates.size() == 1 ? "" : "s", r.msTotal)
                            + (r.night ? " · night mode" : ""));
                    if (r.plates.isEmpty()) {
                        LinearLayout c2 = card();
                        c2.addView(text("No plate found", 17, INK, true));
                        c2.addView(text("Move closer so the plate fills more of the photo, and keep it in focus.", 14, MUTED, false));
                        photoPlates.addView(c2, lp(-1, -2, 0, 10));
                    }
                    boolean anyHit = false;
                    for (int i = 0; i < r.plates.size(); i++) {
                        photoPlates.addView(plateCard(r.plates.get(i), saved.get(i)), lp(-1, -2, 0, 10));
                        anyHit |= saved.get(i).watch > 0;
                    }
                    if (anyHit) { vibrate(600); toast("WATCHLIST MATCH"); }
                });
            } catch (Throwable e) {
                Log.e("BanglaLPR", "photo", e);
                runOnUiThread(() -> photoInfo.setText("Error: " + MainActivity.err(e)));
            }
        });
    }

    View plateCard(Pipeline.Plate p, Records.Row saved) {
        boolean hit = saved != null && saved.watch > 0;
        LinearLayout c = card();
        if (hit) c.setBackground(round(RED_SOFT, RED, 16));
        ImageView iv = new ImageView(this);
        iv.setImageBitmap(MainActivity.toBitmap(p.crop));
        iv.setScaleType(ImageView.ScaleType.FIT_CENTER);
        iv.setBackground(round(Color.rgb(17, 17, 17), 0, 10));
        iv.setClipToOutline(true);
        c.addView(iv, new LinearLayout.LayoutParams(-1, dp(104)));
        LinearLayout tr = row();
        LinearLayout tt = vertical();
        tt.addView(text(p.text.isEmpty() ? "(could not read)" : p.text, 25, INK, true));
        TextView en = text(p.english, 13, MUTED, false);
        en.setTypeface(Typeface.MONOSPACE);
        tt.addView(en);
        tr.addView(tt, new LinearLayout.LayoutParams(0, -2, 1));
        if (!p.text.isEmpty()) {
            Button cp = button("Copy", 1);
            cp.setTextSize(13); cp.setMinHeight(dp(38)); cp.setMinimumHeight(dp(38));
            cp.setOnClickListener(v -> copy(p.text));
            tr.addView(cp);
        }
        c.addView(tr, lp(-1, -2, 0, 10));
        c.addView(chips(statusChip(p.status), p.valid ? chip("BRTA layout ✓", MUTED, BG) : chip("layout not valid", RED, RED_SOFT),
                hit ? chip("WATCHLIST", WHITE, RED) : null, p.night ? nightChip() : null), lp(-1, -2, 0, 8));
        if (saved != null) {
            TextView sv = text("✓ Saved as record #" + saved.id + (saved.crop != null ? "  ·  " + new File(saved.crop).getName() : ""), 12, GREEN, true);
            sv.setSingleLine(true); sv.setEllipsize(TextUtils.TruncateAt.END);
            c.addView(sv, lp(-1, -2, 0, 10));
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : p.readers.entrySet()) sb.append(sb.length() == 0 ? "" : "\n").append(e.getKey()).append(":  ").append(e.getValue().isEmpty() ? "—" : e.getValue());
        if (sb.length() > 0) {
            TextView more = text("What each reader saw  ▾", 13, MUTED, false);
            TextView det = text(sb.toString(), 13, MUTED, false);
            det.setVisibility(View.GONE);
            det.setPadding(dp(10), dp(6), 0, 0);
            more.setPadding(0, dp(8), 0, dp(2));
            more.setOnClickListener(v -> {
                boolean open = det.getVisibility() == View.VISIBLE;
                det.setVisibility(open ? View.GONE : View.VISIBLE);
                more.setText(open ? "What each reader saw  ▾" : "What each reader saw  ▴");
            });
            c.addView(more); c.addView(det);
        }
        return c;
    }

    // ------------------------------------------------------------------------------------------ live scan
    View buildScan() {
        LinearLayout l = vertical();
        l.setPadding(dp(14), dp(14), dp(14), dp(14));
        FrameLayout cam = new FrameLayout(this);
        cam.setBackground(round(Color.rgb(11, 21, 16), 0, 16));
        cam.setClipToOutline(true);
        preview = new PreviewView(this);
        preview.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        overlay = new Overlay(this);
        cam.addView(preview, new FrameLayout.LayoutParams(-1, -1));
        cam.addView(overlay, new FrameLayout.LayoutParams(-1, -1));
        LinearLayout hint = vertical();
        hint.setGravity(Gravity.CENTER);
        Icon hi = new Icon("scan");
        hi.setColor(Color.rgb(127, 153, 140));
        hint.addView(hi, new LinearLayout.LayoutParams(dp(54), dp(54)));
        TextView ht = text("Tap Start, then point the back camera at a number plate", 14, Color.rgb(160, 184, 172), false);
        ht.setGravity(Gravity.CENTER);
        ht.setPadding(dp(24), dp(8), dp(24), 0);
        hint.addView(ht);
        camHint = hint;
        cam.addView(hint, new FrameLayout.LayoutParams(-1, -1));
        camStat = text("", 12, WHITE, false);
        camStat.setBackground(round(Color.argb(150, 0, 0, 0), 0, 8));
        camStat.setPadding(dp(8), dp(4), dp(8), dp(4));
        camStat.setVisibility(View.GONE);
        FrameLayout.LayoutParams sp = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.START);
        sp.setMargins(dp(10), dp(10), dp(10), dp(10));
        cam.addView(camStat, sp);
        l.addView(cam, new LinearLayout.LayoutParams(-1, 0, 1));
        scanBtn = button("Start scanning", 0);
        scanBtn.setOnClickListener(v -> {
            if (scanning) stopCamera();
            else if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera();
            else camPerm.launch(Manifest.permission.CAMERA);
        });
        LinearLayout btns = row();
        btns.addView(scanBtn, new LinearLayout.LayoutParams(0, -2, 1));
        torchBtn = button("Light on", 1);
        torchBtn.setVisibility(View.GONE);
        torchBtn.setOnClickListener(v -> setTorch(!torchOn));
        LinearLayout.LayoutParams tp = new LinearLayout.LayoutParams(-2, -2);
        tp.leftMargin = dp(8);
        btns.addView(torchBtn, tp);
        l.addView(btns, lp(-1, -2, 0, 12));
        LinearLayout c = card();
        LinearLayout top = row();
        top.addView(text("Last saved plate", 12.5f, MUTED, true), new LinearLayout.LayoutParams(0, -2, 1));
        sessionCount = text("", 12.5f, MUTED, false);
        top.addView(sessionCount);
        c.addView(top);
        LinearLayout r = row();
        lastCrop = new ImageView(this);
        lastCrop.setAdjustViewBounds(true);
        lastCrop.setVisibility(View.GONE);
        r.addView(lastCrop, new LinearLayout.LayoutParams(dp(110), -2));
        LinearLayout t = vertical();
        t.setPadding(dp(12), 0, 0, 0);
        lastText = text("—", 22, INK, true);
        lastEn = text("A new readable plate is saved once per 10 seconds; the phone vibrates.", 12.5f, MUTED, false);
        lastChips = row();
        t.addView(lastText); t.addView(lastEn); t.addView(lastChips, lp(-2, -2, 0, 4));
        r.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
        c.addView(r, lp(-1, -2, 0, 6));
        l.addView(c, lp(-1, -2, 0, 12));
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
                camera = camProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, pv, an);
                scanning = true;
                showNight();
                if (nightOn()) setTorch(true);                                   // night: switch the phone light on
                camHint.setVisibility(View.GONE);
                camStat.setVisibility(View.VISIBLE);
                scanBtn.setText("Stop");
                scanBtn.setBackground(round(RED, 0, 12));
            } catch (Exception e) { toast("Camera: " + e.getMessage()); }
        }, ContextCompat.getMainExecutor(this));
    }

    void stopCamera() {
        scanning = false;
        if (camProvider != null) camProvider.unbindAll();
        camera = null; torchOn = false;
        if (torchBtn != null) torchBtn.setText("Light on");
        showNight();
        overlay.show(new ArrayList<>(), 1, 1);
        camHint.setVisibility(View.VISIBLE);
        camStat.setVisibility(View.GONE);
        scanBtn.setText("Start scanning");
        scanBtn.setBackground(round(GREEN, 0, 12));
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
            Pipeline.Result r = lpr.process(Img.fromArgb(px, w, h), nightOn());
            List<Overlay.Item> items = new ArrayList<>();
            for (Pipeline.Plate p : r.plates) {
                Overlay.Item it = new Overlay.Item();
                it.x1 = p.x1; it.y1 = p.y1; it.x2 = p.x2; it.y2 = p.y2; it.status = p.status;
                it.label = p.text.isEmpty() ? "?" : p.text;
                items.add(it);
            }
            overlay.show(items, w, h);
            final double ms = r.msTotal;
            runOnUiThread(() -> camStat.setText(String.format(Locale.US, "%.0f ms per frame · %d plate(s)", ms, r.plates.size())
                    + (r.night ? " · night mode" : "")));
            long now = System.currentTimeMillis();
            for (Pipeline.Plate p : r.plates) {
                if (!p.valid) continue;
                String n = Vocab.normText(p.text);
                Long last = recent.get(n);
                if (last != null && now - last < 10_000) continue;            // same plate: save once per 10 s
                recent.put(n, now);
                Bitmap crop = MainActivity.toBitmap(p.crop);
                Records.Row saved = rec.add(p.text, n, p.english, p.status, true, p.conf, p.detConf, "camera", "live scan", crop, r.night);
                sessionSaved++;
                runOnUiThread(() -> showLast(p, crop, saved));
                vibrate(saved.watch > 0 ? 600 : 60);
                if (saved.watch > 0) toast("WATCHLIST MATCH: " + p.text);
            }
        } catch (Throwable e) {
            Log.e("BanglaLPR", "frame", e);
            runOnUiThread(() -> camStat.setText("error: " + MainActivity.err(e)));
        } finally {
            proxy.close();
        }
    }

    void showLast(Pipeline.Plate p, Bitmap crop, Records.Row saved) {
        lastText.setText(p.text);
        lastEn.setText(p.english + "  ·  record #" + saved.id);
        lastCrop.setImageBitmap(crop);
        lastCrop.setVisibility(crop == null ? View.GONE : View.VISIBLE);
        lastChips.removeAllViews();
        lastChips.addView(statusChip(p.status));
        if (saved.watch > 0) { TextView w = chip("WATCHLIST", WHITE, RED); LinearLayout.LayoutParams q = new LinearLayout.LayoutParams(-2, -2); q.leftMargin = dp(6); lastChips.addView(w, q); }
        sessionCount.setText(sessionSaved + " saved this session");
    }

    void vibrate(long ms) {
        try {
            Vibrator v = (Vibrator) getSystemService(VIBRATOR_SERVICE);
            if (v == null) return;
            if (Build.VERSION.SDK_INT >= 26) v.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE));
            else v.vibrate(ms);
        } catch (Throwable ignored) { }
    }

    // ------------------------------------------------------------------------------------------ records
    View buildRecords() {
        LinearLayout l = vertical();
        statsRow = row();
        l.addView(statsRow);

        LinearLayout files = card();
        files.addView(sectionTitle("Your files"));
        TextView where = text("Plate pictures and plates.csv are saved on this phone in:\n" + rec.folder.getAbsolutePath()
                + "\n(Connect the phone to a PC by USB to copy them, or use Export.)", 12.5f, MUTED, false);
        where.setTextIsSelectable(true);
        files.addView(where, lp(-1, -2, 0, 4));
        Button exp = button("Export CSV + pictures", 0);
        exp.setOnClickListener(v -> doExport());
        files.addView(exp, lp(-1, -2, 0, 12));
        LinearLayout br = row();
        Button share = button("Share CSV", 1), shareZip = button("Share zip", 1);
        share.setOnClickListener(v -> shareFile(rec.csv, "text/csv", "Share plates.csv"));
        shareZip.setOnClickListener(v -> doExportAndShare());
        br.addView(share, new LinearLayout.LayoutParams(0, -2, 1));
        br.addView(shareZip, side(8));
        files.addView(br, lp(-1, -2, 0, 8));
        exportBox = vertical();
        exportBox.setVisibility(View.GONE);
        exportBox.setBackground(round(GREEN_SOFT, 0, 12));
        exportBox.setPadding(dp(12), dp(10), dp(12), dp(10));
        files.addView(exportBox, lp(-1, -2, 0, 10));
        l.addView(files, lp(-1, -2, 0, 12));

        recQ = new EditText(this);
        recQ.setHint("Search a plate (Bangla or English)");
        recQ.setSingleLine(true);
        recQ.setTextSize(15);
        recQ.setBackground(round(WHITE, Color.rgb(207, 217, 211), 12));
        recQ.setPadding(dp(14), dp(12), dp(14), dp(12));
        recQ.addTextChangedListener(new TextWatcher() {
            public void beforeTextChanged(CharSequence s, int a, int b, int c) { }
            public void onTextChanged(CharSequence s, int a, int b, int c) { }
            public void afterTextChanged(Editable s) { loadRecords(); }
        });
        l.addView(recQ, lp(-1, -2, 0, 16));
        HorizontalScrollView hs = new HorizontalScrollView(this);
        hs.setHorizontalScrollBarEnabled(false);
        filterRow = row();
        hs.addView(filterRow);
        l.addView(hs, lp(-1, -2, 0, 10));
        recList = vertical();
        l.addView(recList, lp(-1, -2, 0, 4));

        LinearLayout wc = card();
        wc.addView(sectionTitle("Watchlist"));
        wc.addView(text("Plates to look out for (e.g. a stolen car). A match vibrates, shows in red and is marked in the CSV.", 13, MUTED, false));
        EditText wt = new EditText(this);
        wt.setHint("e.g. ঢাকা মেট্রো-গ ১২-৩৪৫৬");
        wt.setSingleLine(true);
        wt.setBackground(round(WHITE, Color.rgb(207, 217, 211), 12));
        wt.setPadding(dp(14), dp(12), dp(14), dp(12));
        wc.addView(wt, lp(-1, -2, 0, 10));
        Button add = button("Add to watchlist", 1);
        add.setOnClickListener(v -> {
            String s = wt.getText().toString().trim();
            if (Vocab.normText(s).isEmpty()) { toast("Type the plate text first"); return; }
            rec.watchAdd(Vocab.normText(s), s, "");
            wt.setText("");
            loadRecords();
            toast("Added to the watchlist");
        });
        wc.addView(add, lp(-1, -2, 0, 8));
        watchListView = vertical();
        wc.addView(watchListView, lp(-1, -2, 0, 6));
        l.addView(wc, lp(-1, -2, 0, 18));

        Button clr = button("Delete all records", 3);
        clr.setOnClickListener(v -> new AlertDialog.Builder(this).setTitle("Delete all records?")
                .setMessage("All saved plates, plate pictures and the CSV rows on this phone are deleted. Exports already made are kept.")
                .setPositiveButton("Delete", (d, w) -> { rec.clear(); loadRecords(); toast("All records deleted"); })
                .setNegativeButton("Cancel", null).show());
        l.addView(clr, lp(-1, -2, 0, 18));
        return scroller(l);
    }

    View stat(String big, String label, int color) {
        LinearLayout s = vertical();
        s.setBackground(round(WHITE, LINE, 14));
        s.setPadding(dp(10), dp(10), dp(8), dp(10));
        s.addView(text(big, 21, color, true));
        TextView t = text(label, 11.5f, MUTED, false);
        t.setMaxLines(2);
        s.addView(t);
        return s;
    }

    void loadRecords() {
        if (recList == null) return;
        statsRow.removeAllViews();
        String[][] st = {{String.valueOf(rec.count(null)), "saved"}, {String.valueOf(rec.countToday()), "today"},
                {String.valueOf(rec.count("status = 'verified'")), "verified"}, {String.valueOf(rec.count("status = 'review'")), "to check"}};
        int[] cols = {INK, INK, GREEN, AMBER};
        for (int i = 0; i < st.length; i++) statsRow.addView(stat(st[i][0], st[i][1], cols[i]), i == 0 ? new LinearLayout.LayoutParams(0, -2, 1) : side(8));

        filterRow.removeAllViews();
        String[][] fl = {{"", "All"}, {"verified", "Verified"}, {"likely", "Likely"}, {"review", "Check manually"}};
        for (String[] f : fl) {
            boolean on = f[0].equals(recStatus);
            TextView t = text(f[1], 13.5f, on ? WHITE : INK, true);
            t.setBackground(on ? round(GREEN, 0, 99) : round(WHITE, Color.rgb(207, 217, 211), 99));
            t.setPadding(dp(14), dp(7), dp(14), dp(8));
            t.setOnClickListener(v -> { recStatus = f[0]; loadRecords(); });
            LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-2, -2);
            p.rightMargin = dp(8);
            filterRow.addView(t, p);
        }

        recList.removeAllViews();
        List<Records.Row> rows = rec.list(recQ.getText().toString().trim(), recStatus, 200);
        if (rows.isEmpty()) {
            LinearLayout e = card();
            e.setGravity(Gravity.CENTER_HORIZONTAL);
            boolean filtered = !recStatus.isEmpty() || recQ.getText().length() > 0;
            TextView a = text(filtered ? "No records match" : "No records yet", 16, INK, true);
            TextView b = text(filtered ? "Clear the search or pick another filter." : "Read a photo or use Live scan: every plate found is saved here with its picture.", 13.5f, MUTED, false);
            a.setGravity(Gravity.CENTER); b.setGravity(Gravity.CENTER);
            e.addView(a); e.addView(b);
            recList.addView(e, lp(-1, -2, 0, 8));
        }
        DateFormat fmt = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT);
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inSampleSize = 1;
        for (Records.Row r : rows) {
            LinearLayout c = card();
            c.setPadding(dp(10), dp(10), dp(6), dp(10));
            if (r.watch > 0) c.setBackground(round(RED_SOFT, RED, 16));
            LinearLayout rr = row();
            ImageView iv = new ImageView(this);
            iv.setScaleType(ImageView.ScaleType.FIT_CENTER);
            iv.setBackground(round(Color.rgb(17, 17, 17), 0, 8));
            iv.setClipToOutline(true);
            if (r.crop != null) iv.setImageBitmap(BitmapFactory.decodeFile(rec.file(r.crop).getAbsolutePath(), o));
            rr.addView(iv, new LinearLayout.LayoutParams(dp(104), dp(56)));
            LinearLayout t = vertical();
            t.setPadding(dp(12), 0, dp(4), 0);
            TextView big = text(r.text == null || r.text.isEmpty() ? "—" : r.text, 18, INK, true);
            big.setSingleLine(true); big.setEllipsize(TextUtils.TruncateAt.END);
            t.addView(big);
            TextView en = text(r.english, 12, MUTED, false);
            en.setSingleLine(true);
            t.addView(en);
            t.addView(chips(statusChip(r.status), r.watch > 0 ? chip("WATCHLIST", WHITE, RED) : null, r.night ? nightChip() : null), lp(-2, -2, 0, 4));
            TextView meta = text("#" + r.id + " · " + fmt.format(new java.util.Date(r.ts)) + " · " + r.source, 11.5f, MUTED, false);
            t.addView(meta, lp(-2, -2, 0, 4));
            rr.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
            TextView del = text("✕", 18, MUTED, false);
            del.setPadding(dp(10), dp(6), dp(10), dp(6));
            del.setOnClickListener(v -> new AlertDialog.Builder(this).setMessage("Delete record #" + r.id + " and its plate picture?")
                    .setPositiveButton("Delete", (d, w) -> { rec.delete(r.id); loadRecords(); }).setNegativeButton("Cancel", null).show());
            rr.addView(del);
            c.addView(rr);
            c.setOnLongClickListener(v -> { if (r.text != null && !r.text.isEmpty()) copy(r.text); return true; });
            recList.addView(c, lp(-1, -2, 0, 8));
        }
        if (rows.size() == 200) recList.addView(text("Showing the newest 200. Search to find older plates; Export includes all of them.", 12, MUTED, false), lp(-1, -2, 0, 8));

        watchListView.removeAllViews();
        for (String[] wl : rec.watchList()) {
            LinearLayout r = row();
            r.setPadding(0, dp(4), 0, dp(4));
            r.addView(text(wl[1], 17, RED, true), new LinearLayout.LayoutParams(0, -2, 1));
            Button del = button("Remove", 2);
            del.setTextSize(13);
            del.setOnClickListener(v -> { rec.watchDel(wl[0]); loadRecords(); });
            r.addView(del);
            watchListView.addView(r);
        }
    }

    Records.Export lastExport;

    void doExport() { exportThen(false); }
    void doExportAndShare() { exportThen(true); }

    void exportThen(boolean share) {
        toast("Exporting…");
        String q = recQ.getText().toString().trim(), st = recStatus;
        worker.execute(() -> {
            try {
                Records.Export ex = rec.export(this, q, st);
                lastExport = ex;
                runOnUiThread(() -> {
                    exportBox.removeAllViews();
                    exportBox.setVisibility(View.VISIBLE);
                    exportBox.addView(text("✓ " + ex.plates + " plates exported with " + ex.images + " pictures", 14, GREEN, true));
                    String where = ex.publicPath != null ? "Saved in the Files app: " + ex.publicPath : "Saved in the app folder: " + ex.zip.getAbsolutePath();
                    TextView w = text(where + "\nInside: results.csv, plate_images/ and results.html (open in a browser to see each plate with its picture).", 12.5f, INK, false);
                    w.setTextIsSelectable(true);
                    exportBox.addView(w, lp(-1, -2, 0, 4));
                    Button s = button("Share this zip", 1);
                    s.setOnClickListener(v -> shareFile(ex.zip, "application/zip", "Share the export"));
                    exportBox.addView(s, lp(-1, -2, 0, 8));
                    if (share) shareFile(ex.zip, "application/zip", "Share the export");
                });
            } catch (Throwable e) {
                Log.e("BanglaLPR", "export", e);
                toast("Export failed: " + e.getMessage());
            }
        });
    }

    void shareFile(File f, String mime, String title) {
        try {
            Uri u = FileProvider.getUriForFile(this, getPackageName() + ".files", f);
            Intent i = new Intent(Intent.ACTION_SEND).setType(mime).putExtra(Intent.EXTRA_STREAM, u).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(i, title));
        } catch (Exception e) { toast("Share failed: " + e.getMessage()); }
    }

    // ------------------------------------------------------------------------------------------ help
    View buildHelp() {
        LinearLayout l = vertical();
        LinearLayout how = card();
        how.addView(sectionTitle("How to use"));
        String[] steps = {"Photo: take a photo of a vehicle or pick one. The plate is found, cut out and read.",
                "Check the result: the Bangla text is shown big, with its English form and a status.",
                "It is saved by itself: the plate picture and one row in plates.csv.",
                "Live scan: point the camera at traffic. Each new plate is saved once per 10 seconds.",
                "Records: search and filter, then Export to get a zip with the CSV, the plate pictures and a printable page."};
        for (int i = 0; i < steps.length; i++) {
            LinearLayout r = row();
            r.setGravity(Gravity.TOP);
            TextView n = text(String.valueOf(i + 1), 13, GREEN, true);
            n.setGravity(Gravity.CENTER);
            n.setBackground(round(GREEN_SOFT, 0, 99));
            r.addView(n, new LinearLayout.LayoutParams(dp(24), dp(24)));
            TextView t = text(steps[i], 14, INK, false);
            t.setPadding(dp(10), dp(1), 0, 0);
            r.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
            how.addView(r, lp(-1, -2, 0, 10));
        }
        l.addView(how);

        LinearLayout st = card();
        st.addView(sectionTitle("What the status means"));
        Object[][] items = {{"verified", "the readers agree and the plate has a valid BRTA layout."},
                {"likely", "a valid layout, but the readers did not fully agree."},
                {"review", "the text may be wrong: look at the plate picture."}};
        for (Object[] it : items) {
            LinearLayout r = vertical();
            r.addView(chips(statusChip((String) it[0])));
            r.addView(text((String) it[1], 13.5f, MUTED, false), lp(-1, -2, 0, 3));
            st.addView(r, lp(-1, -2, 0, 10));
        }
        l.addView(st, lp(-1, -2, 0, 12));

        LinearLayout f = card();
        f.addView(sectionTitle("The CSV in Excel"));
        f.addView(text("The \"Plate image\" column gives the picture file of each row, relative to the CSV, so an unzipped export keeps "
                + "its links. results.html in each export shows every row with its picture and can be printed.", 13.5f, MUTED, false), lp(-1, -2, 0, 4));
        l.addView(f, lp(-1, -2, 0, 12));

        LinearLayout m = card();
        m.addView(sectionTitle("The model"));
        m.addView(text("V3 uses the V2 reading engine and the same models unchanged (no new training): YOLOv8n-576 detector with a "
                + "YOLOv8s retry (+ CLAHE), two character-YOLOv8n readers, and LPNR YOLO12n + SVTRv2-S/T when they disagree.\n\n"
                + "Measured on the team PC (dataset v1.1 test): 92.65 % of 204 whole test photos read exactly (character accuracy 96.79 %), "
                + "91.71 % of 651 test plate crops. This Java engine gave the same text as the Python system on every one of those test "
                + "photos and crops when run on the PC.\n\n"
                + "Speed on a phone was not measured; it depends on the phone's processor (Live scan shows the time per frame).\n\n"
                + "Everything runs on this phone. The app has no internet permission: photos, plates and records never leave the phone "
                + "unless you share an export.", 13.5f, MUTED, false), lp(-1, -2, 0, 4));
        m.addView(text("NSU CSE 498R · Md. Yousuf · MD. Rokib Hasan Oli · Tashfia Adiba Zaman Nizhum · Md Rakibul Hasan", 12, MUTED, false), lp(-1, -2, 0, 10));
        l.addView(m, lp(-1, -2, 0, 12));
        return scroller(l);
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        scanning = false;
        worker.shutdown();
    }
}
