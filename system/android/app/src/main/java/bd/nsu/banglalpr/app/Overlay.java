package bd.nsu.banglalpr.app;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/** Boxes and plate text drawn over the camera preview (PreviewView uses FILL_CENTER: same mapping here). */
final class Overlay extends View {
    static final class Item { float x1, y1, x2, y2; String label; String status; }

    private final Paint box = new Paint(Paint.ANTI_ALIAS_FLAG), text = new Paint(Paint.ANTI_ALIAS_FLAG), bg = new Paint();
    private List<Item> items = new ArrayList<>();
    private int imgW = 1, imgH = 1;

    Overlay(Context c) {
        super(c);
        box.setStyle(Paint.Style.STROKE);
        box.setStrokeWidth(6);
        text.setColor(Color.WHITE);
        text.setTextSize(44);
        text.setFakeBoldText(true);
    }

    void show(List<Item> it, int w, int h) { items = it; imgW = w; imgH = h; postInvalidate(); }

    static int color(String status) {
        return "verified".equals(status) ? Color.rgb(47, 191, 113) : "likely".equals(status) ? Color.rgb(255, 176, 32) : Color.rgb(255, 77, 94);
    }

    @Override protected void onDraw(Canvas c) {
        float s = Math.max((float) getWidth() / imgW, (float) getHeight() / imgH);
        float ox = (getWidth() - imgW * s) / 2, oy = (getHeight() - imgH * s) / 2;
        for (Item i : items) {
            int col = color(i.status);
            box.setColor(col);
            float x1 = i.x1 * s + ox, y1 = i.y1 * s + oy, x2 = i.x2 * s + ox, y2 = i.y2 * s + oy;
            c.drawRect(x1, y1, x2, y2, box);
            if (i.label != null && !i.label.isEmpty()) {
                float tw = text.measureText(i.label);
                bg.setColor(col);
                float ty = Math.max(50, y1 - 12);
                c.drawRect(x1, ty - 46, x1 + tw + 16, ty + 10, bg);
                c.drawText(i.label, x1 + 8, ty, text);
            }
        }
    }
}
