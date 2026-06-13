package org.telegram.ui.Components;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapShader;
import android.graphics.Canvas;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Shader;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;

import org.telegram.messenger.AndroidUtilities;

/**
 * KrimbaGram cyberdeck: a global, non-interactive overlay that recreates the look of an old
 * electron-beam / oscilloscope CRT — horizontal scanlines, a slow phosphor scan-band sweeping
 * down the screen, and a subtle flicker. Drawn on top of all app content at low opacity so the
 * UI stays readable. Touches always pass through to the views below.
 */
public class CrtOverlayView extends View {

    private final Paint scanlinePaint = new Paint();
    private final Paint bandPaint = new Paint();
    private final Paint beamPaint = new Paint();
    private final long start = SystemClock.elapsedRealtime();

    private final int bandHeight = AndroidUtilities.dp(170);
    private LinearGradient bandGradient;
    private int gradientForHeight = -1;

    public CrtOverlayView(Context context) {
        super(context);
        setWillNotDraw(false);
        setClickable(false);
        setFocusable(false);

        // scanline tile: tight, darker lines for a pronounced CRT raster.
        int line = AndroidUtilities.dp(2);
        Bitmap tile = Bitmap.createBitmap(1, line, Bitmap.Config.ARGB_8888);
        for (int y = 0; y < line; y++) {
            tile.setPixel(0, y, y >= line - 1 ? 0x55000000 : 0x00000000);
        }
        scanlinePaint.setShader(new BitmapShader(tile, Shader.TileMode.REPEAT, Shader.TileMode.REPEAT));

        beamPaint.setColor(0x664DFFA3);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return false; // never consume touches
    }

    @Override
    protected void onDraw(Canvas canvas) {
        final int w = getWidth();
        final int h = getHeight();
        if (w == 0 || h == 0) {
            return;
        }
        final long elapsed = SystemClock.elapsedRealtime() - start;

        // 1) static scanlines across the whole surface
        canvas.drawRect(0, 0, w, h, scanlinePaint);

        // 2) phosphor scan-band sweeping top -> bottom (~5s loop), with a sharp beam edge
        if (bandGradient == null || gradientForHeight != bandHeight) {
            bandGradient = new LinearGradient(0, 0, 0, bandHeight,
                    new int[]{0x00000000, 0x3A4DFFA3, 0x00000000}, null, Shader.TileMode.CLAMP);
            gradientForHeight = bandHeight;
            bandPaint.setShader(bandGradient);
        }
        final float t = (elapsed % 5000L) / 5000f;
        final float bandY = -bandHeight + t * (h + bandHeight);
        canvas.save();
        canvas.translate(0, bandY);
        canvas.drawRect(0, 0, w, bandHeight, bandPaint);
        canvas.restore();
        // sharp electron-beam line at the leading edge of the band
        canvas.drawRect(0, bandY + bandHeight - AndroidUtilities.dp(1.5f), w, bandY + bandHeight, beamPaint);

        // 3) phosphor flicker
        final int a = (int) (10 + 14 * Math.abs(Math.sin(elapsed / 70.0)));
        canvas.drawColor((a << 24) | 0x004DFFA3);

        postInvalidateOnAnimation();
    }
}
