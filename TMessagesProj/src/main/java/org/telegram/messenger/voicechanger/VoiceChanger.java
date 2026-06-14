package org.telegram.messenger.voicechanger;

import android.content.Context;

import java.io.File;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * KrimbaGram on-device voice changer — RVC (Retrieval-based Voice Conversion) inference via
 * ONNX Runtime Mobile. Ported from RVC-WebUI's infer/lib/infer_pack/onnx_inference.py.
 *
 * Pipeline:  PCM(any rate) -> 16k -> HuBERT features -> (repeat x2) -> generator(+f0) -> 40k PCM
 *
 * Needs two ONNX files in {@link #modelDir()}:
 *   hubert.onnx      (ContentVec/HuBERT content encoder; shared across voices)
 *   generator.onnx   (the fine-tuned voice, exported from lecturer_ru.pth)
 *
 * f0/pitch is estimated in Java (YIN) so we don't need rmvpe.onnx for v1. Quality can be
 * upgraded later by swapping {@link #estimateF0} for an rmvpe.onnx session.
 */
public class VoiceChanger {

    public static final int MODEL_SR = 40000;   // generator output sample rate
    private static final int HUBERT_SR = 16000;
    private static final int FEAT_DIM = 768;
    private static final int RND_DIM = 192;
    private static final double F0_MIN = 50.0, F0_MAX = 1100.0;
    private static final double F0_MEL_MIN = 1127.0 * Math.log(1 + F0_MIN / 700.0);
    private static final double F0_MEL_MAX = 1127.0 * Math.log(1 + F0_MAX / 700.0);

    private static volatile VoiceChanger instance;

    private OrtEnvironment env;
    private OrtSession hubert;
    private OrtSession generator;
    private boolean ready;

    public static VoiceChanger getInstance() {
        if (instance == null) {
            synchronized (VoiceChanger.class) {
                if (instance == null) instance = new VoiceChanger();
            }
        }
        return instance;
    }

    public static File modelDir(Context ctx) {
        // Prefer the app-specific EXTERNAL dir so models can be pushed with plain `adb push`:
        //   adb push generator.onnx hubert.onnx /sdcard/Android/data/<pkg>/files/voicechanger/
        File ext = ctx.getExternalFilesDir(null);
        if (ext != null) {
            return new File(ext, "voicechanger");
        }
        return new File(ctx.getFilesDir(), "voicechanger");
    }

    /** Loads the ONNX sessions. Safe to call repeatedly; returns true if both models are present and loaded. */
    public synchronized boolean ensureLoaded(Context ctx) {
        if (ready) return true;
        try {
            File dir = modelDir(ctx);
            File hubertFile = new File(dir, "hubert.onnx");
            File genFile = new File(dir, "generator.onnx");
            android.util.Log.i("KRIMBAVC", "ensureLoaded dir=" + dir + " hubert=" + hubertFile.exists() + " gen=" + genFile.exists());
            if (!hubertFile.exists() || !genFile.exists()) {
                return false;
            }
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
            // NNAPI can be enabled later for speed: opts.addNnapi();
            hubert = env.createSession(hubertFile.getAbsolutePath(), opts);
            generator = env.createSession(genFile.getAbsolutePath(), opts);
            android.util.Log.i("KRIMBAVC", "models loaded. hubertIn=" + hubert.getInputNames() + " genIn=" + generator.getInputNames() + " genOut=" + generator.getOutputNames());
            ready = true;
            return true;
        } catch (Throwable e) {
            android.util.Log.e("KRIMBAVC", "ensureLoaded failed", e);
            ready = false;
            return false;
        }
    }

    public boolean isReady() { return ready; }

    /**
     * Convert a mono PCM clip into the target voice.
     *
     * @param pcmIn      input samples (16-bit mono)
     * @param sampleRate input sample rate (e.g. 48000)
     * @param f0UpKey    pitch shift in semitones (0 = none)
     * @param speakerId  RVC speaker id (usually 0)
     * @return converted PCM (16-bit mono) at {@link #MODEL_SR}, or null on failure
     */
    public synchronized short[] convert(short[] pcmIn, int sampleRate, int f0UpKey, int speakerId) {
        if (!ready || pcmIn == null || pcmIn.length == 0) return null;
        long t0 = System.currentTimeMillis();
        android.util.Log.i("KRIMBAVC", "convert start: samples=" + pcmIn.length + " sr=" + sampleRate);
        try {
            // 1) to float [-1,1] and resample to 16k for HuBERT + f0
            float[] x = new float[pcmIn.length];
            for (int i = 0; i < pcmIn.length; i++) x[i] = pcmIn[i] / 32768f;
            float[] wav16k = resample(x, sampleRate, HUBERT_SR);

            // 2) HuBERT features: (1,1,N) -> (1,T,768)
            float[][][] feats = runHubert(wav16k);   // [1][T][768]
            int t = feats[0].length;
            int len = t * 2;                           // np.repeat(.,2) along time

            // (1, 2T, 768) with each frame duplicated
            float[] hubertFlat = new float[len * FEAT_DIM];
            for (int i = 0; i < t; i++) {
                float[] frame = feats[0][i];
                int o1 = (2 * i) * FEAT_DIM;
                int o2 = (2 * i + 1) * FEAT_DIM;
                System.arraycopy(frame, 0, hubertFlat, o1, FEAT_DIM);
                System.arraycopy(frame, 0, hubertFlat, o2, FEAT_DIM);
            }

            // 3) f0 over wav16k, one value per output frame (len)
            float[] f0 = estimateF0(wav16k, len);
            float pitchShift = (float) Math.pow(2.0, f0UpKey / 12.0);
            float[] pitchf = new float[len];
            long[] pitch = new long[len];
            for (int i = 0; i < len; i++) {
                float v = f0[i] * pitchShift;
                pitchf[i] = v;
                pitch[i] = coarseF0(v);
            }

            // 4) noise rnd (1,192,len)
            float[] rnd = new float[RND_DIM * len];
            java.util.Random rng = new java.util.Random(0);
            for (int i = 0; i < rnd.length; i++) rnd[i] = (float) rng.nextGaussian();

            // 5) generator
            android.util.Log.i("KRIMBAVC", "hubert T=" + t + " feed len=" + len + " (running generator)");
            float[] audio = runGenerator(hubertFlat, len, pitch, pitchf, speakerId, rnd);
            if (audio == null) return null;
            android.util.Log.i("KRIMBAVC", "convert done: outSamples=" + audio.length + " ms=" + (System.currentTimeMillis() - t0));

            // 6) float -> int16
            short[] out = new short[audio.length];
            for (int i = 0; i < audio.length; i++) {
                int s = Math.round(audio[i] * 32767f);
                out[i] = (short) Math.max(-32768, Math.min(32767, s));
            }
            return out;
        } catch (Throwable e) {
            android.util.Log.e("KRIMBAVC", "convert failed", e);
            return null;
        }
    }

    private float[][][] runHubert(float[] wav16k) throws Exception {
        long[] shape = {1, 1, wav16k.length};
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(wav16k), shape)) {
            String name = hubert.getInputNames().iterator().next();
            Map<String, OnnxTensor> m = new HashMap<>();
            m.put(name, in);
            try (OrtSession.Result r = hubert.run(m)) {
                String out = hubert.getOutputNames().iterator().next();
                Object v = r.get(out).get().getValue();
                return (float[][][]) v;   // (1, T, 768)
            }
        }
    }

    private float[] runGenerator(float[] hubertFlat, int len, long[] pitch, float[] pitchf,
                                 int sid, float[] rnd) throws Exception {
        List<String> in = new ArrayList<>(generator.getInputNames());   // model-declared order
        OnnxTensor tHubert = OnnxTensor.createTensor(env, FloatBuffer.wrap(hubertFlat), new long[]{1, len, FEAT_DIM});
        OnnxTensor tLen    = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{len}), new long[]{1});
        OnnxTensor tPitch  = OnnxTensor.createTensor(env, LongBuffer.wrap(pitch), new long[]{1, len});
        OnnxTensor tPitchf = OnnxTensor.createTensor(env, FloatBuffer.wrap(pitchf), new long[]{1, len});
        OnnxTensor tDs     = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{sid}), new long[]{1});
        OnnxTensor tRnd    = OnnxTensor.createTensor(env, FloatBuffer.wrap(rnd), new long[]{1, RND_DIM, len});
        Map<String, OnnxTensor> m = new HashMap<>();
        m.put(in.get(0), tHubert);
        m.put(in.get(1), tLen);
        m.put(in.get(2), tPitch);
        m.put(in.get(3), tPitchf);
        m.put(in.get(4), tDs);
        m.put(in.get(5), tRnd);
        try (OrtSession.Result r = generator.run(m)) {
            String out = generator.getOutputNames().iterator().next();
            Object v = r.get(out).get().getValue();
            return flatten(v);
        } finally {
            tHubert.close(); tLen.close(); tPitch.close(); tPitchf.close(); tDs.close(); tRnd.close();
        }
    }

    /** RVC coarse-pitch mel quantization -> [1,255]; 0 stays 0 (unvoiced). */
    private static long coarseF0(float f0) {
        if (f0 <= 0) return 0;
        double mel = 1127.0 * Math.log(1 + f0 / 700.0);
        double q = (mel - F0_MEL_MIN) * 254.0 / (F0_MEL_MAX - F0_MEL_MIN) + 1.0;
        long c = Math.round(q);
        if (c < 1) c = 1;
        if (c > 255) c = 255;
        return c;
    }

    /** Flatten an ONNX float output of rank 1/2/3 into a 1-D array. */
    private static float[] flatten(Object v) {
        if (v instanceof float[]) return (float[]) v;
        if (v instanceof float[][]) {
            float[][] a = (float[][]) v;
            return a[a.length - 1 >= 0 ? 0 : 0]; // (1, N) -> first row
        }
        if (v instanceof float[][][]) {
            float[][][] a = (float[][][]) v;      // (1, 1, N)
            return a[0][0];
        }
        return null;
    }

    /** Simple linear resampler (mono). Good enough for v1; upgrade to sinc later. */
    private static float[] resample(float[] in, int srIn, int srOut) {
        if (srIn == srOut) return in;
        double ratio = (double) srOut / srIn;
        int n = (int) Math.floor(in.length * ratio);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            double pos = i / ratio;
            int i0 = (int) Math.floor(pos);
            int i1 = Math.min(i0 + 1, in.length - 1);
            double frac = pos - i0;
            out[i] = (float) (in[i0] * (1 - frac) + in[i1] * frac);
        }
        return out;
    }

    /**
     * YIN-style f0 over wav16k, resampled to exactly {@code frames} values (Hz; 0 = unvoiced).
     * Self-contained so v1 needs no rmvpe.onnx.
     */
    private static float[] estimateF0(float[] wav16k, int frames) {
        int win = 1024;
        int hop = Math.max(1, wav16k.length / Math.max(1, frames));
        int tauMin = (int) (HUBERT_SR / F0_MAX);
        int tauMax = (int) (HUBERT_SR / F0_MIN);
        float[] raw = new float[frames];
        for (int f = 0; f < frames; f++) {
            int start = f * hop;
            float hz = 0f;
            if (start + win + tauMax < wav16k.length) {   // need room for the tau lookahead
                float bestVal = 1f;
                int bestTau = -1;
                double cumulative = 0;
                float[] diff = new float[tauMax + 1];
                for (int tau = 1; tau <= tauMax; tau++) {
                    double sum = 0;
                    for (int j = 0; j < win; j++) {
                        float d = wav16k[start + j] - wav16k[start + j + tau];
                        sum += d * d;
                    }
                    diff[tau] = (float) sum;
                }
                for (int tau = 1; tau <= tauMax; tau++) {
                    cumulative += diff[tau];
                    float cmnd = cumulative > 0 ? (float) (diff[tau] * tau / cumulative) : 1f;
                    if (tau >= tauMin && cmnd < bestVal) { bestVal = cmnd; bestTau = tau; }
                }
                if (bestTau > 0 && bestVal < 0.15f) hz = (float) HUBERT_SR / bestTau;
            }
            raw[f] = hz;
        }
        return raw;
    }
}
