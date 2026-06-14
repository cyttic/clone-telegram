# KrimbaGram — On-Device Voice Changer

Status: **working** (v1). Records a voice message in your own voice and sends it re-voiced
through a fine-tuned **RVC** (Retrieval-based Voice Conversion) model, running **entirely on
the phone** via ONNX Runtime. No server, no network — fully offline.

---

## 1. What it does

When you record a voice message, KrimbaGram tees the raw microphone PCM, and **before the
message is sent** it runs the audio through the RVC pipeline, re-encodes the converted audio
back to Opus, and swaps that file into the outgoing message. The recipient hears the target
voice (e.g. the `lecturer_ru` model), not yours.

It is gated by a flag (`MediaController.voiceChangerEnabled`, default `true`) and is a
**graceful no-op** if the model files aren't installed — voice messages then send normally.

---

## 2. How it works (pipeline)

Ported from RVC-WebUI's `infer/lib/infer_pack/onnx_inference.py` into Java:

```
record voice msg
  → tee raw PCM (48 kHz mono int16)            [MediaController.recordRunnable]
on send:
  → resample 48k → 16k                          [VoiceChanger]
  → HuBERT/ContentVec (hubert.onnx)  → (1,T,768) content features
  → repeat ×2 along time            → (1,2T,768)
  → f0 (pitch) estimated in Java (YIN)          → pitch (coarse, int) + pitchf (float)
  → generator (generator.onnx): inputs
        phone (1,2T,768) f32, phone_lengths (1,) i64,
        pitch (1,2T) i64, pitchf (1,2T) f32, ds (1,) i64, rnd (1,192,2T) f32
     → audio @ 40 kHz (float, ×32767 → int16)
  → resample 40k → 48k
  → re-encode to Opus (native startRecord/writeFrame/stopRecord)
  → swap the outgoing voice-message file, recompute duration
```

**Why f0 is done in Java (YIN):** RVC normally uses an `rmvpe.onnx` pitch model. v1 estimates
f0 with a self-contained YIN tracker so we only ship **two** model files instead of three.
Pitch accuracy can be upgraded later by swapping `VoiceChanger.estimateF0()` for an rmvpe.onnx
session.

---

## 3. Code map (where everything lives)

| Piece | Location |
|---|---|
| ONNX Runtime dependency | `TMessagesProj/build.gradle` → `com.microsoft.onnxruntime:onnxruntime-android:1.19.2` |
| Inference engine | `TMessagesProj/src/main/java/org/telegram/messenger/voicechanger/VoiceChanger.java` |
| Enable flag | `MediaController.voiceChangerEnabled` (static, default `true`) |
| PCM tee (capture during record) | `MediaController.recordRunnable` (after `audioRecorder.read`) |
| Tee reset on new recording | `MediaController` (where `recordTimeCount = 0`) |
| Convert + re-encode + swap | `MediaController.krimbaApplyVoiceChanger()` / `krimbaReencodeOpus()` / `krimbaResample()` |
| Hook into send path | `MediaController.stopRecordingInternal()` (inside the `fileEncodingQueue` block, right after `joinRecord`) |

The engine reads the two models from the app's **external** files dir (so they can be pushed
with plain `adb push`, no root):

```
/sdcard/Android/data/org.telegram.messenger.beta/files/voicechanger/
    hubert.onnx       (shared ContentVec v2 encoder, 768-dim / layer 12)
    generator.onnx    (the voice — exported from the RVC .pth)
```
(`VoiceChanger.modelDir()` = `Context.getExternalFilesDir(null)/voicechanger`.)

---

## 4. How the model files are produced

The phone runs **ONNX**, but RVC ships **PyTorch (.pth)**, so the model is exported once.

- **`generator.onnx`** — exported from `lecturer_ru.pth` with RVC-WebUI's
  `infer.modules.onnx.export.export_onnx`.
- **`hubert.onnx`** — the standard ContentVec v2 `vec-768-layer-12.onnx` (same for every voice),
  downloaded from HF (`NaruseMioShirakana/MoeSS-SUBModel`).

Export is automated in: `voice-synthesizer/notebooks/rvc_export_onnx.ipynb` (Kaggle, Python-3.10
env via uv). Key gotchas that the notebook already handles:
- RVC's stack only runs on **Python 3.10** (fairseq/hydra/omegaconf break on 3.11/3.12).
- `pip<24.1` so fairseq's old omegaconf metadata is accepted.
- `torch.load` patched to `weights_only=False` (PyTorch ≥2.6).
- `torch.onnx.export(..., dynamo=False)` to force the legacy exporter RVC was written for.
- extra deps: `onnxsim`, `onnxscript`.

---

## 5. Changing the voice model later  ← (planned)

To swap in a different fine-tuned RVC voice:

1. Train/obtain the new RVC **v2** model (`.pth`) — must be **768-dim / layer-12** so it stays
   compatible with the same `hubert.onnx`. (If it's a v1 / 256-dim model, you'd also need the
   matching `vec-256-layer-9.onnx` and a small change in `VoiceChanger` FEAT_DIM=256.)
2. Export it to `generator.onnx` with the export notebook (point `PTH_FILE` at the new model).
3. Push **only** the new `generator.onnx` (hubert.onnx is unchanged):
   ```
   adb push generator.onnx /sdcard/Android/data/org.telegram.messenger.beta/files/voicechanger/
   ```
4. Restart the app (the engine loads the model once per process; `VoiceChanger` caches the
   session). No rebuild needed — the model lives in storage, not the APK.

The output sample rate is assumed **40 kHz** (`VoiceChanger.MODEL_SR`). If a future model is a
48 kHz RVC model, change that constant.

---

## 6. Future: voice-model chooser in Settings  ← (planned)

The groundwork is there; to add a UI picker:
- Store multiple generators as `voicechanger/<name>.onnx` and a selected-name preference.
- Change `VoiceChanger.modelDir()`/`ensureLoaded()` to load `generator-<name>.onnx` based on the
  pref, and add a `reload()` that disposes the current session and loads the chosen one.
- Add a row in the KrimbaGram settings screen (`SettingsActivity`) — a list of available
  `*.onnx` voices + an on/off toggle bound to `MediaController.voiceChangerEnabled` (persist it
  in prefs instead of the static default).
- Optional: per-voice pitch shift (`f0UpKey`) — already a parameter of `VoiceChanger.convert()`.

---

## 7. Current limitations (v1)

- **Single-segment recordings only.** Paused/resumed recordings (with a previous segment) are
  converted from the teed PCM of the current run; multi-segment join isn't handled specially.
- **CPU only**, so there's a noticeable delay after pressing send (model load ~470 MB + inference).
  NNAPI/GPU delegate is stubbed but off (`OrtSession.SessionOptions.addNnapi()`).
- **Java YIN f0** is decent but not as accurate as rmvpe — pitch can wobble. Upgrade path noted above.
- **Linear resampling** (16k/40k/48k). Fine for v1; a sinc resampler would improve fidelity.
- **No FAISS index** (`index_rate=0`) — slightly less timbre fidelity than the desktop result,
  in exchange for not shipping the 379 MB index.
- **Voice messages only.** Real-time VoIP calls are not done (much harder; streaming + native
  tgcalls hook).
- The `voiceChangerEnabled` flag defaults **on** with no UI yet (see §6).

---

## 8. Build / install / test

```bash
# build
cd /mnt/ssd2/cyttic/projects/clone-telegram
./gradlew :TMessagesProj_App:assembleAfatDebug

# install
adb install -r TMessagesProj_App/build/outputs/apk/afat/debug/app.apk

# push models (first time / when changing voice)
adb shell mkdir -p /sdcard/Android/data/org.telegram.messenger.beta/files/voicechanger
adb push generator.onnx hubert.onnx \
  /sdcard/Android/data/org.telegram.messenger.beta/files/voicechanger/

# watch the pipeline
adb logcat -c && adb logcat -s KRIMBAVC
```
Then record a voice message; `KRIMBAVC` logs show: hook reached → models loaded → convert
start → convert done (with timing).

---

## 9. Tuning knobs (in `VoiceChanger` / the hook call)

- `convert(pcm, sampleRate, f0UpKey, speakerId)` — `f0UpKey` is pitch shift in semitones
  (the hook currently passes `0`).
- `MODEL_SR` (40000) — generator output rate.
- YIN: `F0_MIN`/`F0_MAX`, the voiced threshold `bestVal < 0.15f`, window `win`.
- Coarse-pitch mel constants follow RVC exactly — don't change unless the model changes.
