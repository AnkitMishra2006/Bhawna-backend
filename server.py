"""
server.py — FastAPI WebSocket server for real-time emotion tracking.

Start the server:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

WebSocket endpoint:
    ws://localhost:8000/ws/{session_id}

── Protocol ──────────────────────────────────────────────────────────────────
Client → Server  (per frame, every ~200 ms):
    {
        "type":      "frame",
        "data":      "<base64-encoded JPEG/PNG — may include data URI prefix>",
        "timestamp": 1.23       ← seconds into the video
    }

Server → Client  (response to each frame):
    {
        "type":            "result",
        "face_detected":   true,
        "raw_scores":      {"angry": 1.2, "happy": 84.1, ...},  ← 0–100 %
        "smoothed_scores": {"angry": 0.9, "happy": 79.4, ...},  ← 15-frame avg
        "dominant_emotion": "happy",
        "confidence":       79.4,
        "timestamp":        1.23
    }
    or, if no face found:
    {
        "type":          "result",
        "face_detected": false,
        "timestamp":     1.23
    }

Client → Server  (audio chunk, sent periodically while recording):
    {
        "type": "audio_chunk",
        "data": "<base64-encoded raw audio bytes — WebM/Opus from MediaRecorder>"
    }

Client → Server  (when video ends / user stops):
    {"type": "end"}

Server → Client  (final AI / template report):
    {
        "type": "report",
        "text": "Throughout the session..."
    }

── Health check ──────────────────────────────────────────────────────────────
GET  /health   →  {"status": "ok", "model_loaded": true, "device": "cpu", ...}

── LLM report (optional) ─────────────────────────────────────────────────────
Set GEMINI_API_KEY in a .env file (or as an env var) to enable AI-generated
session reports using Google Gemini (free tier — no billing required).
Without a key the server falls back to a detailed template-based report that
is still informative and human-readable.

── Audio transcription (optional) ────────────────────────────────────────────
Set WHISPER_MODEL in .env to a Whisper model size (tiny / base / small /
medium / large) to enable speech-to-text transcription of the session audio.
The transcript is woven into both the LLM prompt and the fallback template.
Default: base (74 MB, good English accuracy, fast on CPU).
Requires:  pip install openai-whisper   and   ffmpeg on your PATH.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Make sure model.py is importable when the working directory is not backend/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import EmotionNet

# ── Load environment variables from optional .env file ──────────────────────
load_dotenv(os.path.join(HERE, ".env"))

# ── Authentication (Google OAuth + JWT + MongoDB) ───────────────────────────
from auth import auth_router, get_user_from_token, init_db

# ── Globals loaded once at startup ──────────────────────────────────────────
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL:  Optional[EmotionNet] = None
MEAN:   List[float] = [0.5, 0.5, 0.5]
STD:    List[float] = [0.5, 0.5, 0.5]
CLASS_NAMES: List[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# MediaPipe face detector — initialised once, reused across all sessions.
# IMPORTANT: FaceDetection.process() is NOT thread-safe.  All calls are
# serialised through _MP_LOCK so concurrent WebSocket sessions don't race.
FACE_DETECTOR: Optional[mp.solutions.face_detection.FaceDetection] = None
_MP_LOCK = threading.Lock()

# Whisper speech-to-text — loaded at startup if openai-whisper is installed.
WHISPER_MODEL: Optional[Any] = None

# ── Per-session state ────────────────────────────────────────────────────────
# key = session_id (string from the URL path)
SESSIONS: Dict[str, dict] = {}

# The sliding window size (15 frames ≈ 3 s at 5 fps) as used by the project spec
WINDOW_SIZE = 15


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI application."""
    global MODEL, MEAN, STD, CLASS_NAMES, FACE_DETECTOR, WHISPER_MODEL

    # ── Initialise authentication database ────────────────────────────────────
    init_db()

    # ── Load trained emotion model ────────────────────────────────────────────
    model_path = os.path.join(HERE, "emotion_model.pth")
    if os.path.exists(model_path):
        # weights_only=False because the checkpoint contains Python lists
        # (class_names, mean, std) alongside tensors.
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        CLASS_NAMES = ckpt.get("class_names", CLASS_NAMES)
        MEAN        = ckpt.get("mean",        MEAN)
        STD         = ckpt.get("std",         STD)

        MODEL = EmotionNet(num_classes=len(CLASS_NAMES)).to(DEVICE)
        MODEL.load_state_dict(ckpt["model_state_dict"])
        MODEL.eval()

        # Guard: val_acc may be absent in older checkpoints
        val_acc = ckpt.get("val_acc")
        val_acc_str = f"{val_acc:.1f}%" if isinstance(val_acc, (int, float)) else "unknown"
        print(f"[startup] Model loaded (val_acc={val_acc_str}) on {DEVICE}")
    else:
        print(
            f"[startup] WARNING: No model found at {model_path}.\n"
            "          Run  python train.py  inside the backend/ folder first."
        )

    # ── Initialise MediaPipe face detector ────────────────────────────────────
    FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
        model_selection=1,               # model 1: optimised for full-range (0–5 m)
        min_detection_confidence=0.5,
    )
    print("[startup] MediaPipe face detector ready.")

    # ── Load Whisper speech-to-text model (optional) ──────────────────────────
    whisper_model_name = os.getenv("WHISPER_MODEL", "base")
    try:
        import whisper as _whisper   # openai-whisper
        print(f"[startup] Loading Whisper '{whisper_model_name}' model "
              "(downloads ~74 MB on first run)…")
        WHISPER_MODEL = await asyncio.get_running_loop().run_in_executor(
            None, _whisper.load_model, whisper_model_name
        )
        print("[startup] Whisper ready — audio transcription enabled.")
    except ImportError:
        print("[startup] openai-whisper not installed — audio transcription disabled. "
              "Install: pip install openai-whisper (also requires ffmpeg on PATH).")
    except Exception as exc:
        print(f"[startup] Whisper load failed ({exc}) — audio transcription disabled.")

    yield   # ← application is now serving requests

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if FACE_DETECTOR is not None:
        FACE_DETECTOR.close()
    print("[shutdown] Resources released.")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Emotion Tracker API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


# ── Frame processing helpers ─────────────────────────────────────────────────

def _decode_frame(b64_data: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded image (with or without data URI prefix) into a BGR
    numpy array suitable for OpenCV.  Returns None on any decoding failure.
    """
    try:
        # Strip "data:image/jpeg;base64," prefix if present
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_data)
        buf       = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return frame  # may be None if imdecode failed
    except Exception:
        return None


def _detect_and_crop_face(
    frame: np.ndarray, pad: float = 0.15
) -> Tuple[Optional[np.ndarray], float]:
    """
    Detect the largest face in `frame` using MediaPipe, add padding around the
    bounding box, crop it out, and return a (96×96 RGB numpy array, score) tuple.

    Returns (None, 0.0) if no face is detected or if the crop is degenerate.

    The second element is MediaPipe's detection confidence in [0, 1].  High
    confidence means the detector is certain a face is present; low confidence
    (near the 0.5 threshold) means the detection is marginal — perhaps a small
    face, heavy occlusion, or unusual angle.  The caller uses this score to
    weight the frame's contribution to the smoothing window so that uncertain
    frames influence the live chart less than high-quality detections.

    `pad` controls how much extra context to include around the tight bounding
    box as a fraction of the box size — 0.15 means 15 % extra on each side.
    This matches the style of the training images (pre-cropped face regions
    that include forehead, chin, and slight background).

    Thread-safety: all MediaPipe calls are serialised through _MP_LOCK because
    FaceDetection.process() shares internal state and is not safe to call from
    multiple threads simultaneously.
    """
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with _MP_LOCK:
        results = FACE_DETECTOR.process(rgb)
    if not results.detections:
        return None, 0.0

    # Pick the detection with the largest bounding-box area
    best = max(
        results.detections,
        key=lambda d: (
            d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height
        ),
    )

    # MediaPipe detection score: probability that a face is present [0, 1]
    det_score: float = float(best.score[0]) if best.score else 1.0

    bb   = best.location_data.relative_bounding_box
    bw   = bb.width
    bh   = bb.height

    x0 = int((bb.xmin - pad * bw) * w)
    y0 = int((bb.ymin - pad * bh) * h)
    x1 = int((bb.xmin + (1.0 + pad) * bw) * w)
    y1 = int((bb.ymin + (1.0 + pad) * bh) * h)

    # Clamp to frame bounds
    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(w, x1);  y1 = min(h, y1)

    if x1 <= x0 or y1 <= y0:
        return None, 0.0

    crop = rgb[y0:y1, x0:x1]

    # ── Face alignment: rotate so the eye line is horizontal ───────────────
    # MediaPipe FaceDetection provides 6 keypoints per detection:
    #   index 0 = right eye (person's right — appears on the LEFT of screen)
    #   index 1 = left  eye (person's left  — appears on the RIGHT of screen)
    # Training images were aligned (eyes roughly level).  Real webcam users
    # tilt their heads, creating a train-inference distribution mismatch.
    # We correct this by rotating the crop around the eye midpoint so the
    # model always sees a level face, just as during training.
    kps = best.location_data.relative_keypoints
    if len(kps) >= 2:
        # Convert normalised keypoints to pixel coords relative to the crop
        re_x = kps[0].x * w - x0
        re_y = kps[0].y * h - y0
        le_x = kps[1].x * w - x0
        le_y = kps[1].y * h - y0

        dx = le_x - re_x
        dy = le_y - re_y
        if abs(dx) > 1.0:                         # avoid near-zero divide / noise
            angle = float(np.degrees(np.arctan2(dy, dx)))
            # Correct tilts > 2° but clamp at ±30° to ignore bad keypoints
            if 2.0 < abs(angle) <= 30.0:
                ch, cw = crop.shape[:2]
                eye_cx = (re_x + le_x) / 2.0
                eye_cy = (re_y + le_y) / 2.0
                rot = cv2.getRotationMatrix2D(
                    (float(eye_cx), float(eye_cy)), angle, 1.0
                )
                crop = cv2.warpAffine(
                    crop, rot, (cw, ch),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    return crop, det_score   # HWC uint8 RGB, detection confidence [0, 1]


def _preprocess(face_rgb: np.ndarray) -> torch.Tensor:
    """
    96×96 RGB uint8 numpy → normalised float tensor on DEVICE.
    Uses the same mean/std that was computed during training.
    """
    t    = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1)
    t    = (t - mean) / std
    return t.unsqueeze(0).to(DEVICE)  # (1, 3, 96, 96)


def _run_model(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Forward pass → softmax probabilities → dict of {emotion: 0–100 %}.
    Runs inside torch.no_grad(); safe to call from a sync context.
    """
    probs = MODEL.predict_proba(tensor).squeeze(0).cpu().tolist()
    return {name: round(p * 100, 2) for name, p in zip(CLASS_NAMES, probs)}


def _run_model_tta(face_rgb: np.ndarray) -> Dict[str, float]:
    """
    Run the model with horizontal-flip Test-Time Augmentation (TTA).

    Faces are roughly left-right symmetric for most emotions, so averaging
    predictions over the original crop and its mirror reduces per-frame
    variance and gives more stable probability estimates with no extra
    latency budget: both crops are stacked into a single (2, C, H, W) batch
    and processed in one forward pass.
    """
    t_orig = _preprocess(face_rgb)
    t_flip = _preprocess(np.fliplr(face_rgb).copy())     # horizontal mirror
    batch  = torch.cat([t_orig, t_flip], dim=0)          # (2, 3, 96, 96)
    probs  = MODEL.predict_proba(batch)                  # (2, num_classes)
    avg    = probs.mean(dim=0).cpu().tolist()             # (num_classes,)
    return {name: round(p * 100, 2) for name, p in zip(CLASS_NAMES, avg)}


def _smooth_window(
    window: deque,
    weights: deque,
    ema_alpha: float = 0.3,
) -> Dict[str, float]:
    """
    Compute smoothed emotion scores using an exponentially weighted moving average,
    where each frame is additionally weighted by its detection confidence score.

    Why EMA instead of a plain box average?
    ----------------------------------------
    A uniform 15-frame average gives equal weight to a frame from 3 seconds ago
    and one from 50 ms ago.  For a live emotion display this feels sluggish —
    the chart lags noticeably when the user's expression changes.  EMA fixes
    this: recent frames carry exponentially more weight (controlled by alpha).
    alpha=0.3 means the most recent 5–7 frames dominate while still providing
    enough smoothing to hide single-frame flicker (blinks, micro-expressions).

    Why multiply by detection confidence?
    ----------------------------------------
    MediaPipe returns a confidence in [0, 1] for every detection.  A score near
    0.5 (the minimum threshold) means the detector is uncertain — perhaps the
    face is tiny, heavily occluded, or at an extreme angle.  Those frames should
    contribute less to the live chart than frames where the detector is confident
    (score ≥ 0.9 = clear, well-lit, front-on face).  Multiplying each frame's
    probabilities by its detection score before averaging achieves this naturally.

    Implementation
    ----------------------------------------
    We iterate oldest → newest through the deque, applying the EMA accumulation
    rule:  acc = alpha * (w_i * score_i) + (1 − alpha) * acc
    where w_i is the raw score dict and score_i the detection confidence.
    The final values are divided by the accumulated weight sum to keep the
    output in [0, 100] regardless of the number of frames seen so far.
    """
    if not window:
        return {name: 0.0 for name in CLASS_NAMES}

    acc      = {name: 0.0 for name in CLASS_NAMES}
    acc_w    = 0.0
    momentum = 1.0 - ema_alpha

    for scores, w in zip(window, weights):          # oldest → newest
        scaled_w = ema_alpha * w
        for name in CLASS_NAMES:
            acc[name] = scaled_w * scores[name] + momentum * acc[name]
        acc_w = scaled_w + momentum * acc_w

    if acc_w < 1e-9:
        return {name: 0.0 for name in CLASS_NAMES}

    return {name: round(acc[name] / acc_w, 2) for name in CLASS_NAMES}

# ── Audio transcription ────────────────────────────────────────────────────────────

def _transcribe_audio(audio_chunks: List[bytes]) -> Optional[str]:
    """
    Concatenate accumulated audio chunks and transcribe with Whisper.

    The chunks are raw bytes from MediaRecorder (WebM/Opus container).
    Concatenating in order is valid: the first chunk carries the container
    header; subsequent chunks are continuation data.  The assembled file is
    written to a temp directory, transcribed, then immediately deleted.

    Returns the transcript string, or None if Whisper is unavailable, no
    audio was received, or transcription fails.  Never raises — failures are
    logged and swallowed so report generation always succeeds.
    """
    if WHISPER_MODEL is None or not audio_chunks:
        return None

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            for chunk in audio_chunks:
                f.write(chunk)
            tmp_path = f.name

        # fp16=False avoids errors on CPU-only machines (fp16 requires CUDA)
        result = WHISPER_MODEL.transcribe(tmp_path, fp16=False)
        text = result.get("text", "").strip()
        return text if text else None

    except Exception as exc:
        print(f"[whisper] Transcription error: {exc}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    # ── Authenticate via query-string token ────────────────────────────────
    token = websocket.query_params.get("token")
    user = get_user_from_token(token) if token else None
    if not user:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "auth_error",
            "message": "Authentication required. Please log in first.",
        }))
        await websocket.close(code=4001)
        return

    await websocket.accept()

    # Initialise per-session state
    SESSIONS[session_id] = {
        "window":       deque(maxlen=WINDOW_SIZE),   # raw scores — last 15 frames
        "det_weights":  deque(maxlen=WINDOW_SIZE),   # MediaPipe detection confidence per frame
        "history":      [],                           # (timestamp, dominant, smoothed)
        "audio_chunks": [],                           # raw bytes from audio_chunk messages
        "start":        time.monotonic(),
    }
    session = SESSIONS[session_id]
    loop    = asyncio.get_running_loop()

    try:
        while True:
            text = await websocket.receive_text()

            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON."})
                )
                continue

            msg_type = msg.get("type")

            if msg_type == "frame":
                response = await loop.run_in_executor(
                    None, _process_frame_sync, msg, session
                )
                await websocket.send_text(json.dumps(response))

            elif msg_type == "audio_chunk":
                # Accumulate raw audio bytes for end-of-session transcription.
                raw = msg.get("data", "")
                if raw:
                    if "," in raw:
                        raw = raw.split(",", 1)[1]
                    try:
                        session["audio_chunks"].append(base64.b64decode(raw))
                    except Exception:
                        pass   # silently discard corrupt chunks

            elif msg_type == "end":
                # Transcribe audio in a thread (Whisper is CPU-heavy and sync)
                transcript: Optional[str] = await loop.run_in_executor(
                    None, _transcribe_audio, session["audio_chunks"]
                )
                if transcript:
                    print(f"[{session_id}] Transcript ({len(transcript)} chars): "
                          f"{transcript[:80]}{'\u2026' if len(transcript) > 80 else ''}")

                report_text = await _generate_report(session, transcript)
                await websocket.send_text(
                    json.dumps({"type": "report", "text": report_text})
                )
                break   # close normally after sending the report

            else:
                await websocket.send_text(
                    json.dumps({"type": "error",
                                "message": f"Unknown message type: {msg_type!r}"})
                )

    except WebSocketDisconnect:
        pass   # client disconnected — clean up silently
    finally:
        SESSIONS.pop(session_id, None)


def _process_frame_sync(msg: dict, session: dict) -> dict:
    """
    Synchronous frame-processing pipeline (called via run_in_executor so it
    doesn't block the asyncio event loop).

    Pipeline:
        base64 → numpy BGR → MediaPipe face detection → 96×96 RGB crop
        → normalise → EmotionNet → softmax scores → sliding-window smooth
    """
    timestamp: float = float(msg.get("timestamp", 0.0))

    if MODEL is None:
        return {
            "type":    "error",
            "message": "Model not loaded. Run train.py first.",
            "timestamp": timestamp,
        }

    frame = _decode_frame(msg.get("data", ""))
    if frame is None:
        return {"type": "result", "face_detected": False, "timestamp": timestamp}

    face_crop, det_score = _detect_and_crop_face(frame)
    if face_crop is None:
        return {"type": "result", "face_detected": False, "timestamp": timestamp}

    raw_scores = _run_model_tta(face_crop)

    # Advance the sliding window and compute smoothed scores.
    # The detection confidence gates how much this frame contributes: a clear
    # front-on face (score ~0.95) counts almost fully; a marginal blurry
    # detection (score ~0.55) counts roughly 60 % as much.
    session["window"].append(raw_scores)
    session["det_weights"].append(det_score)
    smoothed_scores = _smooth_window(session["window"], session["det_weights"])

    dominant   = max(smoothed_scores, key=smoothed_scores.get)
    confidence = smoothed_scores[dominant]

    # Record for end-of-session report
    session["history"].append((timestamp, dominant, dict(smoothed_scores)))

    return {
        "type":             "result",
        "face_detected":    True,
        "raw_scores":       raw_scores,
        "smoothed_scores":  smoothed_scores,
        "dominant_emotion": dominant,
        "confidence":       confidence,
        "timestamp":        timestamp,
    }


# ── Report generation ─────────────────────────────────────────────────────────

async def _generate_report(session: dict,
                           transcript: Optional[str] = None) -> str:
    """
    Generate an end-of-session summary.

    Uses facial emotion history as the primary signal.  When a Whisper
    transcript is available it is included as additional context in both
    the LLM prompt and the fallback template, allowing the report to
    correlate what was said with what was shown on the face.

    Tries the Gemini LLM path first; falls back to the deterministic template.
    """
    history: List[Tuple] = session["history"]

    if not history:
        return "No emotion data was captured during this session."

    # Aggregate statistics
    duration: float = history[-1][0] - history[0][0] if len(history) > 1 else 0.0
    total_frames    = len(history)
    emotion_counts  = {name: 0 for name in CLASS_NAMES}
    for _, dominant, _ in history:
        emotion_counts[dominant] += 1

    emotion_pct = {
        e: round(c / total_frames * 100, 1)
        for e, c in emotion_counts.items()
    }
    sorted_emotions  = sorted(emotion_pct.items(), key=lambda x: x[1], reverse=True)
    dominant_overall = sorted_emotions[0][0]

    # Try LLM-based report
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            llm_text = await _llm_report(
                history, emotion_pct, duration, api_key, transcript
            )
            if llm_text:
                return llm_text
        except Exception as exc:
            print(f"[report] LLM call failed ({exc}). Falling back to template.")

    return _template_report(duration, sorted_emotions, dominant_overall, history, transcript)


async def _llm_report(
    history:     List[Tuple],
    emotion_pct: dict,
    duration:    float,
    api_key:     str,
    transcript:  Optional[str] = None,
) -> Optional[str]:
    """
    Call Google Gemini to generate a natural-language session report.

    Requires GEMINI_API_KEY in the environment.  The free tier is sufficient —
    no billing setup is required.  Get a key at aistudio.google.com/apikey.

    Override the default model with the GEMINI_MODEL env var.
    Default: gemini-2.0-flash  (free tier, fast, high quality).
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("[report] 'google-generativeai' package not installed. Skipping LLM report.")
        return None

    model_name   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name)

    # Build a sampled timeline (≤ 20 points)
    n        = len(history)
    step     = max(1, n // 20)
    timeline = "\n".join(
        f"  t={ts:.1f}s → {dom}"
        for ts, dom, _ in history[::step]
    )

    dist_block = "\n".join(
        f"  {e}: {p}%"
        for e, p in sorted(emotion_pct.items(), key=lambda x: x[1], reverse=True)
        if p > 0
    )

    transcript_block = (
        f"\n\nAudio transcript of what was said during the session:\n\"{transcript}\""
        if transcript else ""
    )

    prompt = (
        f"You are analysing facial emotion recognition data captured from a video session.\n\n"
        f"Session duration  : {duration:.0f} seconds\n"
        f"Frames analysed   : {n}\n\n"
        f"Emotion distribution (% of frames where each emotion was dominant):\n{dist_block}\n\n"
        f"Emotional timeline (sampled every ~{step} frames):\n{timeline}"
        f"{transcript_block}\n\n"
        "Write a concise 2–3 paragraph report describing the person's emotional journey "
        "through this session. Describe what emotions were dominant, any notable shifts or "
        "transitions, and an overall characterisation of the emotional state."
        + (
            " Where relevant, connect the facial expressions to what was said in the transcript."
            if transcript else ""
        )
        + " Write in a warm, observational tone. Address the subject directly "
        "(\"You appeared...\" / \"Throughout the session...\"). "
        "Do not make clinical diagnoses."
    )

    response = await gemini_model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=600,
            temperature=0.7,
        ),
    )
    return response.text.strip()


def _template_report(
    duration:         float,
    sorted_emotions:  list,
    dominant_overall: str,
    history:          List[Tuple],
    transcript:       Optional[str] = None,
) -> str:
    """
    Build a readable, data-driven report without any external API call.
    This is the fallback and is always used when no LLM API key is set.
    """
    top    = sorted_emotions[0]   # (emotion, %)
    second = next((e for e in sorted_emotions[1:] if e[1] > 5.0),  None)
    third  = next((e for e in sorted_emotions[2:] if e[1] > 3.0),  None)

    # ── Paragraph 1: overview ─────────────────────────────────────────────────
    p1 = (
        f"Throughout the {duration:.0f}-second session, your facial expressions were analysed "
        f"across {len(history)} video frames. "
        f"Your predominant emotion was **{top[0]}**, detected in {top[1]}% of the frames."
    )
    if second:
        p1 += f" **{second[0].capitalize()}** was also notably present ({second[1]}%)"
        if third:
            p1 += f", followed by **{third[0]}** at {third[1]}%."
        else:
            p1 += "."

    # ── Paragraph 2: timeline analysis ────────────────────────────────────────
    mid         = len(history) // 2
    first_half  = [dom for _, dom, _ in history[:mid]]
    second_half = [dom for _, dom, _ in history[mid:]]

    first_dom  = max(set(first_half),  key=first_half.count)  if first_half  else dominant_overall
    second_dom = max(set(second_half), key=second_half.count) if second_half else dominant_overall

    if first_dom != second_dom:
        p2 = (
            f"Notably, the session showed an emotional shift: the first half was dominated by "
            f"**{first_dom}**, while the second half transitioned toward **{second_dom}**. "
            "Such transitions are common as people naturally relax or react to evolving content "
            "during a session."
        )
    else:
        p2 = (
            f"Your emotional state remained consistently **{first_dom}** throughout the session, "
            "suggesting a clear and stable emotional baseline. This kind of consistency is typical "
            "when someone is focused, engaged, or simply at ease during the recording."
        )

    # ── Paragraph 3: system note ──────────────────────────────────────────────
    p3 = (
        "The analysis above was produced by processing each video frame through a deep learning "
        "facial emotion recognition model, using a 3-second (15-frame) sliding window to smooth "
        "out brief interruptions such as blinks or head turns. The live chart shows the full "
        "moment-by-moment probability scores for all seven tracked emotions — angry, disgust, "
        "fear, happy, neutral, sad, and surprise — across the entire duration of your session."
    )

    paragraphs = [p1, p2]
    if transcript:
        p_transcript = (
            f"**What you said:** \"{transcript}\"\n\n"
            "Your spoken words provide additional context for the facial expressions observed above."
        )
        paragraphs.append(p_transcript)
    paragraphs.append(p3)
    return "\n\n".join(paragraphs)


@app.get("/health")
async def health() -> dict:
    return {
        "status":              "ok",
        "model_loaded":        MODEL is not None,
        "device":              str(DEVICE),
        "classes":             CLASS_NAMES,
        "window_size":         WINDOW_SIZE,
        "audio_transcription": WHISPER_MODEL is not None,
        "whisper_model":       os.getenv("WHISPER_MODEL", "base") if WHISPER_MODEL else None,
    }
