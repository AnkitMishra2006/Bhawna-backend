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
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
import torch
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Make sure model.py is importable when the working directory is not backend/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import EmotionNet

# ── Load environment variables from optional .env file ──────────────────────
load_dotenv(os.path.join(HERE, ".env"))

# ── Authentication (Google OAuth + JWT + MongoDB) ───────────────────────────
from auth import auth_router, get_analysis_collection, get_user_from_token, init_db

# ── Globals loaded once at startup ──────────────────────────────────────────
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL:  Optional[EmotionNet] = None
MEAN:   List[float] = [0.5, 0.5, 0.5]
STD:    List[float] = [0.5, 0.5, 0.5]
CLASS_NAMES: List[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# MediaPipe face detector — initialised once, reused across all sessions.
# IMPORTANT: FaceDetector.detect() is serialised through _MP_LOCK so
# concurrent WebSocket sessions don't race on shared internal state.
FACE_DETECTOR: Optional[Any] = None
_MP_LOCK = threading.Lock()

# Whisper speech-to-text — loaded at startup if openai-whisper is installed.
WHISPER_MODEL: Optional[Any] = None
ANALYSIS_COLLECTION: Optional[Any] = None

# ── Per-session state ────────────────────────────────────────────────────────
# key = session_id (string from the URL path)
SESSIONS: Dict[str, dict] = {}

# The sliding window size (15 frames ≈ 3 s at 5 fps) as used by the project spec
WINDOW_SIZE = 15

DEFAULT_REPORT_CONTEXT_KEY = "general"
REPORT_CONTEXT_PRESETS: Dict[str, Dict[str, str]] = {
    "general": {
        "label": "General emotional snapshot",
        "focus_prompt": "Provide a balanced reading of emotional flow, transitions, and stability.",
        "template_hint": "Use a broad and neutral interpretation of the emotional timeline.",
        "default_objective": "Understand the overall emotional trajectory and key shifts.",
        "safety_note": "Keep the interpretation observational and non-judgmental.",
    },
    "candidate_interview": {
        "label": "Candidate interview review",
        "focus_prompt": "Prioritise confidence, stress regulation, recovery after hard questions, and communication composure.",
        "template_hint": "Frame insights as interview-readiness signals and coaching opportunities.",
        "default_objective": "Assess interview confidence and pressure handling.",
        "safety_note": "Avoid hiring recommendations; focus on behavioral observation.",
    },
    "education": {
        "label": "Teaching and learning",
        "focus_prompt": "Focus on engagement rhythm, confusion windows, and signs of sustained attention.",
        "template_hint": "Highlight moments that may correspond to comprehension or cognitive overload.",
        "default_objective": "Measure engagement and identify confusing moments.",
        "safety_note": "Do not infer intelligence or capability from expressions.",
    },
    "customer_support": {
        "label": "Customer support QA",
        "focus_prompt": "Analyse empathy signals, calmness under friction, and de-escalation consistency.",
        "template_hint": "Frame feedback around service quality and emotional resilience.",
        "default_objective": "Evaluate empathy and emotional control during difficult interactions.",
        "safety_note": "Avoid personal judgments and stick to observable patterns.",
    },
    "sales_pitch": {
        "label": "Sales or persuasion",
        "focus_prompt": "Assess conviction, emotional energy, trust-building windows, and momentum drop-offs.",
        "template_hint": "Interpret shifts in emotional intensity as persuasion-strength clues.",
        "default_objective": "Improve persuasive confidence and trust-building moments.",
        "safety_note": "Do not claim conversion outcomes from emotion data alone.",
    },
    "public_speaking": {
        "label": "Public speaking coaching",
        "focus_prompt": "Map stage confidence arc, anxiety regulation, and audience-facing presence.",
        "template_hint": "Translate emotional transitions into speaking-coaching cues.",
        "default_objective": "Coach confidence and steady stage presence.",
        "safety_note": "Keep guidance constructive and non-clinical.",
    },
    "content_creation": {
        "label": "Creator performance",
        "focus_prompt": "Evaluate camera authenticity, emotional pacing, and perceived engagement pull.",
        "template_hint": "Connect the timeline to creator presence and likely audience resonance.",
        "default_objective": "Improve camera presence and emotional pacing.",
        "safety_note": "Avoid claims about audience metrics without supporting data.",
    },
    "ux_research": {
        "label": "UX research session",
        "focus_prompt": "Emphasise friction signals, confusion clusters, and delight windows tied to interaction flow.",
        "template_hint": "Present emotion changes as product-experience evidence.",
        "default_objective": "Identify UX friction points and positive interaction moments.",
        "safety_note": "Treat this as directional evidence, not conclusive usability proof.",
    },
    "therapy_coaching": {
        "label": "Wellbeing coaching",
        "focus_prompt": "Offer reflective emotional insights in supportive language while avoiding diagnosis.",
        "template_hint": "Focus on self-awareness and practical emotional regulation reflection.",
        "default_objective": "Support reflective self-awareness in a non-clinical context.",
        "safety_note": "Never provide clinical diagnosis or treatment advice.",
    },
    "medical_observation": {
        "label": "Clinical observation",
        "focus_prompt": "Produce a structured observational summary suitable for clinician review.",
        "template_hint": "Use clear observational language and avoid diagnostic conclusions.",
        "default_objective": "Create structured observational notes for clinical review.",
        "safety_note": "This output is observational only and is not a diagnosis.",
    },
}


def _normalise_report_context(raw_context: Any) -> Dict[str, str]:
    """Coerce arbitrary client payload into a safe, known report context shape."""
    payload = raw_context if isinstance(raw_context, dict) else {}

    requested_key = str(payload.get("key", "")).strip().lower()
    key = requested_key if requested_key in REPORT_CONTEXT_PRESETS else DEFAULT_REPORT_CONTEXT_KEY
    preset = REPORT_CONTEXT_PRESETS[key]

    label = str(payload.get("label") or preset["label"]).strip() or preset["label"]
    objective = str(payload.get("objective") or "").strip() or preset["default_objective"]
    extra_notes = str(payload.get("extra_notes") or "").strip()

    return {
        "key": key,
        "label": label,
        "objective": objective,
        "extra_notes": extra_notes,
        "focus_prompt": preset["focus_prompt"],
        "template_hint": preset["template_hint"],
        "safety_note": preset["safety_note"],
    }


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI application."""
    global MODEL, MEAN, STD, CLASS_NAMES, FACE_DETECTOR, WHISPER_MODEL, ANALYSIS_COLLECTION

    # ── Initialise authentication database ────────────────────────────────────
    init_db()
    try:
        ANALYSIS_COLLECTION = get_analysis_collection()
        print("[startup] Analysis collection ready.")
    except Exception as exc:
        ANALYSIS_COLLECTION = None
        print(f"[startup] WARNING: analysis collection unavailable ({exc})")

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

    # ── Initialise MediaPipe face detector (Tasks API — mediapipe 0.10+) ────────
    face_model_path = os.path.join(HERE, "blaze_face_short_range.tflite")
    if not os.path.exists(face_model_path):
        import urllib.request
        print("[startup] Downloading MediaPipe face model (~1 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
            face_model_path,
        )
    FACE_DETECTOR = _mp_vision.FaceDetector.create_from_options(
        _mp_vision.FaceDetectorOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=face_model_path),
            min_detection_confidence=0.5,
        )
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
) -> Tuple[Optional[np.ndarray], float, Optional[Dict[str, float]]]:
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

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _MP_LOCK:
        results = FACE_DETECTOR.detect(mp_img)
    if not results.detections:
        return None, 0.0, None

    # Pick the detection with the largest bounding-box area (pixel coordinates)
    best = max(
        results.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
    )

    # MediaPipe detection score: probability that a face is present [0, 1]
    det_score: float = float(best.categories[0].score) if best.categories else 1.0

    bb = best.bounding_box   # pixel coords: origin_x, origin_y, width, height
    bw = bb.width
    bh = bb.height

    x0 = int(bb.origin_x - pad * bw)
    y0 = int(bb.origin_y - pad * bh)
    x1 = int(bb.origin_x + (1.0 + pad) * bw)
    y1 = int(bb.origin_y + (1.0 + pad) * bh)

    # Clamp to frame bounds
    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(w, x1);  y1 = min(h, y1)

    if x1 <= x0 or y1 <= y0:
        return None, 0.0, None

    box_w = x1 - x0
    box_h = y1 - y0
    face_box = {
        "x": round(x0 / max(w, 1), 6),
        "y": round(y0 / max(h, 1), 6),
        "width": round(box_w / max(w, 1), 6),
        "height": round(box_h / max(h, 1), 6),
    }

    crop = rgb[y0:y1, x0:x1]

    # ── Face alignment: rotate so the eye line is horizontal ───────────────
    # MediaPipe FaceDetection provides 6 keypoints per detection:
    #   index 0 = right eye (person's right — appears on the LEFT of screen)
    #   index 1 = left  eye (person's left  — appears on the RIGHT of screen)
    # Training images were aligned (eyes roughly level).  Real webcam users
    # tilt their heads, creating a train-inference distribution mismatch.
    # We correct this by rotating the crop around the eye midpoint so the
    # model always sees a level face, just as during training.
    kps = best.keypoints   # NormalizedKeypoint: .x and .y in [0, 1]
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
    return crop, det_score, face_box   # HWC uint8 RGB, detection confidence [0, 1]


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


def _emotion_distribution_from_history(history: List[Tuple]) -> Dict[str, float]:
    """Convert dominant-emotion counts to percentages for report summaries."""
    if not history:
        return {name: 0.0 for name in CLASS_NAMES}

    counts = {name: 0 for name in CLASS_NAMES}
    for _, dominant, _ in history:
        counts[dominant] += 1

    total = len(history)
    return {name: round(count / total * 100, 2) for name, count in counts.items()}


def _serialise_analysis_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Mongo document fields into JSON-safe response payload."""
    return {
        "id": str(doc.get("_id")),
        "session_id": doc.get("session_id"),
        "backend": doc.get("backend"),
        "user": {
            "id": doc.get("user_id"),
            "email": doc.get("user_email"),
            "name": doc.get("user_name"),
        },
        "created_at": doc.get("created_at"),
        "ended_at": doc.get("ended_at"),
        "duration_seconds": doc.get("duration_seconds", 0.0),
        "total_frames": doc.get("total_frames", 0),
        "detected_frames": doc.get("detected_frames", 0),
        "face_detection_rate": doc.get("face_detection_rate", 0.0),
        "emotion_distribution": doc.get("emotion_distribution", {}),
        "timeline": doc.get("timeline", []),
        "transcript": doc.get("transcript"),
        "report_text": doc.get("report_text", ""),
        "report_context": doc.get("report_context"),
    }


def _extract_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """Parse bearer token and return authenticated user payload."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None
    return get_user_from_token(token)


def _persist_analysis_sync(
    session_id: str,
    session: Dict[str, Any],
    user: Dict[str, Any],
    transcript: Optional[str],
    report_text: str,
) -> Optional[str]:
    """Write completed session metrics/report to MongoDB; returns inserted id."""
    if ANALYSIS_COLLECTION is None:
        return None

    history: List[Tuple] = session.get("history", [])
    frame_events: List[Dict[str, Any]] = session.get("frame_events", [])

    if history:
        duration = max(0.0, float(history[-1][0]) - float(history[0][0]))
    elif frame_events:
        duration = max(0.0, float(frame_events[-1].get("timestamp", 0.0)))
    else:
        duration = 0.0

    detected_frames = len(history)
    total_frames = len(frame_events)
    detection_rate = round((detected_frames / total_frames) * 100, 2) if total_frames else 0.0

    doc = {
        "session_id": session_id,
        "backend": "custom",
        "user_id": user.get("id"),
        "user_email": user.get("email"),
        "user_name": user.get("name"),
        "created_at": float(session.get("created_at", time.time())),
        "ended_at": time.time(),
        "duration_seconds": round(duration, 3),
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "face_detection_rate": detection_rate,
        "emotion_distribution": _emotion_distribution_from_history(history),
        "timeline": frame_events,
        "transcript": transcript,
        "report_text": report_text,
        "report_context": session.get("report_context"),
    }

    try:
        result = ANALYSIS_COLLECTION.insert_one(doc)
        return str(result.inserted_id)
    except Exception as exc:
        print(f"[db] Failed to persist analysis for {session_id}: {exc}")
        return None

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
        "frame_events": [],                           # full frame timeline for DB report
        "audio_chunks": [],                           # raw bytes from audio_chunk messages
        "start":        time.monotonic(),
        "created_at":   time.time(),
        "backend":      "custom",
        "report_context": _normalise_report_context(None),
        "user":         user,
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
                session["report_context"] = _normalise_report_context(
                    msg.get("report_context")
                )
                audio_chunks: List[bytes] = session["audio_chunks"]
                chunk_count = len(audio_chunks)
                chunk_bytes = sum(len(chunk) for chunk in audio_chunks)
                print(
                    f"[{session_id}] Audio chunks received: {chunk_count}, "
                    f"total bytes: {chunk_bytes}"
                )
                # Transcribe audio in a thread (Whisper is CPU-heavy and sync)
                transcript: Optional[str] = await loop.run_in_executor(
                    None, _transcribe_audio, audio_chunks
                )
                if transcript:
                    print(f"[{session_id}] Transcript ({len(transcript)} chars): "
                          f"{transcript[:80]}{'...' if len(transcript) > 80 else ''}")
                else:
                    print(
                        f"[{session_id}] Transcript unavailable "
                        f"(whisper_loaded={WHISPER_MODEL is not None}, "
                        f"audio_chunks={chunk_count})"
                    )

                report_text = await _generate_report(
                    session,
                    transcript,
                    session.get("report_context"),
                )
                analysis_id: Optional[str] = await loop.run_in_executor(
                    None,
                    _persist_analysis_sync,
                    session_id,
                    session,
                    user,
                    transcript,
                    report_text,
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "report",
                            "text": report_text,
                            "analysis_id": analysis_id,
                            "session_id": session_id,
                            "backend": "custom",
                            "report_context": session.get("report_context"),
                        }
                    )
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
    frame_index = len(session.get("frame_events", []))

    if MODEL is None:
        return {
            "type":    "error",
            "message": "Model not loaded. Run train.py first.",
            "timestamp": timestamp,
        }

    frame = _decode_frame(msg.get("data", ""))
    if frame is None:
        session["frame_events"].append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "face_detected": False,
            }
        )
        return {"type": "result", "face_detected": False, "timestamp": timestamp}

    face_crop, det_score, face_box = _detect_and_crop_face(frame)
    if face_crop is None:
        session["frame_events"].append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "face_detected": False,
            }
        )
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
    session["frame_events"].append(
        {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "face_detected": True,
            "dominant_emotion": dominant,
            "confidence": confidence,
            "raw_scores": raw_scores,
            "smoothed_scores": smoothed_scores,
            "face_box": face_box,
            "detection_confidence": round(det_score, 4),
        }
    )

    return {
        "type":             "result",
        "face_detected":    True,
        "raw_scores":       raw_scores,
        "smoothed_scores":  smoothed_scores,
        "dominant_emotion": dominant,
        "confidence":       confidence,
        "face_box":         face_box,
        "timestamp":        timestamp,
    }


# ── Report generation ─────────────────────────────────────────────────────────

async def _generate_report(session: dict,
                           transcript: Optional[str] = None,
                           report_context: Optional[Dict[str, str]] = None) -> str:
    """
    Generate an end-of-session summary.

    Uses facial emotion history as the primary signal.  When a Whisper
    transcript is available it is included as additional context in both
    the LLM prompt and the fallback template, allowing the report to
    correlate what was said with what was shown on the face.

    Tries the Gemini LLM path first; falls back to the deterministic template.
    """
    history: List[Tuple] = session["history"]
    context = report_context or _normalise_report_context(session.get("report_context"))

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
                history, emotion_pct, duration, api_key, transcript, context
            )
            if llm_text:
                return llm_text
        except Exception as exc:
            print(f"[report] LLM call failed ({exc}). Falling back to template.")

    return _template_report(
        duration,
        sorted_emotions,
        dominant_overall,
        history,
        transcript,
        context,
    )


async def _llm_report(
    history:     List[Tuple],
    emotion_pct: dict,
    duration:    float,
    api_key:     str,
    transcript:  Optional[str] = None,
    report_context: Optional[Dict[str, str]] = None,
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

    context = report_context or _normalise_report_context(None)
    context_block = (
        f"\n\nReport lens: {context['label']}\n"
        f"Primary objective: {context['objective']}\n"
        f"Lens guidance: {context['focus_prompt']}\n"
        f"Safety note: {context['safety_note']}"
    )
    if context.get("extra_notes"):
        context_block += f"\nAdditional analyst notes: {context['extra_notes']}"

    prompt = (
        f"You are analysing facial emotion recognition data captured from a video session.\n\n"
        f"Session duration  : {duration:.0f} seconds\n"
        f"Frames analysed   : {n}\n\n"
        f"Emotion distribution (% of frames where each emotion was dominant):\n{dist_block}\n\n"
        f"Emotional timeline (sampled every ~{step} frames):\n{timeline}"
        f"{transcript_block}"
        f"{context_block}\n\n"
        "Write a detailed 4-part report with markdown bold section headers in this order: "
        "**Emotional Arc**, **Context-Specific Interpretation**, **Actionable Guidance**, and "
        "**Caution & Boundaries**.\n"
        "Make each section specific to this session data and the selected report lens. "
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
    report_context:   Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a readable, data-driven report without any external API call.
    This is the fallback and is always used when no LLM API key is set.
    """
    top    = sorted_emotions[0]   # (emotion, %)
    second = next((e for e in sorted_emotions[1:] if e[1] > 5.0),  None)
    third  = next((e for e in sorted_emotions[2:] if e[1] > 3.0),  None)
    context = report_context or _normalise_report_context(None)

    # ── Paragraph 1: overview ─────────────────────────────────────────────────
    p1 = (
        "**Emotional Arc**\n"
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
            "**Context-Specific Interpretation**\n"
            f"Notably, the session showed an emotional shift: the first half was dominated by "
            f"**{first_dom}**, while the second half transitioned toward **{second_dom}**. "
            f"For the selected lens (**{context['label']}**), this suggests: {context['template_hint']}"
        )
    else:
        p2 = (
            "**Context-Specific Interpretation**\n"
            f"Your emotional state remained consistently **{first_dom}** throughout the session, "
            f"suggesting a stable baseline. For the selected lens (**{context['label']}**), "
            f"this supports: {context['template_hint']}"
        )

    switches = sum(
        1 for idx in range(1, len(history)) if history[idx][1] != history[idx - 1][1]
    )
    stability = 100.0 if len(history) <= 1 else round((1 - switches / (len(history) - 1)) * 100, 1)

    p3 = (
        "**Actionable Guidance**\n"
        f"Objective: {context['objective']}. Emotional stability for this session was approximately "
        f"{stability}% based on dominant-emotion transitions. "
        f"Use this as a baseline and compare future runs for trend direction rather than one-off judgment."
    )
    if context.get("extra_notes"):
        p3 += f" Additional notes considered: {context['extra_notes']}."

    # ── Paragraph 3: system note ──────────────────────────────────────────────
    p4 = (
        "**Caution & Boundaries**\n"
        f"{context['safety_note']} "
        "The analysis above was produced by processing each video frame through a deep learning "
        "facial emotion recognition model, using a 3-second (15-frame) sliding window to smooth "
        "out brief interruptions such as blinks or head turns. The live chart shows the full "
        "moment-by-moment probability scores for all seven tracked emotions — angry, disgust, "
        "fear, happy, neutral, sad, and surprise — across the entire duration of your session."
    )

    paragraphs = [p1, p2, p3]
    if transcript:
        p_transcript = (
            f"**What you said:** \"{transcript}\"\n\n"
            "Your spoken words provide additional context for the facial expressions observed above."
        )
        paragraphs.append(p_transcript)
    paragraphs.append(p4)
    return "\n\n".join(paragraphs)


@app.get("/analysis/{analysis_id}")
async def get_analysis_by_id(analysis_id: str, request: Request) -> dict:
    """Fetch one persisted analysis report for the authenticated user."""
    user = _extract_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if ANALYSIS_COLLECTION is None:
        raise HTTPException(status_code=503, detail="Analysis storage unavailable")

    try:
        object_id = ObjectId(analysis_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid analysis id") from exc

    doc = ANALYSIS_COLLECTION.find_one({"_id": object_id, "user_id": user["id"]})
    if not doc:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _serialise_analysis_doc(doc)


@app.get("/analysis/by-session/{session_id}")
async def get_analysis_by_session(session_id: str, request: Request, backend: Optional[str] = None) -> dict:
    """Fetch the latest persisted analysis for a session and authenticated user."""
    user = _extract_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if ANALYSIS_COLLECTION is None:
        raise HTTPException(status_code=503, detail="Analysis storage unavailable")

    query: Dict[str, Any] = {"session_id": session_id, "user_id": user["id"]}
    if backend:
        query["backend"] = backend

    doc = ANALYSIS_COLLECTION.find_one(query, sort=[("ended_at", -1)])
    if not doc:
        raise HTTPException(status_code=404, detail="Analysis not ready")
    return _serialise_analysis_doc(doc)


@app.get("/health")
async def health() -> dict:
    return {
        "status":              "ok",
        "model_loaded":        MODEL is not None,
        "device":              str(DEVICE),
        "classes":             CLASS_NAMES,
        "window_size":         WINDOW_SIZE,
        "analysis_storage":    ANALYSIS_COLLECTION is not None,
        "audio_transcription": WHISPER_MODEL is not None,
        "whisper_model":       os.getenv("WHISPER_MODEL", "base") if WHISPER_MODEL else None,
    }
