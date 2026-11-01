import json
import os
import time
from collections import Counter, deque

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import RTCConfiguration, VideoTransformerBase, webrtc_streamer

from asl_numpy import NumpyMLP

st.set_page_config(page_title="SVS Symposium 2026 - ASL  Recognition", layout="wide")
st.title("SVS Symposium 2026")
st.write("STEAM - ASL Recognition")
st.caption("Static letters A-Y (J and Z need motion and are not supported).")

st.markdown(
    """
    <style>
    div[data-testid="stWebRtcStreamer"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    div[data-testid="stWebRtcStreamer"] video {
        max-width: 640px !important;
        width: 100% !important;
        height: auto !important;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

WEIGHTS_PATH = "asl_weights.npz"  # created once by export_weights.py
LABELS_PATH = "labels.json"
LANDMARKER_PATH = "hand_landmarker.task"
SMOOTHING_WINDOW = 7  # majority vote over the last N frames per hand
CONF_GOOD = 70  # % threshold for green box

# Test switch (also available as a sidebar checkbox while the app runs)
MIRROR_INPUT_DEFAULT = False


@st.cache_resource
def load_classifier():
    missing = [p for p in (WEIGHTS_PATH, LABELS_PATH) if not os.path.exists(p)]
    if missing:
        st.error(
            f"Missing {', '.join(missing)}. Run `python export_weights.py` once "
            "on your computer and commit the two files it creates."
        )
        st.stop()
    mlp = NumpyMLP(WEIGHTS_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    mlp.predict(np.zeros((1, 63), dtype=np.float32))  # warm-up
    return mlp, labels


mlp, LABELS = load_classifier()


def classify(landmarks, mirror_input=False):
    """21 MediaPipe landmarks -> top 3 [(letter, confidence %), ...].

    Input is raw x, y, z (63 values). With mirror_input=True the x values are
    flipped (1 - x) before classifying, to test a left/right mismatch.
    """
    feats = np.array(
        [[(1 - lm.x) if mirror_input else lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    ).reshape(1, 63)
    probs = mlp.predict(feats)[0]
    top = np.argsort(probs)[::-1][:3]
    return [(LABELS[i], float(probs[i]) * 100) for i in top]


class ASLVideoProcessor(VideoTransformerBase):

    def __init__(self):
        # One landmarker per video session, in VIDEO mode (tracks between frames)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.history = {}  # hand index -> deque of recent letters
        self.last_ts = 0
        self.mirror_input = MIRROR_INPUT_DEFAULT

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # un-mirrored: this is what the model expects
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # MediaPipe VIDEO mode needs strictly increasing timestamps (ms)
        ts = int(time.monotonic() * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect_for_video(mp_image, ts)

        # Mirror only for display, after detection
        display = cv2.flip(img, 1)

        seen = set()
        for i, landmarks in enumerate(results.hand_landmarks or []):
            seen.add(i)
            top3 = classify(landmarks, self.mirror_input)
            letter, conf = top3[0]

            hist = self.history.setdefault(i, deque(maxlen=SMOOTHING_WINDOW))
            hist.append(letter)
            smooth_letter = Counter(hist).most_common(1)[0][0]

            # Box coordinates, mirrored to match the flipped display image
            xs = [(1 - lm.x) * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            pad = 25
            x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
            x2, y2 = min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)

            color = (0, 255, 0) if conf > CONF_GOOD else (0, 165, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                f"{smooth_letter} ({conf:.0f}%)",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # Top-3 guesses in the corner, one line per hand (for testing)
            cv2.putText(
                display,
                f"Hand {i + 1}: " + " / ".join(f"{l} {c:.0f}" for l, c in top3),
                (10, 30 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Forget history for hands that disappeared
        for k in list(self.history):
            if k not in seen:
                del self.history[k]

        return av.VideoFrame.from_ndarray(display, format="bgr24")


mirror_input = st.sidebar.checkbox(
    "Mirror landmarks before classifying (test)", value=MIRROR_INPUT_DEFAULT
)

col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    ctx = webrtc_streamer(
        key="asl-stream",
        video_processor_factory=ASLVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
            },
            "audio": False,
        },
    )

# Apply the sidebar toggle to the running video processor
if ctx.video_processor:
    ctx.video_processor.mirror_input = mirror_input
