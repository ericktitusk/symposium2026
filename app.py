import json
import os
import random
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

# Test switch (also available in the sidebar "Testing options")
MIRROR_INPUT_DEFAULT = False

# ---- Spelling game settings ----
# Words use only letters the model can see (A-Y, no J or Z) and have no double letters.
WORDS = [
    "CAT", "DOG", "SUN", "FISH", "STAR", "SKY", "CAKE", "BIRD", "FROG", "MILK",
    "RAIN", "FUN", "MATH", "ART", "PLAY", "BEAR", "LION", "DUCK", "KITE", "SNOW",
    "SHARK",
]
ALLOWED_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")  # no J, no Z
HOLD_FRAMES = 12  # frames a letter must be held (about one second)
GAME_CONF = 60  # minimum confidence (%) to count a letter in the game
CELEBRATE_SECONDS = 3

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Short tips (shown on the video during the spelling game)
LETTER_TIPS = {
    "A": "Fist, thumb along the side",
    "B": "Flat hand, thumb across palm",
    "C": "Curve the hand like a C",
    "D": "Index up, others touch thumb",
    "E": "Fingers curled onto thumb",
    "F": "OK circle, 3 fingers up",
    "G": "Sideways, index and thumb point",
    "H": "Sideways, two fingers point",
    "I": "Fist, pinky finger up",
    "K": "V fingers, thumb between",
    "L": "Index up, thumb out (an L)",
    "M": "Thumb under 3 fingers",
    "N": "Thumb under 2 fingers",
    "O": "Fingertips meet thumb (an O)",
    "P": "Like K, pointing down",
    "Q": "Like G, pointing down",
    "R": "Cross two fingers",
    "S": "Fist, thumb across the front",
    "T": "Fist, thumb between 2 fingers",
    "U": "Two fingers up, together",
    "V": "Two fingers up, apart (a V)",
    "W": "Three fingers up, apart",
    "X": "Fist, index finger hooked",
    "Y": "Thumb and pinky out",
}

# Longer descriptions (shown in the ASL alphabet guide on the page)
LETTER_GUIDE = {
    "A": "Make a fist, with the thumb resting along the side of the index finger, pointing up.",
    "B": "Hold the fingers straight up and close together, with the thumb folded across the palm.",
    "C": "Curve the fingers and thumb into the shape of the letter C, leaving a gap between them.",
    "D": "Point the index finger up. Curl the other fingers so their tips touch the tip of the thumb, making a small circle.",
    "E": "Bend all the fingers down so their tips rest against the thumb, which is tucked underneath.",
    "F": "Touch the tips of the index finger and thumb together to make a circle. Keep the other three fingers straight and slightly apart.",
    "G": "Turn the hand sideways. Point the index finger and thumb forward, parallel to each other, with the other fingers closed.",
    "H": "Turn the hand sideways. Point the index and middle fingers forward together, with the other fingers and thumb tucked in.",
    "I": "Make a fist and point the little finger (pinky) straight up.",
    "K": "Point the index and middle fingers up in a V shape. Place the thumb against the base of the middle finger, between the two fingers.",
    "L": "Point the index finger up and the thumb out to the side, making the shape of the letter L. The other fingers are closed.",
    "M": "Fold the index, middle and ring fingers down over the thumb, so the thumb is hidden under three fingers.",
    "N": "Fold the index and middle fingers down over the thumb, so the thumb is hidden under two fingers.",
    "O": "Curve all the fingers so their tips touch the tip of the thumb, making the shape of the letter O.",
    "P": "Form the same shape as K, but tilt the hand so the fingers point downward.",
    "Q": "Form the same shape as G, but tilt the hand so the index finger and thumb point downward.",
    "R": "Cross the middle finger over the index finger. The other fingers are closed.",
    "S": "Make a fist with the thumb wrapped across the front of the fingers.",
    "T": "Make a fist with the thumb tucked between the index and middle fingers.",
    "U": "Point the index and middle fingers up, held together. The thumb holds down the ring finger and little finger.",
    "V": "Point the index and middle fingers up and spread them apart, like a peace sign.",
    "W": "Point the index, middle and ring fingers up and spread them apart. The thumb holds down the little finger.",
    "X": "Make a fist and bend the index finger into a hook.",
    "Y": "Stretch out the thumb and the little finger. Fold the other three fingers down.",
}


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


def draw_star(img, cx, cy, r, color):
    """Filled 5-point star."""
    pts = []
    for k in range(10):
        ang = -np.pi / 2 + k * np.pi / 5
        rad = r if k % 2 == 0 else r * 0.45
        pts.append([int(cx + rad * np.cos(ang)), int(cy + rad * np.sin(ang))])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)


def put_centered(img, text, y, scale, color, thickness):
    """Centered text with a dark outline so it is easy to read on video."""
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = max(5, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 4)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness)


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
        self.show_top3 = True
        self._init_game()

    # ---------- spelling game ----------
    def _init_game(self):
        self.game_on = False
        self.custom_word = ""
        self.token = 0
        self.stars = 0
        self.rng = np.random.default_rng()
        self.word = random.choice(WORDS)
        self.idx = 0
        self.hold = 0
        self.celebrate_until = 0.0

    def configure(self, game_on, custom_word, token):
        """Called from the sidebar settings on every Streamlit rerun."""
        self.game_on = game_on
        if custom_word != self.custom_word:
            self.custom_word = custom_word
            self._start_word()
        elif token != self.token:
            self.token = token
            self.custom_word = custom_word
            self._start_word(force_new=True)

    def _start_word(self, force_new=False):
        if self.custom_word:
            self.word = self.custom_word
        else:
            choices = [w for w in WORDS if w != self.word] or WORDS
            self.word = random.choice(choices)
        self.idx = 0
        self.hold = 0
        self.celebrate_until = 0.0

    def _update_game(self, matched, now):
        if self.idx >= len(self.word):
            return  # celebrating
        self.hold = self.hold + 1 if matched else 0
        if self.hold >= HOLD_FRAMES:
            self.idx += 1
            self.hold = 0
            if self.idx >= len(self.word):
                self.stars += 1
                self.celebrate_until = now + CELEBRATE_SECONDS

    def _draw_hud(self, display, now):
        h, w, _ = display.shape
        word = self.word

        # dark panel at the bottom
        panel_top = h - 120
        overlay = display.copy()
        cv2.rectangle(overlay, (0, panel_top), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # hint
        if self.idx < len(word):
            hint = f"Sign the letter {word[self.idx]}"
        else:
            hint = f"You spelled {word}!"
        cv2.putText(display, hint, (20, panel_top + 28), FONT, 0.8, (255, 255, 255), 2)
        if self.idx < len(word):
            tip = LETTER_TIPS.get(word[self.idx], "")
            (hw, _), _ = cv2.getTextSize(hint, FONT, 0.8, 2)
            tip_x = 20 + hw + 18
            (tw1, _), _ = cv2.getTextSize(tip, FONT, 1.0, 1)
            tip_scale = min(0.6, (w - tip_x - 10) / max(1, tw1))
            cv2.putText(display, tip, (tip_x, panel_top + 26), FONT, tip_scale, (0, 220, 255), 1)

        # the word, letter by letter
        spacing = int(min(70, (w - 40) / max(1, len(word))))
        start_x = int((w - spacing * len(word)) / 2)
        base_y = panel_top + 78
        for i, ch in enumerate(word):
            x = start_x + i * spacing
            if i < self.idx:
                color = (0, 200, 0)  # done: green
            elif i == self.idx:
                color = (0, 220, 255)  # current: yellow
            else:
                color = (150, 150, 150)  # waiting: gray
            cv2.putText(display, ch, (x, base_y), FONT, 1.6, color, 3)
            if i == self.idx:
                cv2.line(display, (x, base_y + 8), (x + spacing - 12, base_y + 8), color, 3)

        # hold progress bar
        bar_x1, bar_x2 = 20, w - 20
        bar_y1, bar_y2 = h - 22, h - 10
        cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)
        fill = int((bar_x2 - bar_x1) * min(1.0, self.hold / HOLD_FRAMES))
        if fill > 0:
            cv2.rectangle(display, (bar_x1, bar_y1), (bar_x1 + fill, bar_y2), (0, 200, 0), -1)

        # stars counter (top-right)
        draw_star(display, w - 95, 30, 14, (0, 215, 255))
        cv2.putText(display, f"x {self.stars}", (w - 72, 40), FONT, 0.9, (0, 0, 0), 5)
        cv2.putText(display, f"x {self.stars}", (w - 72, 40), FONT, 0.9, (0, 215, 255), 2)

        # celebration
        if now < self.celebrate_until:
            for _ in range(45):
                cx = int(self.rng.integers(0, w))
                cy = int(self.rng.integers(0, h))
                color = tuple(int(c) for c in self.rng.integers(60, 255, size=3))
                cv2.circle(display, (cx, cy), int(self.rng.integers(4, 10)), color, -1)
            put_centered(display, "GREAT JOB!", h // 2, 2.0, (0, 215, 255), 5)

    # ---------- video ----------
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # un-mirrored: this is what the model expects
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        now = time.monotonic()

        # MediaPipe VIDEO mode needs strictly increasing timestamps (ms)
        ts = int(now * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect_for_video(mp_image, ts)

        # Mirror only for display, after detection
        display = cv2.flip(img, 1)

        # Start the next word once the celebration is over
        if self.game_on and self.idx >= len(self.word) and now >= self.celebrate_until:
            self._start_word()
        target = self.word[self.idx] if (self.game_on and self.idx < len(self.word)) else None
        matched = False

        seen = set()
        for i, landmarks in enumerate(results.hand_landmarks or []):
            seen.add(i)
            top3 = classify(landmarks, self.mirror_input)
            letter, conf = top3[0]

            hist = self.history.setdefault(i, deque(maxlen=SMOOTHING_WINDOW))
            hist.append(letter)
            smooth_letter = Counter(hist).most_common(1)[0][0]

            if target is not None and smooth_letter == target and conf > GAME_CONF:
                matched = True

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
                FONT,
                0.8,
                color,
                2,
            )

            # Top-3 guesses in the corner (for testing, free signing only)
            if self.show_top3 and not self.game_on:
                cv2.putText(
                    display,
                    f"Hand {i + 1}: " + " / ".join(f"{l} {c:.0f}" for l, c in top3),
                    (10, 30 + 30 * i),
                    FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Forget history for hands that disappeared
        for k in list(self.history):
            if k not in seen:
                del self.history[k]

        if self.game_on:
            self._update_game(matched, now)
            self._draw_hud(display, now)

        return av.VideoFrame.from_ndarray(display, format="bgr24")


# ---------------- Sidebar: fun zone ----------------
st.sidebar.header("Fun zone")
mode = st.sidebar.radio("What do you want to do?", ["Free signing", "Spelling game"])

custom_raw = st.sidebar.text_input("Spell your own name or word (optional)", max_chars=12)
typed_letters = [c for c in custom_raw.upper() if c.isalpha()]
custom_word = "".join(c for c in typed_letters if c in ALLOWED_LETTERS)
if len(custom_word) < len(typed_letters):
    st.sidebar.caption("J and Z need motion, so those letters are skipped.")

if st.sidebar.button("New random word"):
    st.session_state["word_token"] = st.session_state.get("word_token", 0) + 1
word_token = st.session_state.get("word_token", 0)

st.sidebar.caption(
    "In the Spelling game, hold each letter steady until the bar fills. "
    "For double letters, hold a little longer."
)

with st.sidebar.expander("Testing options"):
    mirror_input = st.checkbox(
        "Mirror landmarks before classifying (test)", value=MIRROR_INPUT_DEFAULT
    )
    show_top3 = st.checkbox("Show top-3 guesses (free signing only)", value=True)

# ---------------- Main page ----------------
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

    st.markdown("---")
    st.subheader("About this app")
    st.markdown(
        """
**How to use**
1. Click **Start** under the video and allow camera access.
2. Hold one hand at arm's length, palm facing the camera, with the whole hand in view.
3. Hold each letter steady for about a second.

A **green box** means the app is confident (above 70%). An **orange box** means it is less sure.

**Want to play?** Open the sidebar on the left and choose **Spelling game**. Sign the letters of the
word one by one to win stars, or type your own name to spell it!

**How it works**
The app finds 21 key points on each hand (knuckles, fingertips and wrist) and a small
neural network reads the shape of those points to predict the letter, live, for up to two hands.
"""
    )

    with st.expander("ASL alphabet guide: how to make each letter"):
        st.markdown(
            "Face the palm toward the camera (or slightly to the side) and keep the whole hand in view.\n\n"
            + "\n".join(f"- **{k}**: {v}" for k, v in LETTER_GUIDE.items())
            + "\n\n**J** and **Z** are drawn in the air with a moving hand, so this app cannot read them."
            + "\n\n**Letters that look alike:** A, E, M, N, S and T are all fists, so keep the thumb position clear. "
            "K and P (and G and Q) are the same shape pointing in different directions. "
            "U keeps the two fingers together and V spreads them apart."
        )

    with st.expander("Fun facts about ASL"):
        st.markdown(
            """
- ASL (American Sign Language) is a real language with its own grammar. It is not just English on the hands.
- Spelling words with the hand alphabet is called **fingerspelling**. It is often used for names and for words that do not have their own sign.
- Facial expressions are an important part of ASL.
- Many Deaf people in the United States and Canada use ASL every day.
"""
        )

# Apply the sidebar settings to the running video processor
if ctx.video_processor:
    ctx.video_processor.mirror_input = mirror_input
    ctx.video_processor.show_top3 = show_top3
    ctx.video_processor.configure(mode == "Spelling game", custom_word, word_token)
