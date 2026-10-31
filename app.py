import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from transformers import AutoImageProcessor, SiglipForImageClassification

st.set_page_config(page_title="Real-Time ASL Recognition", layout="wide")
st.title("🤟 Real-Time Multi-Hand ASL Recognition")
st.write("Cross-platform web demo powered by MediaPipe and SigLIP Transformer.")

@st.cache_resource
def load_models():
    # Load SigLIP model
    model_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipForImageClassification.from_pretrained(model_name)
    model.eval()

    # Load MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return processor, model, detector

processor, model, detector = load_models()

class ASLVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.predictions = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.frame_count += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for i, landmarks in enumerate(results.hand_landmarks):
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                pad = 25
                x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
                x2, y2 = min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)

                if self.frame_count % 3 == 0 and (x2 - x1 > 20) and (y2 - y1 > 20):
                    hand_crop = Image.fromarray(rgb[y1:y2, x1:x2])
                    inputs = processor(images=hand_crop, return_tensors="pt")
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)
                        conf, idx = torch.max(probs, dim=-1)
                    
                    self.predictions[i] = (
                        model.config.id2label[idx.item()], 
                        conf.item() * 100
                    )

                if i in self.predictions:
                    label, score = self.predictions[i]
                    color = (0, 255, 0) if score > 70 else (0, 165, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label} ({score:.0f}%)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

webrtc_streamer(key="asl-stream", video_transformer_factory=ASLVideoTransformer)