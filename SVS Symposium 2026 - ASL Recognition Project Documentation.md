# SVS Symposium 2026: STEAM - ASL Recognition

Project documentation · November 2026

## 1. Overview

This project is a real-time web application that recognizes letters of the American Sign Language (ASL) alphabet from a webcam. A person shows a hand sign to the camera, and the app draws a box around each hand and displays the letter it thinks is being signed, with a confidence percentage.

The app was built for the SVS Symposium 2026 (STEAM) and runs in a web browser, either on Streamlit Cloud or locally on a laptop.

The code is hosted on GitHub in the repository `ericktitusk/symposium2026`.

## 2. What the app does

- Opens the webcam in the browser and shows a mirrored (selfie-style) live video.
- Detects up to two hands at the same time.
- Draws a box around each hand with the predicted letter and confidence. The box is green when confidence is above 70% and orange otherwise.
- Smooths the result over the last 7 frames, so the letter does not flicker.
- Shows the model's top 3 guesses per hand in the top-left corner of the video, which is useful for testing.
- Offers a sidebar test option, "Mirror landmarks before classifying", for checking left-hand and right-hand differences.
- Recognizes the static letters A to Y. J and Z need hand motion, so they are not supported.

## 3. How it works

The app uses a two-step pipeline that runs on every video frame:

1. **Hand detection (MediaPipe Hand Landmarker).** Google's MediaPipe finds each hand in the frame and returns 21 landmark points (for example the wrist, knuckles and fingertips). Each point has x, y and z coordinates.
2. **Letter classification (small neural network).** The 63 numbers (21 points × x, y, z) are passed to a small classifier that outputs a probability for each letter. The highest probability is shown as the prediction.

The classifier looks at the shape of the hand through its landmarks and not at the camera picture itself. This makes it very small and fast compared with an image-based model.

### Why the classifier runs on NumPy

The original classifier is a Keras model. Loading TensorFlow on Streamlit Cloud together with MediaPipe caused the server to crash (a segmentation fault). To avoid this, the Keras model was converted once into plain NumPy weights. The deployed app only needs NumPy to run the classifier, so it installs faster, starts faster and no longer depends on TensorFlow.

### Mirroring

The landmarks are detected on the original, un-mirrored camera image, because that is how the model's author used it. The picture is flipped only for display, and the boxes are mirrored to match.

## 4. Technology used

- **Streamlit**: the web interface.
- **streamlit-webrtc**: streams the webcam video from the browser to the Python code and back.
- **MediaPipe**: hand detection and landmark extraction (`hand_landmarker.task`).
- **NumPy**: runs the letter classifier.
- **OpenCV and PyAV**: video frame handling and drawing.
- **Pre-trained model**: `AmimulBmeIU/asl-alphabet-classifier` on Hugging Face (MIT license). It classifies static ASL letters A-Y from 21 hand landmarks. The model's author reports a validation accuracy of about 94%, measured on their own data. Accuracy on other people, cameras and lighting can be lower.

## 5. Project files

- `app.py`: the Streamlit application (interface, video processing, drawing).
- `asl_numpy.py`: a small NumPy runner for the classifier.
- `asl_weights.npz`: the classifier's weights, exported from the Keras model.
- `labels.json`: the list of letters, in the order the model outputs them.
- `hand_landmarker.task`: the MediaPipe hand detection model.
- `export_weights.py`: a one-time script that creates `asl_weights.npz` and `labels.json` from the original Keras model. It is only needed again if the model is changed.
- `requirements.txt`: Python packages installed on Streamlit Cloud (streamlit, streamlit-webrtc, av, opencv-python-headless, mediapipe, numpy).
- `packages.txt`: system libraries installed on Streamlit Cloud (`libgl1`, `libglib2.0-0`, `libsm6`, `libxrender1`, `libxext6`, `libegl1`, `libgles2`).

## 6. Running the app locally

1. Open Terminal in the project folder.
2. Create and activate a virtual environment: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install the packages: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. Open `http://localhost:8501` in the browser, click **Start** under the video and allow camera access.

Note: the app must be started with `streamlit run app.py`, not `python app.py`.

### Recreating the classifier files (only if needed)

The Keras model is only needed to create `asl_weights.npz` and `labels.json`. On a computer where TensorFlow installs, run:

`pip install tensorflow keras huggingface_hub`

`python export_weights.py`

The script checks that the NumPy output matches the Keras output and prints the result.

## 7. Deploying on Streamlit Cloud

1. Push these files to the GitHub repository: `app.py`, `asl_numpy.py`, `asl_weights.npz`, `labels.json`, `hand_landmarker.task`, `requirements.txt` and `packages.txt`.
2. In Streamlit Cloud, create an app that points to `app.py` in the repository.
3. After changing `packages.txt`, use **Reboot app** so the system libraries are installed.
4. Open the public link, click **Start** and allow the camera.

Free Streamlit apps go to sleep when they are not used, so open the link a few minutes before a demonstration.

## 8. Using the app

- Hold one hand at arm's length, in the middle of the picture, with the palm facing the camera.
- Make sure the whole hand is visible, with good even lighting and a plain background.
- Hold each letter steady for about a second.
- Read the letter and confidence above the box. The top 3 guesses in the corner show what the model considers next most likely.

## 9. Known limitations

- Only static letters A to Y are recognized. J and Z need motion.
- Some letters may be recognized less reliably than others. B was reported as hard to recognize during testing. The model reads raw landmark coordinates, so hand position, distance and angle can affect the result.
- Accuracy depends on lighting, camera quality and the person's hand. The published accuracy comes from the model author's own data.
- The app works with letters one at a time. It does not yet build words or sentences.
- On Streamlit Cloud the video travels over the internet, so a slow connection can make the video lag.

## 10. Troubleshooting

- **Error `libEGL.so.1: cannot open shared object file` on Streamlit Cloud.** MediaPipe needs graphics libraries. Add `libegl1` and `libgles2` to `packages.txt`, push, and reboot the app.
- **"Segmentation fault" on Streamlit Cloud after TensorFlow starts.** This happened when TensorFlow and MediaPipe were used together. The app now avoids TensorFlow by using NumPy.
- **`pip` cannot find `tensorflow-cpu` on a Mac.** There are no Mac versions of that package. The deployed app no longer needs TensorFlow. Use `tensorflow` locally only if you need to run `export_weights.py`.
- **Warning about `AVFFrameReceiver` implemented in both `av` and `cv2` on a Mac.** This is a duplicate-library warning from macOS. It is usually harmless. If the app crashes, report it.
- **"Missing asl\_weights.npz" message in the app.** Run `export_weights.py` once and push the two files it creates.
- **The camera does not start.** Allow camera access for the browser (on a Mac: System Settings, Privacy & Security, Camera) and close other apps that use the camera.
- **A letter is often wrong.** Check the top 3 guesses, try the sidebar mirror option, improve the lighting and keep the whole hand in view.

## 11. Symposium day checklist

- Run the app locally on the laptop instead of relying on the public link, in case the Wi-Fi is slow. After the NumPy change, the model files are inside the project, so no download is needed. Test this once with Wi-Fi turned off before the event.
- Plug in the laptop and close other apps that use the camera.
- Set up good, even lighting and a plain background.
- Practice the letters that were harder, such as B.
- Record a short screen video of the app working as a backup.

## 12. Ideas for future work

- Add a word builder, so visitors can spell their names, and word prediction.
- Retrain the classifier on samples from our own hands to improve accuracy for letters such as B, using landmarks normalized to the wrist.
- Add support for J and Z using motion.
- Build a mobile version that works offline, for example as an installable web app with the same landmark approach.

## 13. Credits

- Letter classifier: `AmimulBmeIU/asl-alphabet-classifier` on Hugging Face (MIT license).
- Hand detection: Google MediaPipe Hand Landmarker.
- Web interface: Streamlit and streamlit-webrtc.
