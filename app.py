import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Smart Person Counter AI", page_icon="👥", layout="wide"
)

# App Title & Styling
st.title("👥 Smart Human Counting & Tracking System (YOLOv8)")
st.markdown(
    "Is app ki madad se aap **Images**, **Live Camera Snapshot**, aur **Videos"
    " (with Tracking)** par humans detect aur count kar sakte hain."
)

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# Mode Selection (3 Options)
app_mode = st.sidebar.selectbox(
    "Mode Select Karein:",
    [
        "🖼️ Image Detection",
        "🔴 Live Camera Snapshot",
        "🎬 Video Upload & Tracking",
    ],
)


# Load YOLOv8 Model
@st.cache_resource
def load_model():
  model = YOLO("yolov8n.pt")
  return model


try:
  model = load_model()
  st.sidebar.success("YOLOv8 Model loaded successfully! ✅")
except Exception as e:
  st.sidebar.error(f"Model load nahi ho saka: {e}")


# ==========================================
# 1. IMAGE DETECTION MODE
# ==========================================
if app_mode == "🖼️ Image Detection":
  st.subheader("🖼️ Upload Image for Person Detection")

  uploaded_image = st.file_uploader(
      "Apni image yahan upload karein...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
      st.subheader("Original Image")
      st.image(
          cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
          channels="RGB",
          use_container_width=True,
      )

    with col2:
      st.subheader("AI Detection Result")

      if st.button("🚀 Detect Persons in Image"):
        with st.spinner("Processing image..."):
          results = model(image, conf=confidence, classes=[0], verbose=False)
          person_count = len(results[0].boxes)
          annotated_image = results[0].plot()

          st.metric(label="Total Persons Detected", value=person_count)
          st.image(
              cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
              channels="RGB",
              use_container_width=True,
          )
          st.success(f"Mukammal! {person_count} persons detect hue hain.")


# ==========================================
# 2. LIVE CAMERA SNAPSHOT MODE
# ==========================================
elif app_mode == "🔴 Live Camera Snapshot":
  st.subheader("🔴 Live Camera Snapshot Detection")
  st.markdown(
      "Neeche diye gaye camera button par click karke apni live tasveer"
      " capture karein. AI foran us par detection kar dega!"
  )

  camera_image = st.camera_input("Apne camera se snapshot lein")

  if camera_image is not None:
    bytes_data = camera_image.getvalue()
    np_image = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    results = model(frame, conf=confidence, classes=[0], verbose=False)
    person_count = len(results[0].boxes)
    annotated_frame = results[0].plot()

    st.metric(label="Live Detected Persons", value=person_count)
    st.image(
        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True,
    )
    st.success(
        f"Live Camera Detection Successful! Total Persons: {person_count}"
    )


# ==========================================
# 3. VIDEO UPLOAD & TRACKING MODE
# ==========================================
elif app_mode == "🎬 Video Upload & Tracking":
  st.subheader("🎬 Video Upload, Tracking & Counting")

  uploaded_file = st.file_uploader(
      "Apni video file yahan upload karein...", type=["mp4", "avi", "mov"]
  )

  if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns(2)

    with col1:
      st.subheader("🎬 Original Video")
      st.video(uploaded_file)

    with col2:
      st.subheader("📊 Live AI Tracking & Count")
      st_frame = st.empty()
      metric_placeholder = st.empty()

    if st.button("🚀 Start Counting & Tracking"):
      cap = cv2.VideoCapture(video_path)
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = int(cap.get(cv2.CAP_PROP_FPS))

      out_path = "output_tracked.mp4"
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

      progress_bar = st.progress(0)
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      frame_count = 0

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            classes=[0],
            verbose=False,
        )

        if results[0].boxes.id is not None:
          person_count = len(results[0].boxes.id)
        else:
          person_count = 0

        annotated_frame = results[0].plot()

        cv2.putText(
            annotated_frame,
            f"Persons: {person_count}",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(
            annotated_frame_rgb, channels="RGB", use_container_width=True
        )
        metric_placeholder.metric(
            label="Current Active Persons", value=person_count
        )

        out.write(annotated_frame)

        if total_frames > 0:
          progress_bar.progress(min(frame_count / total_frames, 1.0))

      cap.release()
      out.release()
      st.success("Processing Mukammal Ho Gayi Hai! 🎉")

      with open(out_path, "rb") as f:
        st.download_button(
            label="📥 Download Processed Video",
            data=f,
            file_name="smart_counted_output.mp4",
            mime="video/mp4",
        )
