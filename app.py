import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Smart Person Counter AI", page_icon="👥", layout="wide"
)

# App Title & Styling
st.title("👥 Smart Human Counting & Tracking System (YOLOv8)")
st.markdown(
    "Google Antigravity environment par mabni YOLOv8 aur Streamlit ka Person"
    " Counter app."
)

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)


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

# Video Uploader Component
uploaded_file = st.file_uploader(
    "Apni video file yahan upload karein...", type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
  # Temporary file create karna taake OpenCV read kar sakay
  tfile = tempfile.NamedTemporaryFile(delete=False)
  tfile.write(uploaded_file.read())
  video_path = tfile.name

  # Layout: Two columns
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("🎬 Original Video")
    st.video(uploaded_file)

  with col2:
    st.subheader("📊 Live AI Detection & Count")
    st_frame = st.empty()
    metric_placeholder = st.empty()

  if st.button("🚀 Start Counting & Tracking"):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video writer setup
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

      # YOLOv8 Tracking execution (Sirf class 0 yani 'person' ko filter karne ke liye)
      results = model.track(
          frame,
          persist=True,
          tracker="bytetrack.yaml",
          conf=confidence,
          classes=[0],
          verbose=False,
      )

      # Unique tracked persons count
      if results[0].boxes.id is not None:
        person_count = len(results[0].boxes.id)
      else:
        person_count = 0

      # Draw bounding boxes
      annotated_frame = results[0].plot()

      # Text display on frame
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

      # Convert BGR to RGB for Streamlit
      annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
      st_frame.image(
          annotated_frame_rgb, channels="RGB", use_container_width=True
      )
      metric_placeholder.metric(
          label="Current Active Persons", value=person_count
      )

      # Write frame to output video
      out.write(annotated_frame)

      # Update progress bar
      if total_frames > 0:
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    st.success("Processing Mukammal Ho Gayi Hai! 🎉")

    # Download Button for Processed Video
    with open(out_path, "rb") as f:
      st.download_button(
          label="📥 Download Processed Video",
          data=f,
          file_name="smart_counted_output.mp4",
          mime="video/mp4",
      )