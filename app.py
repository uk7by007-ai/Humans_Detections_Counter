import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="YOLOv8 Image & Live Camera Detection", page_icon="📸", layout="wide"
)

# App Title & Styling
st.title("👥 YOLOv8 Person Detection: Image & Live Camera")
st.markdown(
    "Is app ki madad se aap **Images** aur **Live Camera Snapshot** par humans"
    " detect aur count kar sakte hain."
)

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# Mode Selection
app_mode = st.sidebar.selectbox(
    "Mode Select Karein:",
    ["🖼️ Image Detection", "🔴 Live Camera / Snapshot Detection"],
)


# Load YOLOv8 Model (Cached taake bar bar load na ho)
@st.cache_resource
def load_model():
  return YOLO("yolov8n.pt")


try:
  model = load_model()
  st.sidebar.success("YOLOv8 Model loaded successfully! ✅")
except Exception as e:
  st.sidebar.error(f"Model load nahi ho saka: {e}")


# --- 1. IMAGE DETECTION MODE ---
if app_mode == "🖼️ Image Detection":
  st.subheader("🖼️ Upload Image for Person Detection")

  uploaded_image = st.file_uploader(
      "Apni image yahan upload karein...", type=["jpg", "jpeg", "png"]
  )

  if uploaded_image is not None:
    # Read image bytes using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1.ch) if False else cv2.imdecode(
        file_bytes, cv2.IMREAD_COLOR
    )

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
          # Run YOLO inference (classes=[0] for person)
          results = model(image, conf=confidence, classes=[0], verbose=False)
          person_count = len(results[0].boxes)

          # Draw bounding boxes
          annotated_image = results[0].plot()

          # Display count metric
          st.metric(label="Total Persons Detected", value=person_count)

          # Show processed image
          st.image(
              cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
              channels="RGB",
              use_container_width=True,
          )
          st.success(f"Mukammal! {person_count} persons detect hue hain.")


# --- 2. LIVE CAMERA / SNAPSHOT MODE ---
elif app_mode == "🔴 Live Camera / Snapshot Detection":
  st.subheader("🔴 Live Camera Snapshot Detection")
  st.markdown(
      "Neeche diye gaye camera button par click karke apni live tasveer"
      " capture karein. AI foran us par detection kar dega!"
  )

  # Streamlit native camera input (Never hangs on web apps)
  camera_image = st.camera_input("Apne camera se snapshot lein")

  if camera_image is not None:
    # Convert camera image to OpenCV format
    bytes_data = camera_image.getvalue()
    np_image = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model(frame, conf=confidence, classes=[0], verbose=False)
    person_count = len(results[0].boxes)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display results
    st.metric(label="Live Detected Persons", value=person_count)
    st.image(
        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True,
    )
    st.success(
        f"Live Camera Detection Successful! Total Persons: {person_count}"
    )
