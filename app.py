import cv2
from ultralytics import YOLO

# YOLOv8 ka lightweight model load karein (yolov8n.pt sab se fast hai)
model = YOLO("yolov8n.pt")


def detect_image(image_path):
  # 1. Image read karein
  image = cv2.imread(image_path)
  if image is None:
    print(
        "❌ Image load nahi ho saki! Barah-e-karam sahi path ya file name dein."
    )
    return

  print("🔄 Image processing ho rahi hai...")

  # 2. YOLOv8 inference run karein (Sirf class 0 yani 'person' ke liye)
  results = model(image, classes=[0], verbose=False)

  # 3. Total persons count nikalein
  person_count = len(results[0].boxes)

  # 4. Bounding boxes draw karein
  annotated_image = results[0].plot()

  # 5. Count text display karein
  cv2.putText(
      annotated_image,
      f"Total Persons: {person_count}",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
      cv2.LINE_AA,
  )

  # 6. Image window show karein
  print(f"✅ Success! Detected Persons: {person_count}")
  cv2.imshow("Image Person Detection", annotated_image)
  print("💡 Window band karne ke liye keyboard ki koi bhi key press karein...")
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def detect_live_camera():
  # 1. Webcam open karein (0 default laptop webcam ke liye hai)
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    print("❌ Webcam open nahi ho saka!")
    return

  print(
      "🔴 Live Camera start ho gaya hai! Band karne ke liye keyboard se 'q'"
      " press karein."
  )

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("❌ Frame read nahi ho saka.")
      break

    # 2. Live camera par tracking run karein (class 0 = person)
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False,
    )

    # Unique tracked persons count
    if results[0].boxes.id is not None:
      person_count = len(results[0].boxes.id)
    else:
      person_count = 0

    # 3. Bounding boxes draw karein
    annotated_frame = results[0].plot()

    # 4. Live count screen par display karein
    cv2.putText(
        annotated_frame,
        f"Live Persons Count: {person_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 5. Frame show karein
    cv2.imshow("Live Camera Person Detection & Tracking", annotated_frame)

    # 'q' press karne par live camera band ho jaye ga
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


# --- Main Menu ---
if __name__ == "__main__":
  print("==========================================")
  print("   YOLOv8 Person Detection & Counting System")
  print("==========================================")
  print("1. Image Detection (Tasveer par check karein)")
  print("2. Live Camera Detection (Webcam par live check karein)")

  choice = input(
      "\nApni marzi ka option select karein (1 ya 2): "
  ).strip()

  if choice == "1":
    img_path = input(
        "Apni image ka path ya file name dein (misal ke tor par"
        " person.jpg): "
    ).strip()
    detect_image(img_path)
  elif choice == "2":
    detect_live_command = detect_live_camera()
  else:
    print("❌ Ghalat option select kiya gaya hai! Program band ho raha hai.")
