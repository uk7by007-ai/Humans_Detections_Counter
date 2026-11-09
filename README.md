# 👥 Smart Human Counting & Tracking System (YOLOv8 & Streamlit)

A professional, web-based computer vision application built with **Streamlit** and **Ultralytics YOLOv8** to detect, count, and track people in video files in real-time.

---

## 🚀 Features

* **Web UI Dashboard:** Clean and interactive user interface powered by Streamlit.
* **YOLOv8 Object Detection & Tracking:** Utilizes the lightweight `yolov8n.pt` model integrated with ByteTrack (`bytetrack.yaml`) for accurate multi-object tracking and unique ID assignment.
* **Class Filtering:** Automatically filters specifically for humans (COCO Class ID `0`) to ignore non-target objects.
* **Adjustable Confidence:** Dynamic sidebar slider to control the detection confidence threshold.
* **Real-time Processing & Metrics:** Live video frame rendering alongside active human count metrics and a progress bar.
* **Processed Video Export:** Option to download the final annotated output video directly from the web app.

---

## 🛠️ Tech Stack

* **Python** (3.8+)
* **Streamlit** (Web Framework)
* **Ultralytics YOLOv8** (AI Model)
* **OpenCV** (Video Processing & Annotation)

---

## 📁 Project Structure

```text
smart-human-counter/
│
├── app.py              # Main Streamlit application file
├── requirements.txt    # Required python dependencies
└── README.md           # Project documentation