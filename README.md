**Overview**

The Hybrid Crowd and Traffic Flow Analysis system leverages YOLO-based object detection and dense optical flow to analyze movement patterns in public spaces. This real-time solution detects pedestrians and vehicles, monitors crowd density, detects anomalies, and visualizes movement trajectories.

**Features**

Real-time Object Detection: Uses YOLOv8 for identifying pedestrians and vehicles.
Dense Optical Flow Analysis: Computes movement magnitude and direction.
Anomaly Detection: Identifies unusual motion patterns.
Trajectory Tracking: Predicts movement trends.
Heatmap Visualization: Generates a density heatmap of high-traffic areas.
Multi-threaded Processing: Uses concurrent.futures for optimized performance.

**Tech Stack**

Programming Language: Python
Computer Vision: OpenCV, YOLOv8 (Ultralytics)
Machine Learning: NumPy, SciPy
Data Processing & Visualization: Matplotlib, Gaussian Filtering
Parallel Computing: concurrent.futures

**Arguments**

--video-source : Path to video file or 0 for webcam.
--model-path : Path to the YOLO model file (default: yolov8n.pt).
--output-resolution : Output video resolution (width, height).

**Example Output**

Annotated video feed with detected objects, flow vectors, and density metrics.
Heatmap overlays to highlight high-movement areas.
CSV file output (hybrid_metrics.csv) containing frame-wise motion statistics.
