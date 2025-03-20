import cv2 as cv
import numpy as np
import argparse
import os
import json
from collections import deque
from ultralytics import YOLO
import time
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class HybridCrowdAndTrafficFlowAnalysis:
    """
    This system uses dense optical flow and YOLO-based detection to analyze both crowd dynamics and traffic flow in public spaces.
    It focuses on advanced movement analysis, crowd density estimation, anomaly detection, trajectory prediction, and flow visualization.
    """
    def __init__(self, video_source, output_csv="hybrid_metrics.csv", model_path="yolov8n.pt", output_resolution=(1280, 720), calibration_data=None):
        self.video_source = video_source
        self.output_csv = output_csv
        self.yolo = YOLO(model_path)
        self.frame_count = 0
        self.anomaly_history = deque(maxlen=20)
        self.output_resolution = output_resolution
        self.calibration_data = calibration_data
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.trajectories = []
        self.trajectory_points = deque(maxlen=50)
        self.density_heatmap = None

    def run(self):
        cap = cv.VideoCapture(self.video_source)
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Unable to access video source.")
            return

        prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            curr_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Step 1: Detect People and Vehicles using YOLO
            objects_boxes = self.detect_objects(frame)

            # Step 2: Calculate Dense Optical Flow
            flow, magnitude, angle = self.calculate_dense_optical_flow(prev_gray, curr_gray)

            # Step 3: Detect Anomalies in Movement
            anomalies, motion_variance = self.detect_motion_anomalies(magnitude)
            self.anomaly_history.append(motion_variance)

            # Step 4: Calculate Crowd and Traffic Density
            crowd_density = self.calculate_density(objects_boxes, frame.shape)

            # Step 5: Visualize Flow Patterns and Trajectories
            mask = self.visualize_flow_patterns(frame, objects_boxes, flow, mask)
            self.update_trajectories(objects_boxes)

            # Step 6: Annotate Frame with Metrics and Display
            tracked_frame = cv.add(frame, mask)
            self.annotate_frame(tracked_frame, crowd_density, motion_variance)

            # Step 7: Generate and Overlay Heatmap
            self.density_heatmap = self.generate_density_heatmap(frame.shape)
            if self.density_heatmap is not None:
                heatmap_overlay = cv.applyColorMap(self.density_heatmap, cv.COLORMAP_JET)
                tracked_frame = cv.addWeighted(tracked_frame, 0.7, heatmap_overlay, 0.3, 0)

            # Resize output frame
            tracked_frame = cv.resize(tracked_frame, self.output_resolution)

            # Display the annotated frame
            cv.imshow("Hybrid Crowd and Traffic Flow Analysis", tracked_frame)
            prev_gray = curr_gray.copy()
            self.frame_count += 1

            # Timing for processing
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Frame {self.frame_count} processed in {processing_time:.2f} seconds")

            # Break on ESC key
            if cv.waitKey(30) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()

    def detect_objects(self, frame):
        # Use YOLO to detect people and vehicles in a given frame
        future = self.executor.submit(self.yolo, frame)
        results = future.result()
        objects_boxes = []
        for result in results:
            for bbox in result.boxes:
                if bbox.cls in [0, 2, 5, 7]:  # Assuming person, car, bus, truck classes
                    objects_boxes.append(bbox.xyxy[0].tolist())
        return objects_boxes

    @staticmethod
    def calculate_dense_optical_flow(prev_frame, curr_frame):
        flow = cv.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        return flow, magnitude, angle

    @staticmethod
    def detect_motion_anomalies(magnitude, threshold=3.0):
        anomalies = magnitude > threshold
        motion_variance = np.var(magnitude)
        return anomalies, motion_variance

    @staticmethod
    def calculate_density(objects_boxes, frame_shape):
        # Estimate density based on detected people and vehicles, divided by frame area
        object_count = len(objects_boxes)
        frame_area = frame_shape[0] * frame_shape[1]
        density = object_count / frame_area  # Objects per pixel
        return density


    def visualize_flow_patterns(self, frame, objects_boxes, flow, mask):
        """
        Visualize flow patterns in crowded areas using optical flow vectors.
        Create an overlay to show movement directions and intensity.
        """
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255  # Saturation to max
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # Value represents magnitude

        flow_map = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        blended = cv.addWeighted(frame, 0.7, flow_map, 0.3, 0)

        for box in objects_boxes:
            x1, y1, x2, y2 = map(int, box)
            roi_flow = flow[y1:y2, x1:x2]
            avg_flow_vector = np.mean(roi_flow, axis=(0, 1))
            arrow_start = ((x1 + x2) // 2, (y1 + y2) // 2)
            arrow_end = (int(arrow_start[0] + avg_flow_vector[0] * 5), int(arrow_start[1] + avg_flow_vector[1] * 5))
            cv.arrowedLine(blended, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.5)

        return blended

    def update_trajectories(self, objects_boxes):
        """
        Update trajectories of detected objects. This helps in predicting future positions.
        """
        for box in objects_boxes:
            x1, y1, x2, y2 = map(int, box)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.trajectory_points.append(centroid)
            self.trajectories.append(list(self.trajectory_points))

    def generate_density_heatmap(self, frame_shape):
        """
        Generate a heatmap based on the density of trajectory points.
        This visualizes which areas of the frame are experiencing the most movement over time.
        """
        heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        for (x, y) in self.trajectory_points:
            heatmap[y, x] += 1

        # Apply Gaussian filter to smooth the heatmap
        heatmap = gaussian_filter(heatmap, sigma=15)

        # Normalize heatmap to fit into 8-bit range
        normalized_heatmap = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return normalized_heatmap

    @staticmethod
    def annotate_frame(frame, density, motion_variance):
        cv.putText(frame, f"Density: {density:.6f} objects/pixel", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 0, 0), 2)
        cv.putText(frame, f"Motion Variance: {motion_variance:.4f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 0, 0), 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Crowd and Traffic Flow Analysis System using Optical Flow and YOLO")
    parser.add_argument("--video-source", type=str, default="0", help="Path to video file or '0' for webcam")
    parser.add_argument("--output-csv", type=str, default="hybrid_metrics.csv", help="Output CSV for metrics logging")
    parser.add_argument("--model-path", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--output-resolution", type=int, nargs=2, default=[1280, 720], help="Output resolution width height")
    parser.add_argument("--calibration-data", type=str, help="Path to calibration data file (JSON format)")
    args = parser.parse_args()

    calibration_data = None
    if args.calibration_data:
        with open(args.calibration_data, 'r') as f:
            calibration_data = json.load(f)

    hybrid_system = HybridCrowdAndTrafficFlowAnalysis(
        args.video_source,
        args.output_csv,
        args.model_path,
        tuple(args.output_resolution),
        calibration_data
    )
    hybrid_system.run()
