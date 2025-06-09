import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime
import argparse

class CrowdDetector:
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4, proximity_threshold=100, crowd_size=3, frame_persistence=10):
        """
        Initialize the crowd detector with configurable parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence for object detection
            nms_threshold (float): Non-maximum suppression threshold
            proximity_threshold (int): Maximum distance between persons to be considered in same crowd (in pixels)
            crowd_size (int): Minimum number of persons to be considered a crowd
            frame_persistence (int): Number of consecutive frames a crowd must persist to be logged
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.proximity_threshold = proximity_threshold
        self.crowd_size = crowd_size
        self.frame_persistence = frame_persistence
        
        # Load YOLO model
        self._load_model()
        
        # Initialize tracking variables
        self.crowd_tracking = {}  # Format: {crowd_id: {'frames': count, 'persons': count, 'last_frame': frame_number}}
        self.next_crowd_id = 0
        self.current_frame = 0
        self.logged_crowds = set()  # To avoid duplicate logging
        
        # Results tracking
        self.results = []  # List to store results before writing to CSV
    
    def _load_model(self):
        """Load the YOLO model for person detection."""
        # Paths to YOLO files (will download if not present)
        self.weights_path = self._get_model_file("yolov4.weights", "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
        self.config_path = self._get_model_file("yolov4.cfg", "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
        self.classes_path = self._get_model_file("coco.names", "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        
        # Load class names
        with open(self.classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load YOLO network
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
        
        # Set preferred backend (CUDA if available)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        print(f"Model loaded successfully. Using {'CUDA' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'CPU'} backend.")
    
    def _get_model_file(self, filename, url):
        """
        Get model file, download if not present.
        
        Args:
            filename (str): Name of the file
            url (str): URL to download the file from
            
        Returns:
            str: Path to the file
        """
        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
        
        file_path = os.path.join("models", filename)
        
        # Download file if it doesn't exist
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            import urllib.request
            urllib.request.urlretrieve(url, file_path)
            print(f"{filename} downloaded successfully.")
        
        return file_path
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame using YOLO.
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            list: List of person bounding boxes in format [x, y, w, h]
        """
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get detections
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for person class (class ID 0 in COCO dataset)
                if class_id == 0 and confidence > self.confidence_threshold:
                    # YOLO returns center, width and height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Extract person boxes
        person_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                person_boxes.append(boxes[i])
        
        return person_boxes
    
    def detect_crowds(self, person_boxes):
        """
        Detect crowds based on person proximity.
        
        Args:
            person_boxes (list): List of person bounding boxes
            
        Returns:
            list: List of crowds, each containing person indices
        """
        # If not enough persons detected, no crowd possible
        if len(person_boxes) < self.crowd_size:
            return []
        
        # Calculate centers of each person
        centers = []
        for box in person_boxes:
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        
        # Build proximity graph: connect persons within threshold distance
        proximity_graph = [[] for _ in range(len(centers))]
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                if dist <= self.proximity_threshold:
                    proximity_graph[i].append(j)
                    proximity_graph[j].append(i)
        
        # Find connected components (crowds)
        visited = [False] * len(centers)
        crowds = []
        
        for i in range(len(centers)):
            if not visited[i]:
                crowd = []
                self._dfs(i, proximity_graph, visited, crowd)
                if len(crowd) >= self.crowd_size:
                    crowds.append(crowd)
        
        return crowds
    
    def _dfs(self, node, graph, visited, component):
        """
        Depth-first search to find connected components.
        
        Args:
            node (int): Current node
            graph (list): Adjacency list representation of graph
            visited (list): List tracking visited nodes
            component (list): Current component being built
        """
        visited[node] = True
        component.append(node)
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                self._dfs(neighbor, graph, visited, component)
    
    def update_crowd_tracking(self, crowds, person_boxes):
        """
        Update crowd tracking to identify persistent crowds.
        
        Args:
            crowds (list): List of crowds detected in current frame
            person_boxes (list): List of person bounding boxes
        """
        # Create dictionary of current crowds
        current_crowds = {}
        for i, crowd in enumerate(crowds):
            # Calculate centroid of crowd
            centers = []
            for person_idx in crowd:
                x, y, w, h = person_boxes[person_idx]
                center_x = x + w // 2
                center_y = y + h // 2
                centers.append((center_x, center_y))
            
            crowd_center_x = sum(c[0] for c in centers) / len(centers)
            crowd_center_y = sum(c[1] for c in centers) / len(centers)
            current_crowds[i] = {
                'center': (crowd_center_x, crowd_center_y),
                'size': len(crowd),
                'id': None  # Will be assigned below
            }
        
        # Match current crowds with tracked crowds
        matched_tracked = set()
        matched_current = set()
        
        # Only try to match if we have tracked crowds
        if self.crowd_tracking:
            # Calculate distances between tracked and current crowds
            for tracked_id, tracked_info in self.crowd_tracking.items():
                if 'center' not in tracked_info:  # Skip if no center info
                    continue
                    
                for current_idx, current_info in current_crowds.items():
                    if current_idx in matched_current:
                        continue
                        
                    dist = np.sqrt(
                        (tracked_info['center'][0] - current_info['center'][0])**2 + 
                        (tracked_info['center'][1] - current_info['center'][1])**2
                    )
                    
                    # If close enough, consider it the same crowd
                    if dist < self.proximity_threshold * 2:  # Larger threshold for crowd matching
                        matched_tracked.add(tracked_id)
                        matched_current.add(current_idx)
                        current_info['id'] = tracked_id
                        
                        # Update tracked info
                        self.crowd_tracking[tracked_id]['frames'] += 1
                        self.crowd_tracking[tracked_id]['persons'] = current_info['size']
                        self.crowd_tracking[tracked_id]['last_frame'] = self.current_frame
                        self.crowd_tracking[tracked_id]['center'] = current_info['center']
                        break
        
        # Create new entries for unmatched current crowds
        for current_idx, current_info in current_crowds.items():
            if current_idx not in matched_current:
                crowd_id = self.next_crowd_id
                self.next_crowd_id += 1
                
                self.crowd_tracking[crowd_id] = {
                    'frames': 1,
                    'persons': current_info['size'],
                    'last_frame': self.current_frame,
                    'center': current_info['center']
                }
                current_info['id'] = crowd_id
        
        # Remove old tracked crowds that haven't been seen recently
        to_remove = []
        for tracked_id in self.crowd_tracking:
            if tracked_id not in matched_tracked and self.current_frame - self.crowd_tracking[tracked_id]['last_frame'] > 30:
                to_remove.append(tracked_id)
        
        for tracked_id in to_remove:
            del self.crowd_tracking[tracked_id]
        
        # Check for crowds that have persisted for the required number of frames
        for tracked_id, info in self.crowd_tracking.items():
            if info['frames'] >= self.frame_persistence and tracked_id not in self.logged_crowds:
                # Log this crowd
                self.results.append({
                    'frame_number': self.current_frame,
                    'persons_in_crowd': info['persons']
                })
                self.logged_crowds.add(tracked_id)
                print(f"Crowd detected at frame {self.current_frame} with {info['persons']} persons")
    
    def process_frame(self, frame):
        """
        Process a single frame for crowd detection.
        
        Args:
            frame (numpy.ndarray): Video frame
            
        Returns:
            tuple: (Processed frame with annotations, person boxes, crowds)
        """
        self.current_frame += 1
        
        # Detect persons
        person_boxes = self.detect_persons(frame)
        
        # Detect crowds
        crowds = self.detect_crowds(person_boxes)
        
        # Update tracking
        self.update_crowd_tracking(crowds, person_boxes)
        
        # Visualize results on frame
        return self.visualize_results(frame, person_boxes, crowds)
    
    def visualize_results(self, frame, person_boxes, crowds):
        """
        Visualize detection results on frame.
        
        Args:
            frame (numpy.ndarray): Video frame
            person_boxes (list): List of person bounding boxes
            crowds (list): List of detected crowds
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw all person boxes
        for i, box in enumerate(person_boxes):
            x, y, w, h = box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw crowd annotations with different colors for each crowd
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, crowd in enumerate(crowds):
            color = colors[i % len(colors)]
            # Draw connections between crowd members
            for j in range(len(crowd)):
                for k in range(j+1, len(crowd)):
                    p1_idx = crowd[j]
                    p2_idx = crowd[k]
                    
                    if p1_idx < len(person_boxes) and p2_idx < len(person_boxes):
                        p1_box = person_boxes[p1_idx]
                        p2_box = person_boxes[p2_idx]
                        
                        p1_center = (p1_box[0] + p1_box[2] // 2, p1_box[1] + p1_box[3] // 2)
                        p2_center = (p2_box[0] + p2_box[2] // 2, p2_box[1] + p2_box[3] // 2)
                        
                        cv2.line(vis_frame, p1_center, p2_center, color, 1)
        
        # Add frame number and other info
        cv2.putText(vis_frame, f"Frame: {self.current_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame, f"Persons: {len(person_boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame, f"Crowds: {len(crowds)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add persistent crowd info
        persistent_crowds = 0
        for info in self.crowd_tracking.values():
            if info['frames'] >= self.frame_persistence:
                persistent_crowds += 1
        
        cv2.putText(vis_frame, f"Persistent Crowds: {persistent_crowds}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame, person_boxes, crowds
    
    def save_results(self, output_path="crowd_detection_results.csv"):
        """
        Save detection results to CSV.
        
        Args:
            output_path (str): Path to output CSV file
        """
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_number', 'persons_in_crowd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        
        print(f"Results saved to {output_path}")
    
    def process_video(self, video_path, output_video_path=None, display=True):
        """
        Process a video file for crowd detection.
        
        Args:
            video_path (str): Path to input video
            output_video_path (str, optional): Path to save output video
            display (bool): Whether to display the video during processing
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Reset tracking variables
        self.crowd_tracking = {}
        self.next_crowd_id = 0
        self.current_frame = 0
        self.logged_crowds = set()
        self.results = []
        
        # Process frames
        start_time = time.time()
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            vis_frame, _, _ = self.process_frame(frame)
            
            # Write frame to output video
            if writer:
                writer.write(vis_frame)
            
            # Display frame
            if display:
                cv2.imshow('Crowd Detection', vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
            
            # Update progress
            processed_frames += 1
            if processed_frames % 100 == 0:
                elapsed_time = time.time() - start_time
                frames_per_second = processed_frames / elapsed_time
                remaining_frames = total_frames - processed_frames
                estimated_time = remaining_frames / frames_per_second if frames_per_second > 0 else 0
                
                print(f"Progress: {processed_frames}/{total_frames} frames "
                      f"({processed_frames/total_frames*100:.1f}%), "
                      f"FPS: {frames_per_second:.2f}, "
                      f"ETA: {int(estimated_time//60)}m {int(estimated_time%60)}s")
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Save results
        output_path = os.path.splitext(video_path)[0] + "_crowd_results.csv"
        self.save_results(output_path)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {processed_frames/elapsed_time:.2f}")
        print(f"Detected {len(self.results)} crowd events")
        print(f"Results saved to {output_path}")
        if output_video_path:
            print(f"Output video saved to {output_video_path}")


def main():
    """Main function to run the crowd detection system."""
    parser = argparse.ArgumentParser(description='Crowd Detection System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file (optional)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for person detection')
    parser.add_argument('--proximity', type=int, default=100, help='Proximity threshold in pixels')
    parser.add_argument('--crowd_size', type=int, default=3, help='Minimum number of persons to be considered a crowd')
    parser.add_argument('--persistence', type=int, default=10, help='Number of frames a crowd must persist')
    parser.add_argument('--display', action='store_true', help='Display video during processing')
    
    args = parser.parse_args()
    
    # Initialize crowd detector
    detector = CrowdDetector(
        confidence_threshold=args.confidence,
        proximity_threshold=args.proximity,
        crowd_size=args.crowd_size,
        frame_persistence=args.persistence
    )
    
    # Process video
    detector.process_video(args.video, args.output, args.display)


if __name__ == "__main__":
    main()
