# Crowd_Detection
Real-time crowd detection system using YOLOv4 and OpenCV to identify and track groups of people based on proximity and persistence in video footage.
This Python application detects crowds in videos by identifying groups of people standing close to each other for a specific duration.

### Video Explanation
Watch a detailed explanation of the approach, implementation, and results:

[🔗 Watch the Demo Video](https://drive.google.com/file/d/13ygcpgt0caLxkx-9C8vpNlyOtAZQ_q0c/view)

## Features

-  Uses **YOLOv4** for person detection  
-  Identifies **crowds based on proximity and persistence**  
-  **Tracks** crowds across video frames  
- Saves **crowd detection results to CSV**  
- **Visualizes** detection results in output video  

---

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy

---

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
# Install dependencies
```bash 
pip install opencv-python numpy
```
```
python crowd_detection.py --video path/to/your/video.mp4 --output output.mp4 --display
Command-line Arguments
Argument	Description	Default
--video	Path to input video file (required)	—
--output	Path to save output video with visualizations (optional)	None
--confidence	Confidence threshold for YOLO detection	0.5
--proximity	Pixel distance to define "close" individuals	100
--crowd_size	Minimum number of people to define a crowd	3
--persistence	Number of consecutive frames a group must persist to be logged as a crowd	10
--display	Display video during processing (flag)	Off
```
Output
The program generates:

 A CSV file with detected crowd events (frame number and person count)

An optional output video with bounding boxes and crowd highlights

How It Works
Person Detection: Uses YOLOv4 to detect persons in each frame.
Crowd Identification: Groups persons standing close to each other.
Persistence Tracking: Monitors groups across consecutive frames.
Event Logging: Logs events when crowd conditions are met.

Example
```bash
# Basic usage
python crowd_detection.py --video sample.mp4

# Full options
python crowd_detection.py \
  --video sample.mp4 \
  --output crowd_detection.mp4 \
  --confidence 0.6 \
  --proximity 80 \
  --crowd_size 3 \
  --persistence 15 \
  --display
```
The YOLOv4 model files (weights, config, and class names) are downloaded automatically on the first run.

CUDA acceleration is used if available to speed up detection.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

