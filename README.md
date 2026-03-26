# ROS 2 Multi-Camera Dashboard with YOLO Detection

This project runs two RTSP camera feeds, performs YOLO inference on a Jetson, publishes detections over ROS 2, and serves a web dashboard that shows both camera streams with bounding boxes.

## Features

- Two camera feeds: `cam1` and `cam2`
- Object detection on the Jetson using TensorRT
- Per-camera ROS 2 detection topics
- Web dashboard with both camera feeds shown side by side
- Bounding boxes for:
  - stop signs
  - pedestrians
  - traffic lights

## System Overview

### Detector node
Each detector instance:
- connects to one RTSP stream
- decodes frames
- runs YOLO inference
- publishes camera-specific ROS 2 detection topics

Example topics for `cam1`:

```bash
/aav/cam1/stop_sign_detected
/aav/cam1/stop_sign_confidence
/aav/cam1/stop_sign_detections
/aav/cam1/pedestrian_detected
/aav/cam1/pedestrian_confidence
/aav/cam1/pedestrian_detections
```

The same pattern is used for `cam2`.

### Dashboard node
The dashboard node:
- subscribes to raw camera image topics
- subscribes to per-camera detection topics
- draws bounding boxes on the matching camera feed
- encodes frames as JPEG
- streams them over HTTP as MJPEG

Browser endpoints:

```bash
http://<JETSON_IP>:8080/
http://<JETSON_IP>:8080/cam1
http://<JETSON_IP>:8080/cam2
```

## Expected ROS 2 Topics

### Camera image topics

```bash
/camera/cam1/image_raw
/camera/cam2/image_raw
```

### Detection topics

```bash
/aav/cam1/stop_sign_detections
/aav/cam2/stop_sign_detections
/aav/cam1/pedestrian_detections
/aav/cam2/pedestrian_detections
```

Optional traffic light topics:

```bash
/aav/cam1/traffic_light_detections
/aav/cam2/traffic_light_detections
```

## Requirements

Before starting, make sure you have:

- Ubuntu
- ROS 2 installed
- OpenCV
- `cv_bridge`
- `vision_msgs`
- Jetson environment set up if using TensorRT and hardware decode
- valid TensorRT engine files
- reachable RTSP streams for both cameras

Typical stack used in this project:

- ROS 2 Humble
- C++
- OpenCV
- TensorRT
- NVIDIA Jetson

## Workspace Layout

Example workspace layout:

```bash
~/aav_ros2_ws/
  src/
    camera_decode/
      src/
      include/
      CMakeLists.txt
      package.xml
```

Adjust paths if your workspace is different.

## Build Instructions

From your ROS 2 workspace:

```bash
cd ~/aav_ros2_ws
colcon build --packages-select camera_decode
source install/setup.bash
```

Every new terminal should source the workspace again:

```bash
cd ~/aav_ros2_ws
source install/setup.bash
```

## Run the Detector Nodes

Run one detector node per camera.

### Camera 1

```bash
ros2 run camera_decode stop_sign_node --ros-args \
  -p camera_name:=cam1 \
  -p rtsp_url:=rtsp://192.168.1.102:8554/cam1 \
  -p stop_engine_path:=/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/best.engine \
  -p tl_engine_path:=/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/traffic_lights.engine \
  -p pedestrian_engine_path:=/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/pedestrian.engine \
  -p conf_thresh:=0.5 \
  -p tl_conf_thresh:=0.5 \
  -p pedestrian_conf_thresh:=0.5
```

### Camera 2

```bash
ros2 run camera_decode stop_sign_node --ros-args \
  -p camera_name:=cam2 \
  -p rtsp_url:=rtsp://192.168.1.102:8554/cam2 \
  -p stop_engine_path:=/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/best.engine \
  -p tl_engine_path:=/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/traffic_lights.engine \
  -p pedestrian_engine_path:=/home/aavjetson/Desktop/aav_ros2_ws/Camera/model/pedestrian.engine \
  -p conf_thresh:=0.5 \
  -p tl_conf_thresh:=0.5 \
  -p pedestrian_conf_thresh:=0.5
```

## Run the Dashboard

In another terminal:

```bash
cd ~/aav_ros2_ws
source install/setup.bash
ros2 run camera_decode web_dashboard_node
```

Open the dashboard in your browser:

```bash
http://<JETSON_IP>:8080
```

If you are testing locally on the same machine:

```bash
http://localhost:8080
```

## Quick Verification

### Check camera image topics

```bash
ros2 topic list | grep camera
```

### Check detection topics

```bash
ros2 topic list | grep aav
```

### Check stop sign detections

```bash
ros2 topic echo /aav/cam1/stop_sign_detections --once
ros2 topic echo /aav/cam2/stop_sign_detections --once
```

### Check pedestrian detections

```bash
ros2 topic echo /aav/cam1/pedestrian_detections --once
ros2 topic echo /aav/cam2/pedestrian_detections --once
```

### Check image rate

```bash
ros2 topic hz /camera/cam1/image_raw
ros2 topic hz /camera/cam2/image_raw
```

### Check MJPEG endpoints directly

```bash
curl -v http://localhost:8080/cam1 --output /dev/null
curl -v http://localhost:8080/cam2 --output /dev/null
```

If bytes are flowing, the MJPEG stream is working.

## Startup Order

Use this order when bringing the system up:

1. Start the RTSP camera sources
2. Start the detector node for `cam1`
3. Start the detector node for `cam2`
4. Start the dashboard node
5. Open the dashboard in the browser

## Common Troubleshooting

### 1. Dashboard says waiting for first frame
Possible causes:
- wrong camera topic name
- no image messages arriving
- dashboard not subscribed to the correct image topic
- `cv_bridge` conversion failed

Checks:

```bash
ros2 topic list | grep image
ros2 topic hz /camera/cam1/image_raw
ros2 topic hz /camera/cam2/image_raw
```

### 2. No boundary boxes appear
Possible causes:
- detector is not publishing detections
- dashboard is subscribed to the wrong detection topics
- confidence threshold is too high
- detections are being published under a different camera name

Checks:

```bash
ros2 topic list | grep detections
ros2 topic echo /aav/cam1/stop_sign_detections --once
ros2 topic echo /aav/cam1/pedestrian_detections --once
```

### 3. Boxes appear on the wrong camera
Possible cause:
- the dashboard is using shared detection state instead of separate per-camera overlay buffers

Fix:
- keep separate detection buffers for `cam1` and `cam2`
- subscribe to camera-specific topics only
- draw only the matching camera’s detections on that feed

### 4. Browser loads but no stream is visible
Check the stream endpoints directly:

```bash
curl -v http://localhost:8080/cam1 --output /dev/null
curl -v http://localhost:8080/cam2 --output /dev/null
```

If the connection opens but no bytes come through, the dashboard probably has no frames in its JPEG buffer yet.

## Notes

- This setup assumes one detector node instance per camera.
- Topic names must match exactly between the detector and dashboard.
- If you change `camera_name`, the detection topic names will also change.
- If you later add more object classes, follow the same per-camera topic structure.

## Example Demo Flow

1. Launch both camera streams
2. Launch both detector nodes
3. Launch the dashboard
4. Open the dashboard in a browser
5. Show a stop sign or pedestrian in front of one camera
6. Confirm that only the matching camera feed shows the correct bounding box
