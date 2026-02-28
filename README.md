# AAV Camera System

Real-time video streaming and stop sign detection for an Autonomous Vehicle.  
Runs on **NVIDIA Jetson Orin AGX** + **Raspberry Pi 5**.

```
Camera (CSI) → MediaMTX (RTSP :8554) → NvDecDecoder → ROS2
                                                          ↓
                                           YoloDetector (TensorRT FP16)
                                                          ↓
                                  /aav/stop_sign_detected   (Bool)
                                  /aav/stop_sign_confidence (Float32)
                                  /aav/stop_sign_detections (Detection2DArray)
```

---

## Requirements

**Hardware**
- NVIDIA Jetson Orin AGX (JetPack 6.x)
- Raspberry Pi 5 (MTX server + dashboard)
- 2× CSI cameras (cam1, cam2)

**Software**
- ROS2 Humble, CUDA 12.x, TensorRT 10.x
- OpenCV 4.x with GStreamer, FFmpeg with `h264_nvv4l2dec`

```bash
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge \
    ros-humble-image-transport ros-humble-compressed-image-transport
```

---

## Repository Structure

```
Camera/
├── camera_decode/
│   ├── include/camera_decode/
│   │   ├── nvdec_decoder.hpp       # RTSP NVDEC decoder interface
│   │   └── yolo_detector.hpp       # TensorRT YOLOv8 detector interface
│   ├── src/
│   │   ├── nvdec_decoder.cpp       # FFmpeg + NVDEC hardware decode
│   │   ├── yolo_detector.cpp       # TensorRT FP16 inference
│   │   ├── camera_decoder_node.cpp # ROS2 node: decode + publish frames
│   │   └── stop_sign_node.cpp      # ROS2 node: infer + publish detections
│   ├── launch/
│   │   └── multi_camera.launch.py
│   └── config/
│       └── cameras.yaml
├── engine/
│   └── tensorRT_engine.py          # ONNX → TensorRT conversion
└── model/
    ├── best.pt                     # YOLOv8 weights (PyTorch)
    ├── best.onnx                   # Exported ONNX model
    └── best.engine                 # TensorRT FP16 engine (Jetson only)
```

---

## Model Export (One-Time)

> The TensorRT engine must be built **on the Jetson** — it is not portable across devices.

**1. Export PyTorch → ONNX**
```bash
python3 engine/tensorRT_engine.py
```

**2. Convert ONNX → TensorRT engine**
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=model/best.onnx \
    --saveEngine=model/best.engine \
    --fp16 \
    --memPoolSize=workspace:4096M
```
Takes 5–15 min. Success = `&&&& PASSED`

---

## Running

**1. MTX Server (Raspberry Pi)** — runs automatically on boot via systemd, or manually:
```bash
/usr/local/bin/mediamtx /usr/local/etc/mediamtx.yml
```

**2. Build workspace (Jetson)**
```bash
source /opt/ros/humble/setup.bash
cd ~/Desktop/aav/Camera
colcon build && source install/setup.bash
```

**3. Launch**
```bash
# All nodes (recommended)
ros2 launch camera_decode multi_camera.launch.py

# With video display (debug)
ros2 launch camera_decode multi_camera.launch.py show_window:=true
```

---

## ROS2 Topics

| Topic | Type | Description |
|---|---|---|
| `/aav/cam1/image_raw` | `sensor_msgs/Image` | Decoded frames from cam1 |
| `/aav/cam2/image_raw` | `sensor_msgs/Image` | Decoded frames from cam2 |
| `/aav/stop_sign_detected` | `std_msgs/Bool` | True when stop sign detected |
| `/aav/stop_sign_confidence` | `std_msgs/Float32` | Detection confidence score |
| `/aav/stop_sign_detections` | `vision_msgs/Detection2DArray` | Bounding boxes + metadata |

---

## Dashboard UI

Web-based monitor for a separate device (e.g. Raspberry Pi) on the same network.

**Setup on Jetson:**
```bash
sudo apt install ros-humble-rosbridge-suite
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Open `aav_dashboard.html` in a browser, enter the Jetson IP, and connect.  
Shows live camera feed (cam1/cam2 toggle), bounding boxes, detection indicator, and event log.

---

## Debugging

```bash
ros2 topic list                          # List active topics
ros2 topic hz /aav/cam1/image_raw        # Check frame rate
ros2 topic echo /aav/stop_sign_detected  # Monitor detections
ros2 topic bw /aav/cam1/image_raw        # Check bandwidth
sudo iftop -i eno1 -f "port 8554"        # RTSP network usage
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| OpenCV build warnings | Safe to ignore — ROS2 (4.5) and system (4.8) coexist fine |
| Camera not starting | Check `cameras.yaml` for correct Raspberry Pi IP |
| Engine fails to load | Rebuild `best.engine` on this Jetson — not cross-device portable |
| RTSP stream not found | Confirm MediaMTX is running on Pi, port 8554 |
| Dashboard won't connect | Confirm rosbridge is running on Jetson, port 9090 reachable |
