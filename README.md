# AAV Camera System

Real-time video streaming and stop sign detection for an Autonomous Vehicle.
Runs on **NVIDIA Jetson Orin AGX** with a built-in web dashboard accessible from any device on the network.

```
Camera (CSI) → MediaMTX (RTSP :8554) → NvDecDecoder → stop_sign_node (TensorRT FP16)
                                                               ↓
                                                   web_dashboard_node (C++)
                                                               ↓
                                              http://<JETSON_IP>:8080
                                         (any browser, any device on network)
```

---

## How the Dashboard Works

The `web_dashboard_node` subscribes to the existing ROS2 topics published by `stop_sign_node` and `camera_decoder_node`. It draws bounding boxes directly onto incoming frames, encodes them as JPEG, and serves everything over HTTP — no rosbridge, no Python, no installs on the viewer device.

| Endpoint | Description |
|---|---|
| `http://<JETSON_IP>:8080/` | Full dashboard HTML |
| `http://<JETSON_IP>:8080/cam1` | Live MJPEG stream with bounding boxes (cam1) |
| `http://<JETSON_IP>:8080/cam2` | Live MJPEG stream with bounding boxes (cam2) |
| `http://<JETSON_IP>:8080/detections` | JSON detection state (polled by dashboard) |

---

## Requirements

**Hardware**
- NVIDIA Jetson Orin AGX (JetPack 6.x)
- Raspberry Pi 5 (MTX server)
- 2× CSI cameras (cam1, cam2)

**Software**
- ROS2 Humble, CUDA 12.x, TensorRT 10.x
- OpenCV 4.x with GStreamer
- FFmpeg with `h264_nvv4l2dec`

```bash
sudo apt install \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport
```

---

## Repository Structure

```
Camera/
├── camera_decode/
│   ├── include/camera_decode/
│   │   ├── nvdec_decoder.hpp
│   │   └── yolo_detector.hpp
│   ├── src/
│   │   ├── nvdec_decoder.cpp          # FFmpeg + NVDEC hardware decode
│   │   ├── yolo_detector.cpp          # TensorRT FP16 inference
│   │   ├── camera_decoder_node.cpp    # Decodes RTSP → publishes /aav/cam*/image_raw
│   │   ├── stop_sign_node.cpp         # Runs inference → publishes /aav/stop_sign_*
│   │   └── web_dashboard_node.cpp     # Subscribes to topics → serves HTTP dashboard
│   ├── launch/
│   │   └── multi_camera.launch.py
│   └── config/
│       └── cameras.yaml
├── engine/
│   └── tensorRT_engine.py             # ONNX → TensorRT conversion
└── model/
    ├── best.pt                        # YOLOv8 weights (PyTorch)
    ├── best.onnx                      # Exported ONNX model
    └── best.engine                    # TensorRT FP16 engine (Jetson only)
```

> **Note:** `setup.py`, `setup.cfg`, `dashboard_node.py`, `__init__.py`, and any standalone `aav_dashboard.html` are no longer needed. The dashboard HTML is embedded inside `web_dashboard_node.cpp`.

---

## Model Export (One-Time Setup)

> The TensorRT engine must be built **on the Jetson** — it cannot be cross-compiled.

**1. Export PyTorch → ONNX**
```bash
python3 engine/tensorRT_engine.py
```

**2. Convert ONNX → TensorRT engine on Jetson**
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=model/best.onnx \
    --saveEngine=model/best.engine \
    --fp16 \
    --memPoolSize=workspace:4096M
```
Takes 5–15 min. Success = `&&&& PASSED`

---

## CMakeLists.txt Setup

Add this block to `camera_decode/CMakeLists.txt` after the existing `add_executable` entries:

```cmake
# web_dashboard_node
add_executable(web_dashboard_node src/web_dashboard_node.cpp)
target_link_libraries(web_dashboard_node ${PROJECT_NAME}_lib pthread)
ament_target_dependencies(web_dashboard_node
  rclcpp std_msgs sensor_msgs vision_msgs cv_bridge OpenCV
)
install(TARGETS web_dashboard_node DESTINATION lib/${PROJECT_NAME})
```

---

## Running

**1. MTX Server (Raspberry Pi)** — starts automatically on boot, or manually:
```bash
/usr/local/bin/mediamtx /usr/local/etc/mediamtx.yml
```

**2. Build workspace (Jetson)**
```bash
source /opt/ros/humble/setup.bash
cd ~/Desktop/aav/Camera
colcon build --packages-select camera_decode
source install/setup.bash
```

**3. Terminal 1 — Camera + Detection pipeline**
```bash
ros2 launch camera_decode multi_camera.launch.py
```

**4. Terminal 2 — Web Dashboard**
```bash
ros2 run camera_decode web_dashboard_node
```

**5. Open the dashboard**

On any laptop, phone, or tablet on the same network:
```
http://<JETSON_IP>:8080
```

---

## ROS2 Topics

| Topic | Type | Publisher | Description |
|---|---|---|---|
| `/aav/cam1/image_raw` | `sensor_msgs/Image` | `camera_decoder_node` | Raw frames from cam1 |
| `/aav/cam2/image_raw` | `sensor_msgs/Image` | `camera_decoder_node` | Raw frames from cam2 |
| `/aav/stop_sign_detected` | `std_msgs/Bool` | `stop_sign_node` | True when stop sign detected |
| `/aav/stop_sign_confidence` | `std_msgs/Float32` | `stop_sign_node` | Detection confidence score |
| `/aav/stop_sign_detections` | `vision_msgs/Detection2DArray` | `stop_sign_node` | Bounding boxes + metadata |

`web_dashboard_node` **subscribes** to all of the above — it publishes nothing.

---

## Dashboard Features

- **Stop sign indicator** — animated octagon turns red when a stop sign is detected
- **CAM1 / CAM2 toggle** — switch between live camera feeds
- **Bounding boxes** — drawn server-side on the Jetson before streaming, no browser canvas needed
- **Detection event log** — timestamped history of every detection and clear event
- **Total detections counter** and last-seen timestamp
- **Works on any device** — pure HTTP, no ROS2 or extra software needed on the viewer

---

## Performance

| Component | CPU | GPU |
|---|---|---|
| NVDEC decode (2 cams) | ~5% | ~5–10% (dedicated HW block) |
| TensorRT inference (FP16) | ~2% | ~25–40% |
| JPEG encode + HTTP serve | ~5–10% | 0% |
| **Total** | **~15–20%** | **~30–50%** |

Tune in `web_dashboard_node.cpp`:
```cpp
static constexpr int  JPEG_QUALITY = 60;   // 30–80, lower = less CPU
static constexpr int  STREAM_FPS   = 15;   // reduce to save CPU
```

---

## Debugging

```bash
ros2 topic list                           # Verify all topics are active
ros2 topic hz /aav/cam1/image_raw         # Check frame rate from decoder
ros2 topic echo /aav/stop_sign_detected   # Monitor detections in terminal
ros2 topic bw /aav/cam1/image_raw         # Check bandwidth usage
sudo iftop -i eno1 -f "port 8554"         # Monitor RTSP stream bandwidth
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Dashboard won't load | Confirm `web_dashboard_node` is running and Jetson port 8080 is reachable |
| Camera feed not showing | Check `ros2 topic hz /aav/cam1/image_raw` — if 0, decoder node isn't running |
| Bounding boxes not appearing | Check `ros2 topic echo /aav/stop_sign_detections` — if empty, stop_sign_node issue |
| High latency on stream | Lower `JPEG_QUALITY` to 40 and `STREAM_FPS` to 10 in `web_dashboard_node.cpp` |
| Camera not starting | Check `cameras.yaml` for correct Raspberry Pi IP address |
| Engine fails to load | Rebuild `best.engine` on this Jetson — not portable across devices |
| OpenCV build warnings | Safe to ignore — ROS2 (4.5) and system (4.8) OpenCV coexist fine |
