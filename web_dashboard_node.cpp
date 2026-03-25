/**
 * web_dashboard_node.cpp
 *
 * Dashboard node:
 * - subscribes to raw camera images
 * - subscribes to per-camera stop sign + pedestrian detections
 * - overlays detections only on the matching camera
 * - shows CAM1 and CAM2 side by side
 * - serves MJPEG dashboard over HTTP
 *
 * Camera topics:
 *   /camera/cam1/image_raw
 *   /camera/cam2/image_raw
 *
 * Detection topics:
 *   /aav/cam1/stop_sign_detections
 *   /aav/cam2/stop_sign_detections
 *   /aav/cam1/pedestrian_detections
 *   /aav/cam2/pedestrian_detections
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cstring>
#include <algorithm>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace std::chrono_literals;

// ─────────────────────────────────────────────────────────────────────────────
// Tuning
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int HTTP_PORT    = 8080;
static constexpr int JPEG_QUALITY = 60;
static constexpr int STREAM_FPS   = 15;

// ─────────────────────────────────────────────────────────────────────────────
// Shared frame buffers
// ─────────────────────────────────────────────────────────────────────────────
struct FrameBuffer {
    std::mutex mtx;
    std::vector<uchar> jpeg;
    bool ready = false;
};

static FrameBuffer g_cam1;
static FrameBuffer g_cam2;

// ─────────────────────────────────────────────────────────────────────────────
// Overlay model
// ─────────────────────────────────────────────────────────────────────────────
struct BBox {
    float x{};
    float y{};
    float w{};
    float h{};
    float conf{};
    std::string class_label;
};

struct OverlayBuffer {
    std::mutex mtx;
    std::vector<BBox> boxes;
};

static OverlayBuffer g_cam1_overlay;
static OverlayBuffer g_cam2_overlay;

// Keep per-camera/per-class storage so each camera only renders its own detections
static std::mutex g_cam1_det_mtx;
static std::vector<BBox> g_cam1_stop_boxes;
static std::vector<BBox> g_cam1_ped_boxes;

static std::mutex g_cam2_det_mtx;
static std::vector<BBox> g_cam2_stop_boxes;
static std::vector<BBox> g_cam2_ped_boxes;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static void encode_and_store(FrameBuffer& buf, cv::Mat& frame)
{
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", frame, jpeg, {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});

    std::lock_guard<std::mutex> lk(buf.mtx);
    buf.jpeg = std::move(jpeg);
    buf.ready = true;
}

static std::string pretty_label(const std::string& cls)
{
    if (cls == "stop_sign")  return "STOP";
    if (cls == "pedestrian") return "PERSON";
    return cls;
}

static cv::Scalar class_color(const std::string& cls)
{
    if (cls == "stop_sign")  return cv::Scalar(50, 52, 232);   // red-ish
    if (cls == "pedestrian") return cv::Scalar(0, 165, 255);   // orange
    return cv::Scalar(0, 255, 255);
}

static std::vector<BBox> convert_detections(
    const vision_msgs::msg::Detection2DArray::SharedPtr msg,
    const std::string& forced_class_label)
{
    std::vector<BBox> boxes;
    boxes.reserve(msg->detections.size());

    for (const auto& d : msg->detections) {
        BBox b;
        b.x = d.bbox.center.position.x - d.bbox.size_x / 2.0f;
        b.y = d.bbox.center.position.y - d.bbox.size_y / 2.0f;
        b.w = d.bbox.size_x;
        b.h = d.bbox.size_y;
        b.conf = d.results.empty() ? 0.0f : d.results[0].hypothesis.score;
        b.class_label = forced_class_label;
        boxes.push_back(b);
    }

    return boxes;
}

static void rebuild_overlay(OverlayBuffer& overlay,
                            const std::vector<BBox>& stop_boxes,
                            const std::vector<BBox>& ped_boxes)
{
    std::vector<BBox> merged;
    merged.reserve(stop_boxes.size() + ped_boxes.size());
    merged.insert(merged.end(), stop_boxes.begin(), stop_boxes.end());
    merged.insert(merged.end(), ped_boxes.begin(), ped_boxes.end());

    std::lock_guard<std::mutex> lk(overlay.mtx);
    overlay.boxes = std::move(merged);
}

static void rebuild_cam1_overlay()
{
    std::lock_guard<std::mutex> lk(g_cam1_det_mtx);
    rebuild_overlay(g_cam1_overlay, g_cam1_stop_boxes, g_cam1_ped_boxes);
}

static void rebuild_cam2_overlay()
{
    std::lock_guard<std::mutex> lk(g_cam2_det_mtx);
    rebuild_overlay(g_cam2_overlay, g_cam2_stop_boxes, g_cam2_ped_boxes);
}

static void draw_boxes(cv::Mat& frame, const std::vector<BBox>& boxes)
{
    for (const auto& b : boxes) {
        cv::Rect r(
            static_cast<int>(b.x),
            static_cast<int>(b.y),
            static_cast<int>(b.w),
            static_cast<int>(b.h)
        );

        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.empty()) continue;

        const cv::Scalar box_color = class_color(b.class_label);
        const cv::Scalar label_bg  = box_color;
        const cv::Scalar label_fg(255, 255, 255);

        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, r, box_color, cv::FILLED);
        cv::addWeighted(overlay, 0.10, frame, 0.90, 0, frame);
        cv::rectangle(frame, r, box_color, 2);

        int cl = std::min(20, std::min(r.width, r.height) / 3);
        cv::line(frame, r.tl(), {r.x + cl, r.y}, box_color, 3);
        cv::line(frame, r.tl(), {r.x, r.y + cl}, box_color, 3);
        cv::line(frame, {r.x + r.width, r.y}, {r.x + r.width - cl, r.y}, box_color, 3);
        cv::line(frame, {r.x + r.width, r.y}, {r.x + r.width, r.y + cl}, box_color, 3);
        cv::line(frame, {r.x, r.y + r.height}, {r.x + cl, r.y + r.height}, box_color, 3);
        cv::line(frame, {r.x, r.y + r.height}, {r.x, r.y + r.height - cl}, box_color, 3);
        cv::line(frame, r.br(), {r.x + r.width - cl, r.y + r.height}, box_color, 3);
        cv::line(frame, r.br(), {r.x + r.width, r.y + r.height - cl}, box_color, 3);

        std::string label =
            pretty_label(b.class_label) + "  " +
            std::to_string(static_cast<int>(b.conf * 100.0f)) + "%";

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        cv::Point lp(r.x, std::max(r.y - 6, ts.height + 4));

        cv::rectangle(frame,
                      lp + cv::Point(0, -ts.height - 4),
                      lp + cv::Point(ts.width + 8, baseline),
                      label_bg, cv::FILLED);

        cv::putText(frame, label, lp + cv::Point(4, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, label_fg, 1, cv::LINE_AA);
    }

    std::string hud = boxes.empty() ? "CLEAR" : "OBJECT DETECTED";
    cv::Scalar hud_c = boxes.empty() ? cv::Scalar(80, 200, 80) : cv::Scalar(50, 52, 232);
    cv::putText(frame, hud, {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, hud_c, 2, cv::LINE_AA);
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedded HTML
// ─────────────────────────────────────────────────────────────────────────────
static const char* DASHBOARD_HTML = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>AAV · Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&display=swap" rel="stylesheet"/>
<style>
  :root{
    --bg:#0a0c0f;--panel:#111418;--border:#1e2430;
    --text:#c8d0dc;--muted:#4a5568;--safe:#1adb6e;
    --mono:'Share Tech Mono',monospace;--head:'Bebas Neue',sans-serif
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{
    background:var(--bg);color:var(--text);font-family:var(--mono);
    height:100vh;display:flex;flex-direction:column;overflow:hidden
  }
  header{
    display:flex;align-items:center;justify-content:space-between;
    padding:14px 24px;border-bottom:1px solid var(--border);background:var(--panel)
  }
  .logo{font-family:var(--head);font-size:1.8rem;letter-spacing:.15em;color:#fff}
  .logo span{color:#e8341a}
  .hm{font-size:.68rem;color:var(--muted);text-align:right;line-height:1.9}

  .sbar{
    display:flex;align-items:center;gap:8px;padding:5px 24px;background:#0d1017;
    border-bottom:1px solid var(--border);font-size:.66rem;color:var(--muted)
  }
  .dot{width:7px;height:7px;border-radius:50%}
  .dot.live{background:var(--safe);box-shadow:0 0 6px var(--safe)}
  .dot.waiting{background:#f0b429;box-shadow:0 0 6px #f0b429}
  .dot.err{background:#e8341a;box-shadow:0 0 6px #e8341a}

  main{flex:1;display:flex;flex-direction:column;overflow:hidden}
  .topbar{
    display:grid;grid-template-columns:1fr 1fr;gap:12px;
    padding:12px;border-bottom:1px solid var(--border);background:var(--panel)
  }
  .status{
    border:1px solid var(--border);border-radius:4px;padding:10px 12px;background:#0d1017
  }
  .status .title{
    font-size:.65rem;letter-spacing:.18em;color:var(--muted);margin-bottom:6px
  }
  .status .value{
    font-family:var(--head);font-size:1.3rem;letter-spacing:.08em
  }

  .cams{
    flex:1;display:grid;grid-template-columns:1fr 1fr;gap:12px;
    padding:12px;overflow:hidden
  }
  .cam{
    position:relative;background:#06080a;border:1px solid var(--border);
    border-radius:4px;overflow:hidden;display:flex;align-items:center;justify-content:center
  }
  .cam canvas{width:100%;height:100%;object-fit:contain;display:block}
  .placeholder{
    position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:10px;color:var(--muted)
  }
  .placeholder.hidden{display:none}
  .cl{
    position:absolute;top:10px;left:10px;background:rgba(10,12,15,.78);
    border:1px solid var(--border);border-radius:3px;padding:3px 10px;
    font-size:.6rem;letter-spacing:.18em;color:var(--muted)
  }

  footer{
    padding:8px 24px;border-top:1px solid var(--border);font-size:.6rem;color:var(--muted);
    display:flex;justify-content:space-between;letter-spacing:.1em
  }
</style>
</head>
<body>
<header>
  <div class="logo">AAV<span>.</span>VISION</div>
  <div class="hm">
    <div id="clock">--:--:--</div>
    <div>LIVE · MJPEG · JETSON</div>
  </div>
</header>

<div class="sbar">
  <div class="dot waiting" id="sd"></div>
  <span id="sl">WAITING FOR FIRST FRAME…</span>
</div>

<main>
  <div class="topbar">
    <div class="status">
      <div class="title">CAM 1 STATUS</div>
      <div class="value" id="cam1_status">LIVE</div>
    </div>
    <div class="status">
      <div class="title">CAM 2 STATUS</div>
      <div class="value" id="cam2_status">LIVE</div>
    </div>
  </div>

  <div class="cams">
    <div class="cam">
      <div class="placeholder" id="ph1">
        <div>CAMERA 1</div>
        <div>AWAITING STREAM</div>
      </div>
      <canvas id="c1"></canvas>
      <div class="cl">CAM1 · MJPEG LIVE</div>
    </div>

    <div class="cam">
      <div class="placeholder" id="ph2">
        <div>CAMERA 2</div>
        <div>AWAITING STREAM</div>
      </div>
      <canvas id="c2"></canvas>
      <div class="cl">CAM2 · MJPEG LIVE</div>
    </div>
  </div>
</main>

<footer>
  <span id="src">STREAM: http://JETSON:8080/cam1 & /cam2</span>
  <span id="fc">FRAMES: 0</span>
</footer>

<script>
  setInterval(() => {
    document.getElementById('clock').textContent =
      new Date().toLocaleTimeString('en-CA', {hour12:false})
  }, 1000)

  let streamLive = false
  let totalFrames = 0

  function setLive() {
    if (streamLive) return
    streamLive = true
    document.getElementById('sd').className = 'dot live'
    document.getElementById('sl').textContent = 'LIVE · MJPEG STREAM ACTIVE'
  }

  function setError(msg) {
    if (!streamLive) {
      document.getElementById('sd').className = 'dot err'
      document.getElementById('sl').textContent = msg
    }
  }

  async function startMjpegStream(url, canvasId, placeholderId) {
    const canvas = document.getElementById(canvasId)
    const ctx = canvas.getContext('2d')
    const ph = document.getElementById(placeholderId)

    while (true) {
      try {
        const resp = await fetch(url)
        if (!resp.ok || !resp.body) throw new Error('bad response')

        const reader = resp.body.getReader()
        let buf = new Uint8Array(0)

        while (true) {
          const {value, done} = await reader.read()
          if (done) break

          const tmp = new Uint8Array(buf.length + value.length)
          tmp.set(buf)
          tmp.set(value, buf.length)
          buf = tmp

          while (true) {
            let soi = -1
            for (let i = 0; i < buf.length - 1; i++) {
              if (buf[i] === 0xFF && buf[i + 1] === 0xD8) { soi = i; break }
            }
            if (soi === -1) break

            let eoi = -1
            for (let i = soi + 2; i < buf.length - 1; i++) {
              if (buf[i] === 0xFF && buf[i + 1] === 0xD9) { eoi = i + 1; break }
            }
            if (eoi === -1) break

            const jpeg = buf.slice(soi, eoi + 1)
            buf = buf.slice(eoi + 1)

            const blob = new Blob([jpeg], {type: 'image/jpeg'})
            const objUrl = URL.createObjectURL(blob)
            const img = new Image()

            img.onload = () => {
              const container = canvas.parentElement
              if (canvas.width !== container.clientWidth || canvas.height !== container.clientHeight) {
                canvas.width = container.clientWidth
                canvas.height = container.clientHeight
              }
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
              URL.revokeObjectURL(objUrl)

              ph.classList.add('hidden')
              setLive()

              totalFrames++
              document.getElementById('fc').textContent = 'FRAMES: ' + totalFrames
            }

            img.onerror = () => URL.revokeObjectURL(objUrl)
            img.src = objUrl
          }
        }
      } catch (e) {
        setError('STREAM ERROR — retrying in 2s…')
      }

      await new Promise(r => setTimeout(r, 2000))
    }
  }

  startMjpegStream('/cam1', 'c1', 'ph1')
  startMjpegStream('/cam2', 'c2', 'ph2')
</script>
</body>
</html>
)HTML";

// ─────────────────────────────────────────────────────────────────────────────
// HTTP helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::string read_request(int fd)
{
    char buf[1024] = {};
    recv(fd, buf, sizeof(buf) - 1, 0);
    return std::string(buf);
}

static std::string parse_path(const std::string& req)
{
    auto p1 = req.find(' ');
    auto p2 = req.find(' ', p1 + 1);
    if (p1 == std::string::npos || p2 == std::string::npos) return "/";
    return req.substr(p1 + 1, p2 - p1 - 1);
}

static void send_str(int fd, const std::string& status,
                     const std::string& ctype, const std::string& body)
{
    std::ostringstream ss;
    ss << "HTTP/1.1 " << status << "\r\n"
       << "Content-Type: " << ctype << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Access-Control-Allow-Origin: *\r\n"
       << "Connection: close\r\n\r\n"
       << body;

    const auto s = ss.str();
    send(fd, s.c_str(), s.size(), 0);
}

static void serve_mjpeg(int fd, FrameBuffer& buf)
{
    const char* hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=--aavframe\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n\r\n";

    send(fd, hdr, strlen(hdr), 0);

    const int delay_ms = 1000 / STREAM_FPS;

    while (true) {
        std::vector<uchar> frame;

        {
            std::lock_guard<std::mutex> lk(buf.mtx);
            if (!buf.ready) {
                std::this_thread::sleep_for(5ms);
                continue;
            }
            frame = buf.jpeg;
        }

        std::ostringstream ph;
        ph << "--aavframe\r\n"
           << "Content-Type: image/jpeg\r\n"
           << "Content-Length: " << frame.size() << "\r\n\r\n";
        const auto phs = ph.str();

        if (send(fd, phs.c_str(), phs.size(), MSG_NOSIGNAL) < 0) break;
        if (send(fd, reinterpret_cast<const char*>(frame.data()), frame.size(), MSG_NOSIGNAL) < 0) break;
        if (send(fd, "\r\n", 2, MSG_NOSIGNAL) < 0) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    close(fd);
}

static void http_server()
{
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(HTTP_PORT);

    bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    listen(srv, 16);

    while (true) {
        int client = accept(srv, nullptr, nullptr);
        if (client < 0) continue;

        std::thread([client]() {
            std::string path = parse_path(read_request(client));

            if (path == "/cam1") {
                serve_mjpeg(client, g_cam1);
            } else if (path == "/cam2") {
                serve_mjpeg(client, g_cam2);
            } else {
                send_str(client, "200 OK", "text/html", DASHBOARD_HTML);
                close(client);
            }
        }).detach();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ROS2 node
// ─────────────────────────────────────────────────────────────────────────────
class WebDashboardNode : public rclcpp::Node
{
public:
    WebDashboardNode() : Node("web_dashboard_node")
    {
        auto qos = rclcpp::SensorDataQoS();

        // Camera streams
        sub_cam1_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/cam1/image_raw", qos,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                process_frame(msg, g_cam1, g_cam1_overlay);
            });

        sub_cam2_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/cam2/image_raw", qos,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                process_frame(msg, g_cam2, g_cam2_overlay);
            });

        // CAM1 stop sign
        sub_cam1_stop_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            "/aav/cam1/stop_sign_detections", 10,
            [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
                auto boxes = convert_detections(msg, "stop_sign");
                {
                    std::lock_guard<std::mutex> lk(g_cam1_det_mtx);
                    g_cam1_stop_boxes = std::move(boxes);
                }
                rebuild_cam1_overlay();
            });

        // CAM1 pedestrian
        sub_cam1_ped_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            "/aav/cam1/pedestrian_detections", 10,
            [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
                auto boxes = convert_detections(msg, "pedestrian");
                {
                    std::lock_guard<std::mutex> lk(g_cam1_det_mtx);
                    g_cam1_ped_boxes = std::move(boxes);
                }
                rebuild_cam1_overlay();
            });

        // CAM2 stop sign
        sub_cam2_stop_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            "/aav/cam2/stop_sign_detections", 10,
            [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
                auto boxes = convert_detections(msg, "stop_sign");
                {
                    std::lock_guard<std::mutex> lk(g_cam2_det_mtx);
                    g_cam2_stop_boxes = std::move(boxes);
                }
                rebuild_cam2_overlay();
            });

        // CAM2 pedestrian
        sub_cam2_ped_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            "/aav/cam2/pedestrian_detections", 10,
            [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
                auto boxes = convert_detections(msg, "pedestrian");
                {
                    std::lock_guard<std::mutex> lk(g_cam2_det_mtx);
                    g_cam2_ped_boxes = std::move(boxes);
                }
                rebuild_cam2_overlay();
            });

        http_thread_ = std::thread(http_server);
        http_thread_.detach();

        RCLCPP_INFO(get_logger(), "Web dashboard running at http://<JETSON_IP>:%d", HTTP_PORT);
        RCLCPP_INFO(get_logger(), "CAM1 image topic: /camera/cam1/image_raw");
        RCLCPP_INFO(get_logger(), "CAM2 image topic: /camera/cam2/image_raw");
        RCLCPP_INFO(get_logger(), "CAM1 detections: /aav/cam1/stop_sign_detections, /aav/cam1/pedestrian_detections");
        RCLCPP_INFO(get_logger(), "CAM2 detections: /aav/cam2/stop_sign_detections, /aav/cam2/pedestrian_detections");
    }

private:
    void process_frame(const sensor_msgs::msg::Image::SharedPtr& msg,
                       FrameBuffer& frame_buf,
                       OverlayBuffer& overlay_buf)
    {
        try {
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();

            std::vector<BBox> boxes_copy;
            {
                std::lock_guard<std::mutex> lk(overlay_buf.mtx);
                boxes_copy = overlay_buf.boxes;
            }

            draw_boxes(frame, boxes_copy);
            encode_and_store(frame_buf, frame);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(get_logger(), "cv_bridge error: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam1_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam2_;

    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_cam1_stop_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_cam1_ped_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_cam2_stop_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_cam2_ped_;

    std::thread http_thread_;
};

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WebDashboardNode>());
    rclcpp::shutdown();
    return 0;
}
