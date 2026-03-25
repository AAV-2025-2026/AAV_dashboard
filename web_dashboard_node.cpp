/**
 * web_dashboard_node.cpp
 *
 * Lightweight ROS2 node that subscribes to existing topics and serves
 * a live web dashboard over HTTP — no rosbridge, no extra dependencies.
 *
 * Endpoints:
 *   http://<JETSON_IP>:8080/          → dashboard HTML
 *   http://<JETSON_IP>:8080/cam1      → MJPEG live stream (cam1 with bboxes)
 *   http://<JETSON_IP>:8080/cam2      → MJPEG live stream (cam2)
 *   http://<JETSON_IP>:8080/detections → JSON detection state
 *
 * Notes:
 *   - Bounding boxes are only drawn on CAM1 because the stop-sign detections
 *     currently come from CAM1.
 *   - Camera topics fixed to:
 *       /camera/cam1/image_raw
 *       /camera/cam2/image_raw
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
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

// ── Tuning ────────────────────────────────────────────────────────────────────
static constexpr int HTTP_PORT    = 8080;
static constexpr int JPEG_QUALITY = 60;
static constexpr int STREAM_FPS   = 15;

// ── Shared frame buffers ──────────────────────────────────────────────────────
struct FrameBuffer {
    std::mutex         mtx;
    std::vector<uchar> jpeg;
    bool               ready = false;
};

static FrameBuffer g_cam1;
static FrameBuffer g_cam2;

// ── Detection state ───────────────────────────────────────────────────────────
// Keep overall detection state for the dashboard card / JSON endpoint,
// but keep boxes separate per camera so overlays do not bleed across feeds.
struct BBox {
    float x{};
    float y{};
    float w{};
    float h{};
    float conf{};
};

struct DetectionOverlay {
    std::mutex mtx;
    std::vector<BBox> boxes;
};

struct DetectionSummary {
    std::mutex mtx;
    bool  detected   = false;
    float confidence = 0.0f;
};

static DetectionSummary g_det_summary;
static DetectionOverlay g_det_cam1;
static DetectionOverlay g_det_cam2;  // currently unused / kept empty

// ── Draw bounding boxes onto frame ────────────────────────────────────────────
static void draw_boxes(cv::Mat& frame, const std::vector<BBox>& boxes)
{
    const cv::Scalar BOX_COLOR(50, 52, 232);
    const cv::Scalar LABEL_BG (50, 52, 232);
    const cv::Scalar LABEL_FG (255, 255, 255);

    for (const auto& b : boxes) {
        cv::Rect r(
            static_cast<int>(b.x),
            static_cast<int>(b.y),
            static_cast<int>(b.w),
            static_cast<int>(b.h)
        );

        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.empty()) continue;

        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, r, BOX_COLOR, cv::FILLED);
        cv::addWeighted(overlay, 0.12, frame, 0.88, 0, frame);
        cv::rectangle(frame, r, BOX_COLOR, 2);

        int cl = std::min(20, std::min(r.width, r.height) / 3);
        cv::line(frame, r.tl(),              {r.x + cl, r.y},               BOX_COLOR, 3);
        cv::line(frame, r.tl(),              {r.x, r.y + cl},               BOX_COLOR, 3);
        cv::line(frame, {r.x + r.width, r.y},{r.x + r.width - cl, r.y},    BOX_COLOR, 3);
        cv::line(frame, {r.x + r.width, r.y},{r.x + r.width, r.y + cl},    BOX_COLOR, 3);
        cv::line(frame, {r.x, r.y + r.height},{r.x + cl, r.y + r.height},  BOX_COLOR, 3);
        cv::line(frame, {r.x, r.y + r.height},{r.x, r.y + r.height - cl},  BOX_COLOR, 3);
        cv::line(frame, r.br(),              {r.x + r.width - cl, r.y + r.height}, BOX_COLOR, 3);
        cv::line(frame, r.br(),              {r.x + r.width, r.y + r.height - cl}, BOX_COLOR, 3);

        std::string label = "STOP  " + std::to_string(static_cast<int>(b.conf * 100.0f)) + "%";
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        cv::Point lp(r.x, std::max(r.y - 6, ts.height + 4));

        cv::rectangle(frame,
                      lp + cv::Point(0, -ts.height - 4),
                      lp + cv::Point(ts.width + 8, baseline),
                      LABEL_BG, cv::FILLED);

        cv::putText(frame, label, lp + cv::Point(4, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, LABEL_FG, 1, cv::LINE_AA);
    }

    std::string hud   = boxes.empty() ? "CLEAR" : "STOP SIGN DETECTED";
    cv::Scalar  hud_c = boxes.empty() ? cv::Scalar(80, 200, 80) : cv::Scalar(50, 52, 232);
    cv::putText(frame, hud, {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, hud_c, 2, cv::LINE_AA);
}

// ── Encode frame → JPEG → store in buffer ─────────────────────────────────────
static void encode_and_store(FrameBuffer& buf, cv::Mat& frame)
{
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", frame, jpeg, {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});

    std::lock_guard<std::mutex> lk(buf.mtx);
    buf.jpeg  = std::move(jpeg);
    buf.ready = true;
}

// ── Dashboard HTML (embedded) ─────────────────────────────────────────────────
static const char* DASHBOARD_HTML = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>AAV · Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&display=swap" rel="stylesheet"/>
<style>
  :root{--bg:#0a0c0f;--panel:#111418;--border:#1e2430;--accent:#e8341a;
    --safe:#1adb6e;--dim:#2a3040;--text:#c8d0dc;--muted:#4a5568;
    --mono:'Share Tech Mono',monospace;--head:'Bebas Neue',sans-serif}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);
    height:100vh;display:flex;flex-direction:column;overflow:hidden}
  body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,
    rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px)}

  header{display:flex;align-items:center;justify-content:space-between;
    padding:14px 24px;border-bottom:1px solid var(--border);
    background:var(--panel);flex-shrink:0}
  .logo{font-family:var(--head);font-size:1.8rem;letter-spacing:.15em;color:#fff}
  .logo span{color:var(--accent)}
  .hm{font-size:.68rem;color:var(--muted);text-align:right;line-height:1.9}
  #clock{color:var(--text)}

  .sbar{display:flex;align-items:center;gap:8px;padding:5px 24px;
    background:#0d1017;border-bottom:1px solid var(--border);
    font-size:.66rem;color:var(--muted);flex-shrink:0}
  .dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;
    transition:background .4s,box-shadow .4s}
  .dot.live{background:var(--safe);box-shadow:0 0 6px var(--safe)}
  .dot.waiting{background:#f0b429;box-shadow:0 0 6px #f0b429;animation:blink 1s infinite}
  .dot.err{background:var(--accent);box-shadow:0 0 6px var(--accent)}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

  main{flex:1;display:grid;grid-template-columns:320px 1fr;overflow:hidden}

  .lp{border-right:1px solid var(--border);display:flex;flex-direction:column;
    align-items:center;gap:14px;padding:20px 16px;overflow-y:auto;
    scrollbar-width:thin;scrollbar-color:var(--border) transparent}

  .sc{width:100%;background:var(--panel);border:1px solid var(--border);
    border-radius:4px;padding:26px 20px;display:flex;flex-direction:column;
    align-items:center;gap:13px;position:relative;overflow:hidden;transition:border-color .3s}
  .sc::before{content:'';position:absolute;inset:0;opacity:0;
    transition:opacity .5s;pointer-events:none;border-radius:4px}
  .sc.det{border-color:var(--accent);animation:pb .8s ease-out}
  .sc.det::before{background:radial-gradient(ellipse at center,
    rgba(232,52,26,.13) 0%,transparent 70%);opacity:1}
  @keyframes pb{0%{box-shadow:0 0 0 0 rgba(232,52,26,.6)}
    100%{box-shadow:0 0 0 22px rgba(232,52,26,0)}}

  .ss{width:108px;height:108px;transition:filter .4s,transform .3s}
  .ss .oc{fill:var(--dim);stroke:var(--muted);stroke-width:3;transition:fill .4s,stroke .4s}
  .ss .rg{fill:none;stroke:var(--muted);stroke-width:4;transition:stroke .4s}
  .ss .tx{fill:var(--muted);font-family:var(--head);font-size:36px;
    letter-spacing:4px;transition:fill .4s}
  .sc.det .ss{filter:drop-shadow(0 0 14px rgba(232,52,26,.7));transform:scale(1.05)}
  .sc.det .oc{fill:var(--accent);stroke:#ff6b55}
  .sc.det .tx{fill:#fff}
  .sc.det .rg{stroke:rgba(255,255,255,.55)}

  .stl{font-family:var(--head);font-size:2rem;letter-spacing:.12em;
    color:var(--muted);transition:color .3s}
  .sc.det .stl{color:var(--accent);text-shadow:0 0 18px rgba(232,52,26,.5)}
  .sts{font-size:.64rem;color:var(--muted);letter-spacing:.2em;text-transform:uppercase}
  .sc.det .sts{color:#e8341a99}

  .sr{display:grid;grid-template-columns:1fr 1fr;gap:10px;width:100%}
  .sb{background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:12px 14px}
  .sbl{font-size:.58rem;letter-spacing:.22em;color:var(--muted);margin-bottom:4px;text-transform:uppercase}
  .sbv{font-family:var(--head);font-size:1.6rem;color:var(--text)}

  .lb{width:100%;background:var(--panel);border:1px solid var(--border);border-radius:4px;overflow:hidden}
  .lh{padding:7px 12px;border-bottom:1px solid var(--border);font-size:.58rem;
    letter-spacing:.22em;color:var(--muted);text-transform:uppercase;
    display:flex;justify-content:space-between;align-items:center}
  .lh button{background:none;border:1px solid var(--border);color:var(--muted);
    font-family:var(--mono);font-size:.55rem;padding:2px 8px;cursor:pointer;
    transition:color .2s,border-color .2s}
  .lh button:hover{color:var(--text);border-color:var(--text)}
  #ll{list-style:none;max-height:120px;overflow-y:auto;
    scrollbar-width:thin;scrollbar-color:var(--border) transparent}
  #ll li{padding:6px 12px;font-size:.64rem;border-bottom:1px solid #1a1f28;
    display:flex;justify-content:space-between;animation:si .2s ease}
  @keyframes si{from{opacity:0;transform:translateX(-5px)}to{opacity:1;transform:translateX(0)}}
  #ll li .m{color:var(--accent)} #ll li .m.c{color:var(--safe)}
  #ll li .t{color:var(--muted);font-size:.6rem}
  #ll:empty::after{content:'No events yet';display:block;padding:12px;font-size:.64rem;color:var(--muted)}

  .rp{display:flex;flex-direction:column;overflow:hidden}
  .ct{display:flex;border-bottom:1px solid var(--border);background:var(--panel);flex-shrink:0}
  .tab{flex:1;padding:10px 0;font-family:var(--head);font-size:1rem;
    letter-spacing:.12em;text-align:center;cursor:pointer;color:var(--muted);
    background:transparent;border:none;border-right:1px solid var(--border);
    position:relative;transition:color .2s,background .2s}
  .tab:last-child{border-right:none}
  .tab::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:var(--accent);transform:scaleX(0);transition:transform .2s}
  .tab.active{color:var(--text);background:rgba(232,52,26,.05)}
  .tab.active::after{transform:scaleX(1)}

  .cv{flex:1;background:#06080a;overflow:hidden;display:none;position:relative}
  .cv.active{display:flex;align-items:center;justify-content:center}
  .cv canvas{width:100%;height:100%;object-fit:contain;display:block}

  .placeholder{position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:10px;color:var(--muted)}
  .placeholder .pi{font-size:2.5rem;opacity:.25}
  .placeholder .pt{font-family:var(--head);font-size:1.2rem;letter-spacing:.15em;opacity:.3}
  .placeholder .ps{font-size:.65rem;letter-spacing:.2em;opacity:.28}
  .placeholder.hidden{display:none}

  .cl{position:absolute;top:10px;left:10px;background:rgba(10,12,15,.78);
    border:1px solid var(--border);border-radius:3px;padding:3px 10px;
    font-size:.6rem;letter-spacing:.18em;color:var(--muted);pointer-events:none}
  .db{position:absolute;top:10px;right:10px;background:rgba(232,52,26,.15);
    border:1px solid var(--accent);border-radius:3px;padding:3px 12px;
    font-family:var(--head);font-size:.68rem;letter-spacing:.18em;
    color:var(--accent);opacity:0;transition:opacity .3s;animation:blink .7s infinite}
  .db.on{opacity:1}

  footer{padding:8px 24px;border-top:1px solid var(--border);font-size:.6rem;
    color:var(--muted);display:flex;justify-content:space-between;
    letter-spacing:.1em;flex-shrink:0}
</style>
</head>
<body>
<header>
  <div class="logo">AAV<span>.</span>VISION</div>
  <div class="hm"><div id="clock">--:--:--</div><div>LIVE · MJPEG · JETSON</div></div>
</header>
<div class="sbar">
  <div class="dot waiting" id="sd"></div>
  <span id="sl">WAITING FOR FIRST FRAME…</span>
</div>

<main>
  <div class="lp">
    <div class="sc" id="sc">
      <svg class="ss" viewBox="0 0 160 160">
        <polygon class="oc" points="56,10 104,10 150,56 150,104 104,150 56,150 10,104 10,56"/>
        <polygon class="rg" points="60,18 100,18 142,60 142,100 100,142 60,142 18,100 18,60"/>
        <text class="tx" x="80" y="94" text-anchor="middle">STOP</text>
      </svg>
      <div class="stl" id="stl">CLEAR</div>
      <div class="sts" id="sts">NO STOP SIGN DETECTED</div>
    </div>
    <div class="sr">
      <div class="sb"><div class="sbl">Detections</div><div class="sbv" id="tc">0</div></div>
      <div class="sb"><div class="sbl">Last Seen</div>
        <div class="sbv" id="ls" style="font-size:.85rem;padding-top:6px">—</div></div>
    </div>
    <div class="lb">
      <div class="lh"><span>Event Log</span>
        <button onclick="document.getElementById('ll').innerHTML=''">CLEAR</button></div>
      <ul id="ll"></ul>
    </div>
  </div>

  <div class="rp">
    <div class="ct">
      <button class="tab active" id="t1" onclick="sw('cam1')">CAM 1</button>
      <button class="tab"        id="t2" onclick="sw('cam2')">CAM 2</button>
    </div>
    <div class="cv active" id="v1">
      <div class="placeholder" id="ph1">
        <div class="pi">◉</div><div class="pt">CAMERA 1</div>
        <div class="ps">AWAITING STREAM</div>
      </div>
      <canvas id="c1"></canvas>
      <div class="cl">CAM1 · MJPEG LIVE</div>
      <div class="db" id="b1">● STOP SIGN</div>
    </div>
    <div class="cv" id="v2">
      <div class="placeholder" id="ph2">
        <div class="pi">◉</div><div class="pt">CAMERA 2</div>
        <div class="ps">AWAITING STREAM</div>
      </div>
      <canvas id="c2"></canvas>
      <div class="cl">CAM2 · MJPEG LIVE</div>
      <div class="db" id="b2">● STOP SIGN</div>
    </div>
  </div>
</main>
<footer>
  <span id="src">STREAM: http://JETSON:8080/cam1  &amp;  /cam2</span>
  <span id="fc">FRAMES: 0</span>
</footer>

<script>
  setInterval(() => {
    document.getElementById('clock').textContent =
      new Date().toLocaleTimeString('en-CA', {hour12:false})
  }, 1000)

  function sw(cam) {
    ['cam1','cam2'].forEach((c, i) => {
      document.getElementById('v' + (i + 1)).classList.toggle('active', c === cam)
      document.getElementById('t' + (i + 1)).classList.toggle('active', c === cam)
    })
  }

  let streamLive = false
  let totalFrames = 0

  function setLive() {
    if (streamLive) return
    streamLive = true
    document.getElementById('sd').className = 'dot live'
    document.getElementById('sl').textContent = 'LIVE · MJPEG STREAM ACTIVE'
    document.getElementById('src').textContent =
      'STREAM: ' + location.hostname + ':8080/cam1  &  /cam2'
  }

  function setError(msg) {
    if (!streamLive) {
      document.getElementById('sd').className = 'dot err'
      document.getElementById('sl').textContent = msg
    }
  }

  async function startMjpegStream(url, canvasId, placeholderId) {
    const canvas = document.getElementById(canvasId)
    const ctx    = canvas.getContext('2d')
    const ph     = document.getElementById(placeholderId)

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

            const blob   = new Blob([jpeg], {type: 'image/jpeg'})
            const objUrl = URL.createObjectURL(blob)
            const img    = new Image()

            img.onload = () => {
              const container = canvas.parentElement
              if (canvas.width !== container.clientWidth || canvas.height !== container.clientHeight) {
                canvas.width  = container.clientWidth
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

  let total = 0
  let active = false
  let clearTimer = null

  async function poll() {
    try {
      const r = await fetch('/detections')
      const d = await r.json()
      update(d.detected)
    } catch (e) {}
    setTimeout(poll, 200)
  }
  poll()

  function update(det) {
    const sc  = document.getElementById('sc')
    const stl = document.getElementById('stl')
    const sts = document.getElementById('sts')

    // Detection belongs to CAM1 only.
    document.getElementById('b1').classList.toggle('on', det)
    document.getElementById('b2').classList.remove('on')

    if (det && !active) {
      active = true
      total++
      document.getElementById('tc').textContent = total
      const now = new Date().toLocaleTimeString('en-CA', {hour12:false})
      document.getElementById('ls').textContent = now
      sc.classList.add('det')
      stl.textContent = 'STOP SIGN'
      sts.textContent = 'DETECTED'
      addLog('STOP SIGN DETECTED (CAM1)', now, false)
    }

    if (det) {
      if (clearTimer) clearTimeout(clearTimer)
      clearTimer = setTimeout(() => {
        active = false
        sc.classList.remove('det')
        stl.textContent = 'CLEAR'
        sts.textContent = 'NO STOP SIGN DETECTED'
        addLog('CLEAR', new Date().toLocaleTimeString('en-CA', {hour12:false}), true)
      }, 2000)
    }
  }

  function addLog(msg, time, clear) {
    const li = document.createElement('li')
    li.innerHTML = `<span class="m ${clear ? 'c' : ''}">${msg}</span><span class="t">${time}</span>`
    const l = document.getElementById('ll')
    l.prepend(li)
    while (l.children.length > 50) l.removeChild(l.lastChild)
  }
</script>
</body>
</html>
)HTML";

// ── HTTP helpers ──────────────────────────────────────────────────────────────
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

// ── MJPEG stream per-connection ───────────────────────────────────────────────
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

// ── HTTP server thread ────────────────────────────────────────────────────────
static void http_server()
{
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(HTTP_PORT);

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
            } else if (path == "/detections") {
                bool detected = false;
                float confidence = 0.0f;

                {
                    std::lock_guard<std::mutex> lk(g_det_summary.mtx);
                    detected   = g_det_summary.detected;
                    confidence = g_det_summary.confidence;
                }

                std::string json =
                    std::string("{\"detected\":") +
                    (detected ? "true" : "false") +
                    ",\"confidence\":" + std::to_string(confidence) + "}";

                send_str(client, "200 OK", "application/json", json);
                close(client);
            } else {
                send_str(client, "200 OK", "text/html", DASHBOARD_HTML);
                close(client);
            }
        }).detach();
    }
}

// ── ROS2 Node ─────────────────────────────────────────────────────────────────
class WebDashboardNode : public rclcpp::Node
{
public:
    WebDashboardNode() : Node("web_dashboard_node")
    {
        sub_detected_ = create_subscription<std_msgs::msg::Bool>(
            "/aav/stop_sign_detected", 10,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(g_det_summary.mtx);
                g_det_summary.detected = msg->data;
            });

        sub_confidence_ = create_subscription<std_msgs::msg::Float32>(
            "/aav/stop_sign_confidence", 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(g_det_summary.mtx);
                g_det_summary.confidence = msg->data;
            });

        sub_detections_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            "/aav/stop_sign_detections", 10,
            [this](const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
                std::vector<BBox> cam1_boxes;

                for (const auto& d : msg->detections) {
                    BBox b;
                    b.x    = d.bbox.center.position.x - d.bbox.size_x / 2.0f;
                    b.y    = d.bbox.center.position.y - d.bbox.size_y / 2.0f;
                    b.w    = d.bbox.size_x;
                    b.h    = d.bbox.size_y;
                    b.conf = d.results.empty() ? 0.0f : d.results[0].hypothesis.score;
                    cam1_boxes.push_back(b);
                }

                {
                    std::lock_guard<std::mutex> lk(g_det_cam1.mtx);
                    g_det_cam1.boxes = std::move(cam1_boxes);
                }

                // Clear CAM2 overlays so stale/shared boxes never appear there.
                {
                    std::lock_guard<std::mutex> lk(g_det_cam2.mtx);
                    g_det_cam2.boxes.clear();
                }
            });

        auto qos = rclcpp::SensorDataQoS();

        sub_cam1_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/cam1/image_raw", qos,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                process_frame(msg, g_cam1, g_det_cam1);
            });

        sub_cam2_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/cam2/image_raw", qos,
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                process_frame(msg, g_cam2, g_det_cam2);
            });

        http_thread_ = std::thread(http_server);
        http_thread_.detach();

        RCLCPP_INFO(get_logger(),
            "Web dashboard running at http://<JETSON_IP>:%d", HTTP_PORT);
        RCLCPP_INFO(get_logger(),
            "Subscribed camera topics: /camera/cam1/image_raw and /camera/cam2/image_raw");
    }

private:
    void process_frame(const sensor_msgs::msg::Image::SharedPtr& msg,
                       FrameBuffer& buf,
                       DetectionOverlay& det_overlay)
    {
        try {
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();

            std::vector<BBox> boxes_copy;
            {
                std::lock_guard<std::mutex> lk(det_overlay.mtx);
                boxes_copy = det_overlay.boxes;
            }

            draw_boxes(frame, boxes_copy);
            encode_and_store(buf, frame);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(get_logger(), "cv_bridge error: %s", e.what());
        }
    }

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr                sub_detected_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr             sub_confidence_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_detections_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr            sub_cam1_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr            sub_cam2_;

    std::thread http_thread_;
};

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WebDashboardNode>());
    rclcpp::shutdown();
    return 0;
}
