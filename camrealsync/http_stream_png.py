#!/usr/bin/env python3
import os, time, threading, numpy as np, cv2
import pyrealsense2 as rs
from pathlib import Path
import gi
import datetime 
import zipfile


###CHAT CODE
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

NTH_TO_SHARE = 10        # share every 10th saved frame
HTTP_PORT = 8000         # change if needed

_shared = {
    "arducam_jpeg": b"",
    "rgb_jpeg": b"",
    "depth_png": b"",    # colorized depth PNG
}
_frame_counter_for_share = 0
###END CHAT CODE


gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

# CONFIG
TARGET_FPS = 30
CAPTURE_FZ = .5
OUT_DIR = "~/media/pear/ssssd/DFD_DOE_Test_2"
ARDUCAM_DIR = "arducam" + '/'
REALDEPTH_DIR = "realdepth" + '/'
REAL_RGB_DIR = "realsense" + '/'

RESOLUTION = None
stop_event = threading.Event()

session_time = time.strftime("%Y%m%d_%H%M%S")
session_dir = Path(OUT_DIR) / f"session_{session_time}"
session_dir.mkdir(parents=True, exist_ok=True)

arducam_dir = Path(session_dir/ARDUCAM_DIR)
arducam_dir.mkdir(parents=True, exist_ok=True)

realdepth_dir = Path(session_dir/REALDEPTH_DIR)
realdepth_dir.mkdir(parents=True, exist_ok=True)

real_rgb_dir = Path(session_dir/REAL_RGB_DIR)
real_rgb_dir.mkdir(parents=True, exist_ok=True)


# === Helper: Crop to match target aspect ratio ===
def crop_center(img, target_ratio):
    h, w = img.shape[:2]
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is too wide
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        return img[:, x1:x1+new_w]
    else:
        # Image is too tall
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        return img[y1:y1+new_h, :]

# === ARDUCAM ===
class ArduCam:
    def __init__(self):
        Gst.init(None)
        pipeline_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.pipeline.set_state(Gst.State.PLAYING)
        time.sleep(1)

    def get_frame(self):
        sample = self.appsink.try_pull_sample(2_000_000_000)
        if not sample or stop_event.is_set():
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            img = np.frombuffer(map_info.data, dtype=np.uint8)
            img = img.reshape((h, w, 3))  # Ensure 3 channels
            if RESOLUTION:
                img = cv2.resize(img, RESOLUTION)
            return img
        finally:
            buf.unmap(map_info)

    def close(self):
        self.pipeline.set_state(Gst.State.NULL)
        
# === REALSENSE (RGB + DEPTH) ===
class RealsenseCam:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # Enable color and depth streams
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, TARGET_FPS)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, TARGET_FPS)

        # Start streaming
        self.pipeline.start(cfg)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # in millimeters

        # Crop to 16:9 (like Arducam)
        target_aspect_ratio = 16 / 9
        color_image = crop_center(color_image, target_aspect_ratio)
        depth_image = crop_center(depth_image, target_aspect_ratio)

        if RESOLUTION:
            color_image = cv2.resize(color_image, RESOLUTION)
            depth_image = cv2.resize(depth_image, RESOLUTION)

        return color_image, depth_image

    def close(self):
        self.pipeline.stop()

# === CAPTURE + DISPLAY LOOP ===
def capture_loop():
    ar_writer = None
    rs_writer = None
    rsd_writer = None
    period = 1/CAPTURE_FZ
    frames_written = 0
    file_num = 0
    last_run = datetime.datetime.now()
    
    while not stop_event.is_set():
        now = datetime.datetime.now()
        delta = (now - last_run).total_seconds()
        if delta >= period:  
            file_name = f'{file_num}.png'
            ar_frame = ardu_cam.get_frame()
            real_rgb_result, real_depth_result = rs_cam.get_frame()


            
            cv2.imwrite(os.path.join(arducam_dir,file_name), ar_frame)
            cv2.imwrite(os.path.join(real_rgb_dir,file_name), real_rgb_result)
            cv2.imwrite(os.path.join(realdepth_dir,file_name), real_depth_result)
            
            file_num += 1
            last_run = datetime.datetime.now()
            # Share every Nth saved frame to the HTTP server
            maybe_share_every_nth(ar_frame, real_rgb_result, real_depth_result, nth=NTH_TO_SHARE)
            
        else:
            time.sleep(0.0005)

        # frames_written += 1

        #TODO: post every nth frame to http server
        # Show both feeds 
        # cv2.imshow("ArduCam", ar_frame)
        # cv2.imshow("Realsense RGB (cropped)", color_frame)
        # cv2.imshow("Realsense Depth (cropped)", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    # Release resources
    if ar_writer: ar_writer.release()
    if rs_writer: rs_writer.release()
    if rsd_writer: rsd_writer.release()
    cv2.destroyAllWindows()
    print(f"[SYNC] Video recording stopped, {frames_written} frames written.")

###START CHAT CODE

def _encode_jpeg(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""

def _encode_png(img_bgr_or_gray):
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    return buf.tobytes() if ok else b""

def _colorize_depth_for_view(depth_mm):
    # Scale (0..4000mm) to 0..255 for viewing; adjust as you like
    depth_clip = np.clip(depth_mm.astype(np.float32), 0, 4000)
    depth_8u = cv2.convertScaleAbs(depth_clip, alpha=(255.0/4000.0))
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_color

class _FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/arducam.jpg":
            payload, ctype = _shared["arducam_jpeg"], "image/jpeg"
        elif path == "/rgb.jpg":
            payload, ctype = _shared["rgb_jpeg"], "image/jpeg"
        elif path == "/depth.png":
            payload, ctype = _shared["depth_png"], "image/png"
        else:
            html = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Frames</title></head>
<body style="font-family:sans-serif">
<h2>Live frames (every 10th saved)</h2>
<div style="display:flex;gap:16px;flex-wrap:wrap">
<div><h4>ArduCam</h4><img src="/arducam.jpg" width="480" /></div>
<div><h4>Realsense RGB</h4><img src="/rgb.jpg" width="480" /></div>
<div><h4>Realsense Depth</h4><img src="/depth.png" width="480" /></div>
</div>
<script>
  // refresh images every 500ms without caching
  setInterval(()=> {
    for (const p of ["arducam.jpg","rgb.jpg","depth.png"]) {
      const img = document.querySelector(`img[src$='${p}']`);
      if (img) img.src = `/${p}?t=` + Date.now();
    }
  }, 500);
</script>
</body></html>"""
            payload, ctype = html, "text/html; charset=utf-8"

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

def start_http_server(port=HTTP_PORT):
    srv = HTTPServer(("0.0.0.0", port), _FrameHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[HTTP] Serving on http://0.0.0.0:{port}  (open / for a viewer)")
    
def maybe_share_every_nth(ar_img_bgr, rgb_bgr, depth_mm, nth=NTH_TO_SHARE):
    global _frame_counter_for_share
    _frame_counter_for_share += 1
    if _frame_counter_for_share % nth != 0:
        return

    # Encode ArduCam & RGB to JPEG
    _shared["arducam_jpeg"] = _encode_jpeg(ar_img_bgr)
    _shared["rgb_jpeg"] = _encode_jpeg(rgb_bgr)

    # Colorize depth for the browser, then encode as PNG
    depth_color = _colorize_depth_for_view(depth_mm)
    _shared["depth_png"] = _encode_png(depth_color)
###end CHAT CODE



# === MAIN ===
if __name__ == "__main__":
    try:
        print("[MAIN] Initializing cameras...")
        ardu_cam = ArduCam()
        rs_cam = RealsenseCam()
        start_http_server(HTTP_PORT)

        print(f"[MAIN] Recording videos to {session_dir}")
        print("[MAIN] Press Enter to stop recording or 'q' to quit display")

        capture_thread = threading.Thread(target=capture_loop)
        capture_thread.start()

        try:
            input()
        except KeyboardInterrupt:
            print("\n[MAIN] KeyboardInterrupt received")
        stop_event.set()
        capture_thread.join()
    finally:
        print("[MAIN] Closing cameras...")
        if 'ardu_cam' in globals(): ardu_cam.close()
        if 'rs_cam' in globals(): rs_cam.close()
        with ZipFile(session_dir) as myzip:
            myzip
            
        print("[MAIN] Done.")
