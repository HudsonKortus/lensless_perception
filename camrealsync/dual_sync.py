#!/usr/bin/env python3
import os, time, threading, numpy as np, cv2
import pyrealsense2 as rs
from pathlib import Path
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

# CONFIG
TARGET_FPS = 30
OUT_DIR = "./captures"
RESOLUTION = None
stop_event = threading.Event()

session_time = time.strftime("%Y%m%d_%H%M%S")
session_dir = Path(OUT_DIR) / f"session_{session_time}"
session_dir.mkdir(parents=True, exist_ok=True)

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

# === REALSENSE ===
class RealsenseCam:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, TARGET_FPS)
        self.pipeline.start(cfg)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            return None
        img = np.asanyarray(color.get_data())
        if RESOLUTION:
            img = cv2.resize(img, RESOLUTION)
        return img

    def close(self):
        self.pipeline.stop()

# === CAPTURE + DISPLAY LOOP ===
def capture_loop():
    ar_writer = None
    rs_writer = None
    frames_written = 0

    while not stop_event.is_set():
        ar_frame = ardu_cam.get_frame()
        rs_frame = rs_cam.get_frame()

        if ar_frame is None or rs_frame is None:
            continue

        # Initialize VideoWriters
        if ar_writer is None:
            h, w = ar_frame.shape[:2]
            ar_writer = cv2.VideoWriter(str(session_dir / "arducam.mov"),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        TARGET_FPS, (w, h))
        if rs_writer is None:
            h, w = rs_frame.shape[:2]
            rs_writer = cv2.VideoWriter(str(session_dir / "realsense.mov"),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        TARGET_FPS, (w, h))

        # Write exactly what is displayed
        ar_writer.write(ar_frame)
        rs_writer.write(rs_frame)
        frames_written += 1

        cv2.imshow("ArduCam", ar_frame)
        cv2.imshow("Realsense", rs_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    if ar_writer: ar_writer.release()
    if rs_writer: rs_writer.release()
    cv2.destroyAllWindows()
    print(f"[SYNC] Video recording stopped, {frames_written} frames written.")

# === MAIN ===
if __name__ == "__main__":
    try:
        print("[MAIN] Initializing cameras...")
        ardu_cam = ArduCam()
        rs_cam = RealsenseCam()

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
        print("[MAIN] Done.")
