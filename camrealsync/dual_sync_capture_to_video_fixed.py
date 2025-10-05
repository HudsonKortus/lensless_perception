# dual_sync_capture_to_video_fixed.py
import os, csv, time, threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

# =========================
# CONFIG — edit these
# =========================
# Where to put session folders (CSVs + run_info.txt + MP4s)
OUT_DIR = "/mnt/ssd/captures"

DURATION_S = 10            # seconds to record after start
FPS = 30                   # target FPS for both streams

# CSI (ArduCam) settings
CSI_WIDTH = 1280
CSI_HEIGHT = 720
CSI_SENSOR_ID = 0
CSI_FLIP = 0               # 0..7 (Jetson nvvidconv flip-method)

# RealSense (depth) settings
RS_WIDTH = 640
RS_HEIGHT = 480
RS_MAX_DEPTH_M = 3.0       # clamp > this to 0 for visualization
# Optional RealSense visual preset. Common D4xx values:
#   0=Custom, 1=Default, 2=Hand, 3=High Accuracy, 4=High Density, 5=Medium Density, 6=Short Range
# Set to None to skip preset, or to an int like 6 for "Short Range".
RS_VISUAL_PRESET = None

# Sync & warmup
READY_TIMEOUT_S = 10.0     # fail fast if a device never gets ready
CSI_WARMUP_S = 1.0         # seconds of reads before signaling ready
RS_WARMUP_FRAMES = 30      # frames before signaling ready
START_OFFSET_S = 0.10      # start 100ms after both are ready
# =========================


# ---------- Helpers ----------
def now_ns():
    return time.perf_counter_ns()  # monotonic

def iso_utc():
    return datetime.now(timezone.utc).isoformat()

def make_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

def open_csv(csv_path: Path, headers):
    first = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if first:
        w.writerow(headers)
    return f, w

class Box:
    def __init__(self, value=None):
        self.value = value


# ---------- CSI (ArduCam) via GStreamer ----------
def csi_gst_pipeline(width, height, fps, sensor_id=0, flip_method=0):
    # NV12 -> BGR
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"! video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1, format=NV12 "
        f"! nvvidconv flip-method={flip_method} "
        f"! video/x-raw, format=BGRx "
        f"! videoconvert "
        f"! video/x-raw, format=BGR "
        f"! appsink drop=true max-buffers=1 sync=false"
    )

def csi_capture_loop(session_dir, width, height, fps, sensor_id,
                     ready_evt, start_evt, start_ns_box, end_ns_box,
                     csi_csv_path, video_path, csi_start_ns_box, csi_end_ns_box):
    fcsv, writer = open_csv(csi_csv_path, ["frame_idx","monotonic_ns","walltime_iso"])
    cap = None
    writer_mp4 = None
    try:
        # Open camera
        cap = cv2.VideoCapture(
            csi_gst_pipeline(width, height, fps, sensor_id, CSI_FLIP),
            cv2.CAP_GSTREAMER
        )
        if not cap.isOpened():
            raise RuntimeError("Failed to open CSI camera via nvarguscamerasrc. Check sensor-id and drivers.")

        # Warm up
        t0 = time.time()
        while time.time() - t0 < CSI_WARMUP_S:
            cap.read()

        # Prepare MP4 writer
        ensure_parent(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # software encoder (simple & universal)
        writer_mp4 = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer_mp4.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for CSI at {video_path}")

        # Ready → wait for global start
        ready_evt.set()
        start_evt.wait()
        start_ns = start_ns_box.value
        end_ns = end_ns_box.value

        frame_interval_ns = int(1e9 / fps)
        next_frame_ns = start_ns

        idx = 0
        first_frame_written = False

        while now_ns() < end_ns:
            t_ns = now_ns()
            if t_ns < next_frame_ns:
                time.sleep(max(0.0, (next_frame_ns - t_ns)/1e9))

            ok, frame = cap.read()
            t_ns = now_ns()
            if not ok:
                next_frame_ns += frame_interval_ns
                continue

            # write frame to MP4 and log timestamp
            writer_mp4.write(frame)
            writer.writerow([idx, t_ns, iso_utc()])

            if not first_frame_written:
                csi_start_ns_box.value = t_ns  # ACTUAL first frame timestamp
                first_frame_written = True

            idx += 1
            next_frame_ns += frame_interval_ns

        csi_end_ns_box.value = now_ns()

    finally:
        try:
            if cap is not None:
                cap.release()
        except:
            pass
        try:
            if writer_mp4 is not None:
                writer_mp4.release()
        except:
            pass
        fcsv.close()


# ---------- RealSense (Depth) ----------
def rs_capture_loop(session_dir, width, height, fps,
                    ready_evt, start_evt, start_ns_box, end_ns_box,
                    rs_csv_path, video_path, max_depth_m,
                    rs_start_ns_box, rs_end_ns_box,
                    visual_preset=None):
    fcsv, writer = open_csv(rs_csv_path, ["frame_idx","monotonic_ns","walltime_iso"])
    pipe = None
    writer_mp4 = None
    try:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        pipe = rs.pipeline()
        profile = pipe.start(cfg)

        # Depth scale (meters per unit) — CRITICAL for correct meters <-> units
        depth_sensor = profile.get_device().first_depth_sensor()
        try:
            depth_scale = depth_sensor.get_depth_scale()  # meters per depth unit
        except Exception:
            # Fallback (typical value), but warn in metadata later
            depth_scale = 0.001

        # Optional visual preset
        try:
            if visual_preset is not None:
                if depth_sensor.supports(rs.option.visual_preset):
                    depth_sensor.set_option(rs.option.visual_preset, float(visual_preset))
        except Exception:
            # ignore if not supported or invalid
            pass

        # Warm up
        for _ in range(RS_WARMUP_FRAMES):
            pipe.wait_for_frames()

        # Prepare MP4 writer (we write an 8-bit visualization)
        ensure_parent(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_mp4 = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer_mp4.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for RealSense at {video_path}")

        # Ready → wait for global start
        ready_evt.set()
        start_evt.wait()
        start_ns = start_ns_box.value
        end_ns = end_ns_box.value

        idx = 0
        first_frame_written = False
        max_depth_m = float(max_depth_m)

        while now_ns() < end_ns:
            frames = pipe.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth:
                continue

            # Get raw units (uint16), convert to meters using device's scale
            depth_units = np.asanyarray(depth.get_data()).astype(np.float32)
            depth_m = depth_units * float(depth_scale)

            # Clamp > max to 0.0 for visualization (keeps your original intent)
            depth_m[depth_m > max_depth_m] = 0.0

            # Visualize 0..max_depth_m -> 0..255
            # Avoid division by zero if max_depth_m == 0
            if max_depth_m > 0:
                vis8 = cv2.convertScaleAbs(depth_m, alpha=255.0/max_depth_m)
            else:
                vis8 = np.zeros_like(depth_units, dtype=np.uint8)

            vis8 = cv2.cvtColor(vis8, cv2.COLOR_GRAY2BGR)

            t_ns = now_ns()
            writer_mp4.write(vis8)
            writer.writerow([idx, t_ns, iso_utc()])

            if not first_frame_written:
                rs_start_ns_box.value = t_ns  # ACTUAL first frame timestamp
                first_frame_written = True

            idx += 1

        rs_end_ns_box.value = now_ns()

    finally:
        try:
            if pipe is not None:
                pipe.stop()
        except:
            pass
        try:
            if writer_mp4 is not None:
                writer_mp4.release()
        except:
            pass
        fcsv.close()


# ---------- Main ----------
def main():
    # session folder (for CSVs + run_info + sync_summary + MP4s)
    out_root = make_dir(Path(OUT_DIR))
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = make_dir(out_root / f"session_{session_tag}")

    # Compute per-run MP4 paths inside the session (prevents overwrite)
    csi_mp4_path = session_dir / "csi_rgb.mp4"
    rs_mp4_path  = session_dir / "rs_depth_vis.mp4"

    # events/boxes
    ready_csi = threading.Event()
    ready_rs  = threading.Event()
    start_evt = threading.Event()
    start_ns_box = Box(None)
    end_ns_box   = Box(None)

    csi_start_ns_box = Box(None)
    csi_end_ns_box   = Box(None)
    rs_start_ns_box  = Box(None)
    rs_end_ns_box    = Box(None)

    # per-camera CSV logs
    csi_csv = session_dir / "csi_timestamps.csv"
    rs_csv  = session_dir / "realsense_timestamps.csv"

    # spawn
    t_csi = threading.Thread(
        target=csi_capture_loop,
        args=(session_dir, CSI_WIDTH, CSI_HEIGHT, FPS, CSI_SENSOR_ID,
              ready_csi, start_evt, start_ns_box, end_ns_box,
              csi_csv, csi_mp4_path, csi_start_ns_box, csi_end_ns_box),
        daemon=True
    )
    t_rs = threading.Thread(
        target=rs_capture_loop,
        args=(session_dir, RS_WIDTH, RS_HEIGHT, FPS,
              ready_rs, start_evt, start_ns_box, end_ns_box,
              rs_csv, rs_mp4_path, RS_MAX_DEPTH_M,
              rs_start_ns_box, rs_end_ns_box, RS_VISUAL_PRESET),
        daemon=True
    )

    print("[info] opening devices and warming up...")
    t_csi.start()
    t_rs.start()

    # wait for readiness
    if not ready_csi.wait(timeout=READY_TIMEOUT_S):
        raise RuntimeError("CSI camera did not become ready in time.")
    if not ready_rs.wait(timeout=READY_TIMEOUT_S):
        raise RuntimeError("RealSense did not become ready in time.")

    # common start/end
    start_ns = now_ns() + int(START_OFFSET_S * 1e9)
    end_ns   = start_ns + int(DURATION_S * 1e9)
    start_ns_box.value = start_ns
    end_ns_box.value   = end_ns

    # metadata
    run_info_path = session_dir / "run_info.txt"
    with open(run_info_path, "w") as f:
        f.write(f"started_utc={iso_utc()}\n")
        f.write(f"duration_s={DURATION_S}\n")
        f.write(f"fps={FPS}\n")
        f.write(f"start_ns={start_ns}\n")
        f.write(f"end_ns={end_ns}\n")
        f.write(f"csi={CSI_WIDTH}x{CSI_HEIGHT}@{FPS}, sensor_id={CSI_SENSOR_ID}, flip={CSI_FLIP}\n")
        f.write(f"rs_depth={RS_WIDTH}x{RS_HEIGHT}@{FPS}, max_depth_m={RS_MAX_DEPTH_M}\n")
        f.write(f"rs_visual_preset={RS_VISUAL_PRESET}\n")
        f.write(f"csi_mp4={csi_mp4_path}\n")
        f.write(f"rs_mp4={rs_mp4_path}\n")

    print("[info] synchronized start scheduled...")
    start_evt.set()

    # join
    t_csi.join()
    t_rs.join()

    # quick sync summary CSV
    sync_csv = session_dir / "sync_summary.csv"
    with open(sync_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream","start_ns_seen","end_ns_seen","start_delta_ns_vs_global","end_delta_ns_vs_global"])
        w.writerow([
            "CSI",
            csi_start_ns_box.value, csi_end_ns_box.value,
            (None if csi_start_ns_box.value is None else csi_start_ns_box.value - start_ns),
            (None if csi_end_ns_box.value   is None else csi_end_ns_box.value   - end_ns)
        ])
        w.writerow([
            "RealSense",
            rs_start_ns_box.value, rs_end_ns_box.value,
            (None if rs_start_ns_box.value is None else rs_start_ns_box.value - start_ns),
            (None if rs_end_ns_box.value   is None else rs_end_ns_box.value   - end_ns)
        ])

    print(f"[done] Videos saved:\n  CSI -> {csi_mp4_path}\n  RS  -> {rs_mp4_path}")
    print(f"[done] Logs & summary in: {session_dir}")


if __name__ == "__main__":
    main()