#!/usr/bin/env python3
"""
Face Lock — Ubuntu Screen Lock via Face Recognition
====================================================
Locks your screen when YOUR face is not detected.
Uses guided multi-pose enrollment for high accuracy.

INSTALL:
    pip install opencv-contrib-python --break-system-packages

FIRST TIME SETUP (enroll your face):
    python3 face_lock.py --enroll

DAILY USE:
    python3 face_lock.py

AUTO-START ON LOGIN:
    python3 face_lock.py --install
"""

import cv2
import subprocess
import time
import sys
import signal
import logging
import argparse
import os
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CAMERA_INDEX       = 0
GRACE_PERIOD       = 20.0      # seconds without YOUR face before locking
CHECK_INTERVAL     = 0.8      # seconds between recognition checks
CONFIDENCE_THRESH  = 75       # LBPH: lower = stricter. Tune if needed (60-85)
FACE_MODEL_PATH    = os.path.expanduser("~/.face_lock_model.xml")
SHOW_PREVIEW       = True

# Enrollment poses — each captures SAMPLES_PER_POSE frames
ENROLLMENT_POSES = [
    ("CENTER",       "Look straight at the camera",         (0, 255, 120)),
    ("SLIGHT LEFT",  "Slowly turn your head a little LEFT", (0, 220, 255)),
    ("SLIGHT RIGHT", "Slowly turn your head a little RIGHT",(255, 200, 0)),
    ("CHIN DOWN",    "Tilt your chin slightly DOWN",         (200, 120, 255)),
    ("CHIN UP",      "Tilt your chin slightly UP",           (255, 120, 120)),
    ("EXPRESSIONS",  "Vary your expression — smile, neutral, serious", (120, 255, 200)),
]
SAMPLES_PER_POSE   = 25       # samples per pose  → 6 × 25 = 150 total

# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_lock")


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    casc = cv2.CascadeClassifier(path)
    if casc.empty():
        sys.exit("❌  Could not load Haar cascade. Is opencv-contrib-python installed?")
    return casc


def get_face(frame, cascade, min_size=(90, 90)):
    """Return (crop_200x200_gray, rect) for the largest detected face, else (None, None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.25, minNeighbors=6, minSize=min_size
    )
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    return crop, (x, y, w, h)


def lock_screen():
    for cmd in (
        ["loginctl", "lock-session"],
        ["dbus-send", "--type=method_call",
         "--dest=org.gnome.ScreenSaver",
         "/org/gnome/ScreenSaver",
         "org.gnome.ScreenSaver.Lock"],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            log.info("🔒  Screen locked.")
            return
        except Exception:
            continue
    log.error("Could not lock screen — no working lock command found.")


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=2):
    """Draw a rounded rectangle on img."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius),   0, 0, 90, color, thickness)


def overlay_hud(frame, line1, line2, color, rect=None,
                progress=None, progress_color=(0, 200, 100)):
    """Draw HUD overlay on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 56), (10, 10, 10), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    # Status text
    cv2.putText(frame, line1, (14, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2, (14, 48),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (180, 180, 180), 1, cv2.LINE_AA)

    # Face bounding box
    if rect is not None:
        x, y, bw, bh = rect
        draw_rounded_rect(frame, (x-4, y-4), (x+bw+4, y+bh+4), color, radius=10, thickness=2)

    # Progress bar at bottom
    if progress is not None:
        cv2.rectangle(frame, (0, h-6), (w, h), (30, 30, 30), -1)
        filled = int(progress * w)
        cv2.rectangle(frame, (0, h-6), (filled, h), progress_color, -1)

    # Footer hint
    cv2.putText(frame, "Q  quit", (w-70, h-10),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (90, 90, 90), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  ENROLLMENT
# ══════════════════════════════════════════════════════════════════════════════

def enroll(cascade):
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          FACE LOCK  —  ENROLLMENT MODE              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  We will capture 6 poses × 25 samples = 150 total   ║")
    print("║  This gives the recognizer a rich, robust model.    ║")
    print("║                                                      ║")
    print("║  Controls during capture:                           ║")
    print("║    SPACE  →  start / advance to next pose           ║")
    print("║    Q      →  quit                                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open camera {CAMERA_INDEX}")

    all_samples = []
    total_needed = len(ENROLLMENT_POSES) * SAMPLES_PER_POSE

    for pose_idx, (pose_name, pose_instruction, pose_color) in enumerate(ENROLLMENT_POSES):
        pose_samples = []
        waiting      = True   # waiting for SPACE before we start capturing

        print(f"\n  Pose {pose_idx+1}/{len(ENROLLMENT_POSES)}:  {pose_name}")
        print(f"  → {pose_instruction}")
        print(f"  → Position your face and press SPACE to begin.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            crop, rect = get_face(frame, cascade)
            captured   = len(pose_samples)
            total_so_far = len(all_samples) + captured

            if waiting:
                ready_msg = "Face in frame — press SPACE to start" if rect else "Move into frame…"
                overlay_hud(
                    frame,
                    line1=f"Pose {pose_idx+1}/{len(ENROLLMENT_POSES)}: {pose_name}",
                    line2=ready_msg,
                    color=pose_color,
                    rect=rect,
                    progress=total_so_far / total_needed,
                    progress_color=pose_color,
                )
            else:
                if crop is not None:
                    pose_samples.append(crop)

                pct = captured / SAMPLES_PER_POSE
                overlay_hud(
                    frame,
                    line1=f"Capturing {pose_name}: {captured}/{SAMPLES_PER_POSE}",
                    line2=pose_instruction,
                    color=pose_color,
                    rect=rect,
                    progress=total_so_far / total_needed,
                    progress_color=pose_color,
                )

                if captured >= SAMPLES_PER_POSE:
                    print(f"  ✅  {pose_name} complete ({len(pose_samples)} samples)")
                    break

            cv2.imshow("Face Lock — Enrollment", frame)
            key = cv2.waitKey(60) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("\nEnrollment cancelled.")
                sys.exit(0)
            elif key == ord(' '):
                if waiting and rect is not None:
                    waiting = False
                elif not waiting and captured >= SAMPLES_PER_POSE:
                    break  # advance

        all_samples.extend(pose_samples)

        # Brief pause between poses so user can reposition
        pause_start = time.time()
        while time.time() - pause_start < 1.2:
            ret, frame = cap.read()
            if ret:
                crop, rect = get_face(frame, cascade)
                overlay_hud(frame, "Pose complete — get ready for next…", "",
                            (255, 255, 255), rect=rect,
                            progress=len(all_samples) / total_needed,
                            progress_color=(0, 200, 100))
                cv2.imshow("Face Lock — Enrollment", frame)
                cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n  Training on {len(all_samples)} total samples…")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=8, grid_x=8, grid_y=8
    )
    labels = np.zeros(len(all_samples), dtype=np.int32)
    recognizer.train(all_samples, labels)
    recognizer.save(FACE_MODEL_PATH)

    print(f"  ✅  Model saved → {FACE_MODEL_PATH}")
    print()
    print("  All done! Start face lock with:")
    print("      python3 face_lock.py")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  INSTALL AUTOSTART
# ══════════════════════════════════════════════════════════════════════════════

def install_autostart():
    script_path = os.path.abspath(__file__)
    desktop_dir = os.path.expanduser("~/.config/autostart")
    os.makedirs(desktop_dir, exist_ok=True)
    desktop_file = os.path.join(desktop_dir, "face_lock.desktop")
    content = f"""[Desktop Entry]
Type=Application
Name=Face Lock
Comment=Lock screen when your face is absent
Exec=python3 {script_path}
Hidden=false
X-GNOME-Autostart-enabled=true
"""
    with open(desktop_file, "w") as f:
        f.write(content)
    print(f"✅  Autostart entry created: {desktop_file}")
    print("   Face Lock will now start automatically when you log in.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WATCH LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(FACE_MODEL_PATH):
        sys.exit(
            "❌  No enrolled face found.\n"
            "    Run first:  python3 face_lock.py --enroll\n"
        )

    cascade = load_cascade()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(FACE_MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open camera {CAMERA_INDEX}")

    log.info("Face Lock started — watching for YOUR face.")
    log.info(f"Grace period: {GRACE_PERIOD}s  |  Confidence threshold: ≤{CONFIDENCE_THRESH}")

    absent_since  = None
    face_present  = False   # True only when YOUR face is currently recognised
    locked        = False
    last_check    = 0.0
    last_rect     = None
    last_status   = ("Initializing…", "", (200, 200, 200))

    def shutdown(sig=None, _=None):
        cap.release()
        cv2.destroyAllWindows()
        log.info("Face Lock stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.3)
            continue

        now = time.time()

        # ── Recognition ───────────────────────────────────────────────────
        if now - last_check >= CHECK_INTERVAL:
            last_check = now
            crop, rect = get_face(frame, cascade)
            last_rect  = rect

            if crop is not None:
                _, conf = recognizer.predict(crop)
                matched  = conf <= CONFIDENCE_THRESH

                if matched:
                    if absent_since is not None:
                        log.info(f"✅  Your face recognised  (conf {conf:.0f})")
                    absent_since = None   # clears timer immediately
                    face_present = True
                    locked       = False
                    last_status  = (
                        f"✓  Recognised  (conf {conf:.0f})",
                        "Your face detected — screen protected",
                        (0, 210, 90),
                    )
                else:
                    face_present = False
                    if absent_since is None:
                        absent_since = now
                        log.info(f"⚠  Unknown face  (conf {conf:.0f}) — countdown started")
                    last_status = (
                        f"⚠  Unknown face  (conf {conf:.0f})",
                        "Not your face — counting down",
                        (0, 130, 255),
                    )
            else:
                face_present = False
                if absent_since is None:
                    absent_since = now
                    log.info("⚠  No face detected — countdown started")
                last_status = (
                    "No face detected",
                    "Step away detected — counting down",
                    (0, 60, 220),
                )

            # Lock if grace period expired
            if not locked and absent_since and (now - absent_since) >= GRACE_PERIOD:
                lock_screen()
                locked = True

        # ── Preview ───────────────────────────────────────────────────────
        if SHOW_PREVIEW:
            display     = frame.copy()
            line1, line2, color = last_status

            if face_present:
                # YOUR face is here — instantly clear timer and show full green bar
                line1_out = line1
                progress  = 1.0
                prog_col  = (0, 210, 90)
            elif absent_since and not locked:
                # Counting down — show shrinking red bar + seconds remaining
                remaining = max(0.0, GRACE_PERIOD - (now - absent_since))
                line1_out = f"{line1}  —  locking in {remaining:.1f}s"
                progress  = remaining / GRACE_PERIOD
                prog_col  = (0, 60, 220)
            else:
                # Locked or initializing
                line1_out = line1
                progress  = 0.0
                prog_col  = (0, 60, 220)

            overlay_hud(display, line1_out, line2, color,
                        rect=last_rect,
                        progress=progress,
                        progress_color=prog_col)

            cv2.imshow("Face Lock", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                shutdown()

        time.sleep(0.025)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Face Lock — lock Ubuntu when your face is absent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 face_lock.py --enroll    # capture your face (first time)\n"
            "  python3 face_lock.py             # start watching\n"
            "  python3 face_lock.py --install   # add to login autostart\n"
            "  python3 face_lock.py --headless  # run without preview window\n"
        ),
    )
    ap.add_argument("--enroll",   action="store_true", help="Enroll your face")
    ap.add_argument("--install",  action="store_true", help="Add to GNOME autostart")
    ap.add_argument("--headless", action="store_true", help="Run without preview window")
    args = ap.parse_args()

    if args.headless:
        SHOW_PREVIEW = False

    if args.install:
        install_autostart()
    elif args.enroll:
        enroll(load_cascade())
    else:
        main()
