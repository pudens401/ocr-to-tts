"""
OCR to TTS Reader
=================
Captures frames from a USB camera, runs Tesseract OCR, and reads the result
aloud via text-to-speech.

Platform support
----------------
  Windows 11  : PowerShell SAPI5  + cv2.CAP_DSHOW  + windowed display
  Raspberry Pi 5 (Pi OS Bookworm) : espeak-ng (subprocess) + cv2.CAP_V4L2 + headless

Controls (windowed mode)
------------------------
  SPACE  – capture and read
  R      – repeat last text
  S      – stop speech
  T      – test speech
  Q/ESC  – quit

Headless mode (Pi without display)
-----------------------------------
  Controls are read from stdin in a background thread.
  Alternatively, wire a momentary push-button to GPIO 17 (see GPIO_PIN below).
  If RPi.GPIO is not installed the GPIO path is silently skipped.
"""

import os
# If the Qt platform plugin (e.g. "wayland") is missing, prefer X11's
# `xcb` backend to avoid the Qt plugin error printed by OpenCV's Qt
# build. This must be set before importing `cv2` so Qt picks it up.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import pytesseract
import numpy as np
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time

# ─────────────────────────────────────────────
# Platform detection
# ─────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# Headless when no display server is present (standard Pi OS Bookworm headless)
_has_display = bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)
HEADLESS = IS_LINUX and not _has_display

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CAMERA_INDEX          = 0
CAPTURE_DIR           = "captures"
SPEECH_RATE_WIN       = 150          # words-per-minute for SAPI5
ESPEAK_VOICE          = "en"         # espeak-ng voice tag
ESPEAK_SPEED          = 150          # espeak-ng words-per-minute
ESPEAK_AMPLITUDE      = 100          # espeak-ng amplitude 0-200

# Page edge detection / auto capture
AUTO_CAPTURE_ENABLED  = True
PAGE_STABLE_FRAMES    = 8            # consecutive frames before auto-capture
AUTO_CAPTURE_COOLDOWN = 2.5          # seconds between auto-captures
DETECT_HEIGHT         = 500          # resize height for edge detection
MIN_PAGE_AREA_RATIO   = 0.20         # min contour area vs frame area
MAX_QUAD_SHIFT        = 20.0         # px mean corner movement to be “stable”

MIN_BRIGHTNESS        = 70
MAX_BRIGHTNESS        = 230
MIN_VARIANCE_LAPLACIAN = 100.0
MIN_TEXT_DENSITY      = 0.01
MIN_TEXT_LENGTH       = 30

FRAME_WARMUP_COUNT    = 30           # frames to discard before accepting input
NO_FRAME_LOG_INTERVAL = 5.0          # seconds between "camera not responding" messages

TESSERACT_PATH_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional GPIO button pin (BCM numbering). Set to None to disable.
GPIO_PIN = 17

# ─────────────────────────────────────────────
# TTS backend
# ─────────────────────────────────────────────

class _TTSBackendWindows:
    """
    Windows TTS via a single persistent PowerShell process.

    Why persistent?
    ---------------
    Spawning one powershell.exe per utterance costs 1.5–3 s of CLR + SAPI5
    cold-start time before the first syllable.  Short phrases (< 1 s of audio)
    are inaudible because the startup silence swallows the perceived onset.

    Design
    ------
    One PowerShell process is started at __init__ and kept alive.  It runs a
    loop that reads lines from its stdin pipe:
      - ordinary text  → $s.Speak(text)  [synchronous, blocks until done]
      - "__STOP__"     → $s.SpeakAsyncCancelAll()  [clears current speech]
      - "__EXIT__"     → exits the loop

    speak() writes a line to the pipe.
    stop()  drains the Python-side queue, then writes "__STOP__" to the pipe.
    No per-utterance process spawning; no COM threading issues.
    """

    # SAPI5 rate: -10 (slowest) to 10 (fastest); 0 ≈ 150 wpm
    @staticmethod
    def _wpm_to_sapi_rate(wpm: int) -> int:
        return max(-10, min(10, round((wpm - 150) / 25)))

    # PowerShell script run once for the lifetime of the program.
    # Uses synchronous $s.Speak() so each line blocks until audio finishes —
    # the next stdin.ReadLine() is not reached until speaking is done.
    _PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate   = {rate}
$s.Volume = 100
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
while ($true) {{
    $line = [Console]::In.ReadLine()
    if ($null -eq $line)            {{ break }}
    if ($line -eq '__EXIT__')       {{ break }}
    if ($line -eq '__STOP__')       {{ $s.SpeakAsyncCancelAll() }}
    elseif ($line.Trim() -ne '')    {{ $s.Speak($line) }}
}}
"""

    def __init__(self):
        self._rate      = self._wpm_to_sapi_rate(SPEECH_RATE_WIN)
        self._lock      = threading.Lock()   # guards _proc and _stopped
        self._stopped   = False              # set during a stop() call
        self._proc      = self._launch()

    def _launch(self) -> subprocess.Popen:
        script = self._PS_SCRIPT.format(rate=self._rate)
        proc   = subprocess.Popen(
            [
                "powershell", "-NoProfile", "-NonInteractive",
                "-WindowStyle", "Hidden",
                "-Command", script,
            ],
            stdin  = subprocess.PIPE,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            text   = True,
            encoding = "utf-8",
        )
        return proc

    def _writeline(self, line: str):
        """Write one line to the PowerShell stdin pipe, restarting if needed."""
        with self._lock:
            proc = self._proc
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (OSError, BrokenPipeError):
            # Process died unexpectedly — restart it and retry once
            print("[TTS] PowerShell process died; restarting.")
            with self._lock:
                self._proc = self._launch()
                proc = self._proc
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except Exception as exc:
                print(f"[TTS] Could not restart PowerShell: {exc}")

    def speak(self, text: str):
        if not text.strip():
            return
        # Escape any bare newlines so the pipe sees a single logical line
        safe = text.replace("\r", " ").replace("\n", " ")
        self._writeline(safe)

    def stop(self):
        """Cancel current speech and clear the PS-side queue via __STOP__."""
        self._writeline("__STOP__")

    def shutdown(self):
        try:
            self._writeline("__EXIT__")
            with self._lock:
                proc = self._proc
            proc.stdin.close()
            proc.wait(timeout=3.0)
        except Exception:
            with self._lock:
                self._proc.kill()


class _TTSBackendEspeakNG:
    """
    espeak-ng via subprocess with a sequential worker queue.
    speak() enqueues; stop() is an explicit user action only — never called
    inside speak() — so consecutive speak() calls play in order, not clobber.
    """
    def __init__(self):
        self._lock   = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._q      = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _espeak_available(self) -> bool:
        try:
            subprocess.run(
                ["espeak-ng", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def speak(self, text: str):
        """Enqueue text. Never kills a running utterance — use stop() for that."""
        if not text.strip():
            return
        self._q.put(text)

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            cmd = [
                "espeak-ng",
                "-v", ESPEAK_VOICE,
                "-s", str(ESPEAK_SPEED),
                "-a", str(ESPEAK_AMPLITUDE),
                item,
            ]
            try:
                with self._lock:
                    self._proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                self._proc.wait()
            except FileNotFoundError:
                print("[TTS] espeak-ng not found. Install with: sudo apt install espeak-ng")
            except Exception as exc:
                print(f"[TTS] espeak-ng error: {exc}")
            finally:
                with self._lock:
                    self._proc = None

    def stop(self):
        """Drain queue then kill current process. Explicit user action only."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None

    def shutdown(self):
        self.stop()
        self._q.put(None)
        self._worker_thread.join(timeout=3.0)


def _build_tts():
    if IS_WINDOWS:
        return _TTSBackendWindows()
    # Linux / Pi
    backend = _TTSBackendEspeakNG()
    if not backend._espeak_available():
        print(
            "[TTS] espeak-ng not found.\n"
            "      Install: sudo apt install espeak-ng\n"
            "      Continuing without speech."
        )
    return backend


# ─────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        # Fallback: let OpenCV pick the backend
        cap = cv2.VideoCapture(CAMERA_INDEX)
    return cap


# ─────────────────────────────────────────────
# Image quality checks
# ─────────────────────────────────────────────

def check_brightness(gray: np.ndarray) -> tuple[bool, str]:
    brightness = float(np.mean(gray))
    if brightness < MIN_BRIGHTNESS:
        return False, "Image too dark. Add more light."
    if brightness > MAX_BRIGHTNESS:
        return False, "Image too bright. Reduce glare or light."
    return True, ""


def check_blur(gray: np.ndarray) -> tuple[bool, str]:
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance < MIN_VARIANCE_LAPLACIAN:
        return False, "Image is blurry. Hold the camera steady or move it closer."
    return True, ""


def check_text_density(thresh: np.ndarray) -> bool:
    """Receives an already-thresholded image to avoid recomputing it."""
    black_pixels = int(np.sum(thresh == 0))
    density = black_pixels / float(thresh.size)
    return density >= MIN_TEXT_DENSITY


# ─────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────

def deskew_image(gray: np.ndarray) -> np.ndarray:
    """
    Correct page tilt using minAreaRect.
    coords are (row, col) from np.where → passed to minAreaRect as points
    which expects (x, y) = (col, row), so we swap axes explicitly.
    """
    rows, cols = np.where(gray < 128)
    if rows.size == 0:
        return gray
    # Build (x, y) = (col, row) array as float32 for minAreaRect
    points = np.column_stack((cols, rows)).astype(np.float32)
    angle  = cv2.minAreaRect(points)[-1]
    # minAreaRect returns angles in (-90, 0]; normalise to a small tilt
    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle          # already negative; keep sign for warpAffine
    (h, w)  = gray.shape[:2]
    center  = (w // 2, h // 2)
    matrix  = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (processed_for_ocr, threshold_for_density_check).
    Using CLAHE instead of equalizeHist to avoid noise amplification.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE: local contrast enhancement, much gentler than global equalizeHist
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    gray  = deskew_image(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )
    return thresh, thresh   # same image used for both OCR and density check


# ─────────────────────────────────────────────
# Page edge detection + perspective transform
# ─────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def detect_page_quad(frame: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    scale = DETECT_HEIGHT / float(h)
    resized = cv2.resize(frame, (int(w * scale), DETECT_HEIGHT))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(resized.shape[0] * resized.shape[1])
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        area = cv2.contourArea(cnt)
        if area / frame_area < MIN_PAGE_AREA_RATIO:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32")
            quad /= scale
            return quad
    return None


# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────

def extract_text(image: np.ndarray) -> str:
    # psm 3 = fully automatic page segmentation (handles columns, mixed layouts)
    config = "--oem 1 --psm 3"
    return pytesseract.image_to_string(image, lang="eng", config=config)


def clean_text(text: str) -> str:
    cleaned = text.replace("\x0c", " ")
    cleaned = re.sub(r"[\t\r]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


# ─────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────

def save_capture(image: np.ndarray, filename: str) -> str:
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, filename)
    ok   = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"cv2.imwrite failed for: {path}")
    return path


# ─────────────────────────────────────────────
# Core capture-and-read routine
# ─────────────────────────────────────────────

def _ocr_worker(frame: np.ndarray, tts, state: dict, page_quad: np.ndarray | None):
    """
    Runs on a dedicated thread so the main loop (and cv2.waitKey) keeps
    ticking during the 2-5 second Tesseract call.
    """
    tts.speak("Image captured. Checking image quality.")

    if page_quad is not None:
        try:
            frame = _four_point_transform(frame, page_quad)
        except Exception:
            pass

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    timestamp = time.strftime("%Y%m%d_%H%M%S")

    try:
        save_capture(frame, f"capture_{timestamp}_original.png")
    except IOError as exc:
        print(f"[WARN] Could not save original: {exc}")
        tts.speak("Warning: failed to save original image.")

    processed, thresh = preprocess_image(frame)

    #if not check_text_density(thresh):
        #tts.speak("No readable text found.")
       # with state["lock"]:
         #   state["ocr_busy"] = False
        #return

    try:
        save_capture(processed, f"capture_{timestamp}_processed.png")
    except IOError as exc:
        print(f"[WARN] Could not save processed: {exc}")

    tts.speak("Reading started.")

    raw_text = extract_text(processed)
    text     = clean_text(raw_text)

    with state["lock"]:
        state["last_text"] = text
        state["ocr_busy"]  = False

    print("\n--- OCR TEXT ---")
    print(text)
    print("----------------\n")

    tts.speak(text)
    tts.speak("Reading finished. Turn to the next page when ready.")


def capture_and_read(frame: np.ndarray, tts, state: dict, page_quad: np.ndarray | None = None):
    """
    Entry point called from the main loop on the main thread.
    Guards against overlapping captures with ocr_busy flag, then hands off
    to a background thread so cv2.waitKey() keeps running during OCR.
    """
    with state["lock"]:
        if state["ocr_busy"]:
            tts.speak("Still processing. Please wait.")
            return
        state["ocr_busy"] = True

    t = threading.Thread(
        target=_ocr_worker,
        args=(frame.copy(), tts, state, page_quad),   # copy frame — camera will overwrite it
        daemon=True,
    )
    t.start()


# ─────────────────────────────────────────────
# Headless input: stdin + optional GPIO
# ─────────────────────────────────────────────

def _stdin_input_thread(cmd_queue: queue.Queue):
    """Reads single-character commands from stdin (headless mode)."""
    print("Headless controls: SPACE/c=capture  r=repeat  s=stop  t=test  q=quit")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:        # EOF (e.g., systemd service with no stdin)
                return
            ch = line.strip().lower()
            if ch in ("", "c", " "):
                cmd_queue.put("capture")
            elif ch == "r":
                cmd_queue.put("repeat")
            elif ch == "s":
                cmd_queue.put("stop")
            elif ch == "t":
                cmd_queue.put("test")
            elif ch == "q":
                cmd_queue.put("q")
                return
        except (EOFError, OSError):
            return


def _try_setup_gpio(cmd_queue: queue.Queue):
    """
    Optionally wire a push-button to GPIO_PIN (BCM).  Silently skipped if
    RPi.GPIO is not installed or GPIO_PIN is None.
    """
    if GPIO_PIN is None:
        return
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        def _callback(channel):
            cmd_queue.put("capture")

        GPIO.add_event_detect(
            GPIO_PIN,
            GPIO.FALLING,
            callback=_callback,
            bouncetime=500,
        )
        print(f"[GPIO] Button on BCM pin {GPIO_PIN} active.")
    except ImportError:
        pass   # RPi.GPIO not available – stdin-only is fine
    except Exception as exc:
        print(f"[GPIO] Setup failed: {exc}")


# ─────────────────────────────────────────────
# Tesseract setup
# ─────────────────────────────────────────────

def setup_tesseract():
    if IS_WINDOWS:
        if not os.path.exists(TESSERACT_PATH_WINDOWS):
            raise FileNotFoundError(
                f"Tesseract not found at {TESSERACT_PATH_WINDOWS}.\n"
                "Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_WINDOWS
    else:
        # On Pi, tesseract should be on PATH after: sudo apt install tesseract-ocr
        try:
            subprocess.run(
                ["tesseract", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise FileNotFoundError(
                "Tesseract not found.\n"
                "Install: sudo apt install tesseract-ocr tesseract-ocr-eng"
            )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # ── Tesseract ──────────────────────────────
    try:
        setup_tesseract()
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    # ── TTS ────────────────────────────────────
    tts = _build_tts()

    # ── Camera ─────────────────────────────────
    cap = open_camera()
    if not cap.isOpened():
        tts.speak("Camera not found.")
        print("[ERROR] Could not open camera.")
        tts.shutdown()
        sys.exit(1)

    # ── Shared state (protected by a lock) ─────
    state: dict = {
        "last_text": "",
        "ocr_busy":  False,
        "lock":      threading.Lock(),
        "last_page_quad": None,
        "stable_frames": 0,
        "last_auto_capture": 0.0,
    }

    # ── Windowed vs headless ───────────────────
    cmd_queue: queue.Queue = queue.Queue()

    if HEADLESS:
        # Under systemd services, stdin is typically not attached to a TTY and
        # readline() returns EOF immediately. Only start the stdin thread when
        # stdin is interactive so the service doesn't exit on EOF.
        if sys.stdin.isatty():
            stdin_thread = threading.Thread(
                target=_stdin_input_thread, args=(cmd_queue,), daemon=True
            )
            stdin_thread.start()
        _try_setup_gpio(cmd_queue)
    else:
        cv2.namedWindow("OCR to TTS Reader", cv2.WINDOW_NORMAL)

    tts.speak("OCR reader started.")
    tts.speak("Place the book under the camera.")

    print("Platform :", "Windows" if IS_WINDOWS else "Linux/Pi")
    print("Headless  :", HEADLESS)
    if not HEADLESS:
        print("Controls  : SPACE=Read  R=Repeat  S=Stop  T=Test  Q/ESC=Quit")

    # ── Frame warm-up ──────────────────────────
    warmup_remaining = FRAME_WARMUP_COUNT
    last_no_frame_log = 0.0

    try:
        while True:
            ret, frame = cap.read()

            # ── No frame handling ───────────────
            if not ret:
                now = time.monotonic()
                if now - last_no_frame_log >= NO_FRAME_LOG_INTERVAL:
                    print("[WARN] Camera not responding.")
                    tts.speak("Camera not responding.")
                    last_no_frame_log = now

                if not HEADLESS:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        blank, "Waiting for camera...",
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2,
                    )
                    cv2.imshow("OCR to TTS Reader", blank)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.1)

                # Check headless command queue (e.g. quit)
                _drain_cmd_queue_for_quit(cmd_queue)
                continue

            # ── Warm-up: discard early frames ───
            if warmup_remaining > 0:
                warmup_remaining -= 1
                if not HEADLESS:
                    cv2.imshow("OCR to TTS Reader", frame)
                    cv2.waitKey(1)
                continue

            # ── Windowed mode: display + keyboard
            if not HEADLESS:
                display = frame.copy()
                cv2.putText(
                    display,
                    "SPACE: Read | R: Repeat | S: Stop | Q: Quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 220, 0), 2,
                )

                with state["lock"]:
                    page_quad = state["last_page_quad"]
                    stable_frames = state["stable_frames"]

                if page_quad is not None:
                    pts = page_quad.astype(int)
                    cv2.polylines(display, [pts], True, (0, 165, 255), 2)
                    if AUTO_CAPTURE_ENABLED and stable_frames >= PAGE_STABLE_FRAMES:
                        cv2.putText(
                            display,
                            "Auto capture ready",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 165, 255), 2,
                        )

                cv2.imshow("OCR to TTS Reader", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    tts.speak("Closing reader.")
                    break
                elif key == ord(" "):
                    cmd_queue.put("capture")
                elif key in (ord("r"), ord("R")):
                    cmd_queue.put("repeat")
                elif key in (ord("s"), ord("S")):
                    cmd_queue.put("stop")
                elif key in (ord("t"), ord("T")):
                    cmd_queue.put("test")

            # ── Process commands ────────────────
            try:
                while True:
                    cmd = cmd_queue.get_nowait()
                    if cmd == "capture":
                        with state["lock"]:
                            quad = state["last_page_quad"]
                        capture_and_read(frame, tts, state, quad)
                    elif cmd == "repeat":
                        with state["lock"]:
                            txt = state["last_text"]
                        if txt:
                            tts.speak("Repeating last page.")
                            tts.speak(txt)
                        else:
                            tts.speak("No previous text to repeat.")
                    elif cmd == "stop":
                        tts.stop()
                        tts.speak("Speech stopped.")
                    elif cmd == "test":
                        tts.speak("Test message. Text to speech is working.")
                    elif cmd == "q":
                        tts.speak("Closing reader.")
                        raise _QuitSignal()
            except queue.Empty:
                pass

            # ── Page edge detection + auto capture ─────────
            if AUTO_CAPTURE_ENABLED or not HEADLESS:
                page_quad = detect_page_quad(frame)
                with state["lock"]:
                    prev_quad = state["last_page_quad"]
                    if page_quad is not None:
                        if prev_quad is not None and _quad_mean_shift(prev_quad, page_quad) <= MAX_QUAD_SHIFT:
                            state["stable_frames"] += 1
                        else:
                            state["stable_frames"] = 1
                        state["last_page_quad"] = page_quad
                    else:
                        state["stable_frames"] = 0
                        state["last_page_quad"] = None

                    stable_frames = state["stable_frames"]
                    last_auto = state["last_auto_capture"]
                    ocr_busy = state["ocr_busy"]

                if (
                    AUTO_CAPTURE_ENABLED
                    and page_quad is not None
                    and not ocr_busy
                    and stable_frames >= PAGE_STABLE_FRAMES
                    and (time.monotonic() - last_auto) >= AUTO_CAPTURE_COOLDOWN
                ):
                    capture_and_read(frame, tts, state, page_quad)
                    with state["lock"]:
                        state["stable_frames"] = 0
                        state["last_auto_capture"] = time.monotonic()

    except _QuitSignal:
        pass
    except KeyboardInterrupt:
        tts.speak("Interrupted. Closing.")
    finally:
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        tts.shutdown()
        # Clean up GPIO if it was initialised
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except Exception:
            pass
        print("Reader closed.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

class _QuitSignal(Exception):
    """Used to break out of nested loops cleanly."""


def _drain_cmd_queue_for_quit(cmd_queue: queue.Queue):
    """Check if a quit command arrived while we are in the no-frame branch."""
    try:
        cmd = cmd_queue.get_nowait()
        if cmd == "q":
            raise _QuitSignal()
        # Put it back if it wasn't a quit
        cmd_queue.put(cmd)
    except queue.Empty:
        pass


def _quad_mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


if __name__ == "__main__":
    main()