"""
OCR to TTS Reader
=================
Captures frames from a USB camera, runs Tesseract OCR, and reads the result
aloud via text-to-speech.

Platform support
----------------
  Windows 11  : pyttsx3 (SAPI5)   + cv2.CAP_DSHOW  + windowed display
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

MIN_BRIGHTNESS        = 70
MAX_BRIGHTNESS        = 230
MIN_VARIANCE_LAPLACIAN = 80.0
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
    pyttsx3 with SAPI5, running entirely on a single dedicated thread so the
    COM-initialised engine is never touched from another thread.
    """
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        import pyttsx3  # imported here so the engine lives on this thread
        engine = pyttsx3.init()
        engine.setProperty("rate", SPEECH_RATE_WIN)
        engine.setProperty("volume", 1.0)
        while True:
            item = self._q.get()
            if item is None:          # sentinel – shutdown
                engine.stop()
                return
            cmd, payload = item
            if cmd == "say":
                try:
                    engine.say(payload)
                    engine.runAndWait()
                except Exception:
                    pass
            elif cmd == "stop":
                try:
                    engine.stop()
                except Exception:
                    pass
            self._q.task_done()

    def speak(self, text: str):
        self._q.put(("say", text))

    def stop(self):
        # Drain the queue then inject a stop command so the engine halts the
        # current utterance as soon as the worker next reads from the queue.
        # We do NOT call engine.stop() from this thread.
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass
        self._q.put(("stop", ""))

    def shutdown(self):
        self._q.put(None)
        self._thread.join(timeout=3.0)


class _TTSBackendEspeakNG:
    """
    espeak-ng via subprocess. Each utterance is a separate process so stopping
    is simply killing the current process – no thread-safety concerns at all.
    """
    def __init__(self):
        self._lock   = threading.Lock()
        self._proc: subprocess.Popen | None = None

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
        if not text.strip():
            return
        self.stop()
        cmd = [
            "espeak-ng",
            "-v", ESPEAK_VOICE,
            "-s", str(ESPEAK_SPEED),
            "-a", str(ESPEAK_AMPLITUDE),
            text,
        ]
        with self._lock:
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                print("[TTS] espeak-ng not found. Install with: sudo apt install espeak-ng")

    def stop(self):
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

def capture_and_read(frame: np.ndarray, tts, state: dict):
    """
    Runs synchronously on the main thread.  TTS calls are non-blocking
    (queued / subprocess), so the camera loop is not frozen waiting for audio.
    """
    tts.stop()
    tts.speak("Image captured. Checking image quality.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ok, msg = check_brightness(gray)
    if not ok:
        tts.speak(msg)
        return

    ok, msg = check_blur(gray)
    if not ok:
        tts.speak(msg)
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    try:
        save_capture(frame, f"capture_{timestamp}_original.png")
    except IOError as exc:
        print(f"[WARN] Could not save original: {exc}")
        tts.speak("Warning: failed to save original image.")

    processed, thresh = preprocess_image(frame)

    if not check_text_density(thresh):
        tts.speak("No readable text found.")
        return

    try:
        save_capture(processed, f"capture_{timestamp}_processed.png")
    except IOError as exc:
        print(f"[WARN] Could not save processed: {exc}")

    tts.speak("Reading started.")

    raw_text = extract_text(processed)
    text     = clean_text(raw_text)

    if len(text) < MIN_TEXT_LENGTH:
        tts.speak("No readable text found.")
        return

    # Thread-safe write: GIL makes dict assignment atomic in CPython, but we
    # use a lock to be correct regardless of implementation.
    with state["lock"]:
        state["last_text"] = text

    print("\n--- OCR TEXT ---")
    print(text)
    print("----------------\n")

    tts.speak(text)
    tts.speak("Reading finished. Turn to the next page when ready.")


# ─────────────────────────────────────────────
# Headless input: stdin + optional GPIO
# ─────────────────────────────────────────────

def _stdin_input_thread(cmd_queue: queue.Queue):
    """Reads single-character commands from stdin (headless mode)."""
    print("Headless controls: SPACE/c=capture  r=repeat  s=stop  t=test  q=quit")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:        # EOF
                cmd_queue.put("q")
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
            cmd_queue.put("q")
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
        "lock":      threading.Lock(),
    }

    # ── Windowed vs headless ───────────────────
    cmd_queue: queue.Queue = queue.Queue()

    if HEADLESS:
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
                        capture_and_read(frame, tts, state)
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


if __name__ == "__main__":
    main()