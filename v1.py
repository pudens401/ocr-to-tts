import cv2
import pytesseract
import pyttsx3
import numpy as np
import platform
import os
import threading
import time
import re

# -----------------------------
# Config
# -----------------------------
CAMERA_INDEX = 0
CAPTURE_DIR = "captures"
TESSERACT_PATH_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SPEECH_RATE = 150
SPEECH_VOLUME = 1.0
DIRECT_SPEECH = False
MIN_BRIGHTNESS = 70
MAX_BRIGHTNESS = 230
MIN_VARIANCE_LAPLACIAN = 80.0
MIN_TEXT_DENSITY = 0.01
MIN_TEXT_LENGTH = 30
FRAME_WARMUP_COUNT = 10
NO_FRAME_RETRY_LIMIT = 120

engine = None
speaking_thread = None
last_text = ""


def setup_tesseract():
    if platform.system() == "Windows":
        if os.path.exists(TESSERACT_PATH_WINDOWS):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_WINDOWS
        else:
            raise FileNotFoundError(
                "Tesseract not found. Install it or update TESSERACT_PATH_WINDOWS."
            )


def setup_tts():
    global engine
    engine = pyttsx3.init()
    engine.setProperty("rate", SPEECH_RATE)
    engine.setProperty("volume", SPEECH_VOLUME)


def open_camera():
    return cv2.VideoCapture(CAMERA_INDEX)


def speak_sequence(messages):
    global speaking_thread

    if engine is None:
        return

    filtered = [msg.strip() for msg in messages if msg and msg.strip()]
    if not filtered:
        filtered = ["No readable text found."]

    stop_speech()

    def run_speech(msgs):
        try:
            for msg in msgs:
                engine.say(msg)
            engine.runAndWait()
        except Exception:
            pass

    if DIRECT_SPEECH:
        run_speech(filtered)
        return

    speaking_thread = threading.Thread(target=run_speech, args=(filtered,), daemon=True)
    speaking_thread.start()


def speak(text):
    speak_sequence([text])


def stop_speech():
    try:
        if engine is not None:
            engine.stop()
    except Exception:
        pass
    if speaking_thread and speaking_thread.is_alive():
        speaking_thread.join(timeout=0.5)


def check_brightness(gray):
    brightness = float(np.mean(gray))
    if brightness < MIN_BRIGHTNESS:
        return False, "Image too dark. Add more light."
    if brightness > MAX_BRIGHTNESS:
        return False, "Image too bright. Reduce glare or light."
    return True, ""


def check_blur(gray):
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance < MIN_VARIANCE_LAPLACIAN:
        return False, "Image is blurry. Adjust the camera or page."
    return True, ""


def check_text_density(gray):
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    black_pixels = np.sum(thresh == 0)
    density = black_pixels / float(thresh.size)
    return density >= MIN_TEXT_DENSITY


def deskew_image(gray):
    coords = np.column_stack(np.where(gray < 128))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gray = deskew_image(gray)

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    return processed


def extract_text(image):
    config = "--oem 1 --psm 6"
    text = pytesseract.image_to_string(image, lang="eng", config=config)
    return text


def clean_text(text):
    cleaned = text.replace("\x0c", " ")
    cleaned = re.sub(r"[\t\r]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def save_capture(image, filename):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, filename)
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError("Failed to save capture.")
    return path


def capture_and_read(frame):
    global last_text

    stop_speech()
    speak_sequence(["Image captured.", "Checking image quality."])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ok_brightness, brightness_message = check_brightness(gray)
    if not ok_brightness:
        speak(brightness_message)
        return

    ok_blur, blur_message = check_blur(gray)
    if not ok_blur:
        speak(blur_message)
        return

    if not check_text_density(gray):
        speak("No readable text found.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    try:
        save_capture(frame, f"capture_{timestamp}_original.png")
    except Exception:
        speak("Failed to save capture.")

    processed = preprocess_image(frame)

    try:
        save_capture(processed, f"capture_{timestamp}_processed.png")
    except Exception:
        speak("Failed to save capture.")

    speak("Reading started.")

    raw_text = extract_text(processed)
    text = clean_text(raw_text)

    if len(text) < MIN_TEXT_LENGTH:
        speak("No readable text found.")
        return

    last_text = text

    print("\n--- OCR TEXT ---")
    print(text)
    print("----------------\n")

    speak_sequence([
        text,
        "Reading finished.",
        "Turn to the next page when ready.",
    ])


def main():
    global last_text

    try:
        setup_tesseract()
    except Exception as exc:
        print(str(exc))
        return

    try:
        setup_tts()
    except Exception:
        print("Failed to initialize text to speech.")
        return

    cap = open_camera()

    if not cap.isOpened():
        print("Camera not found.")
        speak("Camera not found.")
        return

    cv2.namedWindow("OCR to TTS Reader", cv2.WINDOW_NORMAL)

    speak("OCR reader started.")
    speak("Place the book under the camera.")

    print("Controls:")
    print("SPACE = Capture and read")
    print("R     = Repeat last text")
    print("S     = Stop speech")
    print("T     = Test speech")
    print("Q     = Quit")

    warmup = FRAME_WARMUP_COUNT
    no_frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            no_frame_count += 1
            if warmup > 0:
                warmup -= 1
            if no_frame_count % NO_FRAME_RETRY_LIMIT == 0:
                print("Failed to read camera.")
                speak("Failed to read camera.")
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Waiting for camera frames...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("OCR to TTS Reader", blank)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                speak("Closing reader.")
                break
            if key in (ord("t"), ord("T")):
                speak("Test message. Text to speech is working.")
            continue
        no_frame_count = 0

        cv2.putText(
            frame,
            "SPACE: Read | R: Repeat | S: Stop | Q: Quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("OCR to TTS Reader", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            speak("Closing reader.")
            break

        elif key == ord(" "):
            capture_and_read(frame)

        elif key == ord("r"):
            if last_text:
                speak("Repeating last page.")
                speak(last_text)
            else:
                speak("There is no previous text to repeat.")

        elif key in (ord("s"), ord("S")):
            stop_speech()
            speak("Speech stopped.")

        elif key in (ord("t"), ord("T")):
            speak("Test message. Text to speech is working.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()