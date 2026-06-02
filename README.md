# OCR to TTS Reader

Capture pages from a USB camera, run Tesseract OCR, and read aloud via text-to-speech.

## Features
- Manual capture (SPACE) with repeat/stop/test controls
- Automatic page capture when page edges are stable
- Works on Windows (SAPI5) and Linux/Raspberry Pi (espeak-ng)

## Requirements
- Python 3.10+
- Tesseract OCR
- OpenCV

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install system packages (Linux/Raspberry Pi):

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng espeak-ng
```

## Run

```bash
python v1.py
```

## Auto-capture settings
Adjust in `v1.py` if needed:
- `AUTO_CAPTURE_ENABLED`
- `PAGE_STABLE_FRAMES`
- `AUTO_CAPTURE_COOLDOWN`
- `MIN_PAGE_AREA_RATIO`
- `MAX_QUAD_SHIFT`
