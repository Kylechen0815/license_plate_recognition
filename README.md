# Real-Time License Plate Recognition (ALPR) Pipeline

An end-to-end computer vision and image processing pipeline built in Python to detect, dewarp, and recognize vehicle license plates from video streams using OpenCV and Tesseract OCR.

---

## Project Overview
This project processes video streams to automatically identify and extract license plate numbers (specifically optimized for standard 7-character license plates: 3 letters + 4 digits). Instead of relying on heavy deep-learning object detectors, it employs robust traditional image processing algorithms—achieving high efficiency and explainability with low computational overhead.

---

## Visual Pipeline Results

### 1. Contrast Enhancement and Edge Detection
Adaptive histogram equalization on the luminance channel followed by edge extraction.
<img width="861" height="645" alt="Canny and Contour Processing" src="https://github.com/user-attachments/assets/1d916330-bfe5-445e-8ecd-b2057e23cb3d" />

### 2. Plate Localization and Perspective Transform
Four-point contour approximation and geometric perspective correction to rectify skewed plates.
<img width="1382" height="379" alt="Perspective Rectification" src="https://github.com/user-attachments/assets/2e00a4f6-06d1-4d78-bf61-eb58e40e6e3b" />

### 3. Thresholding and Character Segmentation
Otsu's binarization and morphological filtering to isolate high-contrast character regions.
<img width="1350" height="799" alt="Thresholding and Binary Masking" src="https://github.com/user-attachments/assets/c03a51c1-50e6-42d6-b96e-a970c0056ff2" />

### 4. Dynamic Frame Tracking and Real-Time Output
Continuous detection and OCR recognition results superimposed on video frames.
<img width="1297" height="779" alt="Real-time Detection Result 1" src="https://github.com/user-attachments/assets/6057a5e8-4851-4ed6-8624-48d67b42f94d" />
<img width="1301" height="773" alt="Real-time Detection Result 2" src="https://github.com/user-attachments/assets/0c9de3b8-221d-4b01-83f1-ba704e0ea94c" />
<img width="1318" height="724" alt="Real-time Detection Result 3" src="https://github.com/user-attachments/assets/12d30f0f-d1ff-4ce6-8c4f-815016f496f0" />

---

## Technical Workflow

1. **Illumination Normalization (HSV + CLAHE)**:
   - Converts frames from BGR to HSV color space.
   - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) with `clipLimit=4.0` to the Value (V) channel to mitigate varying outdoor lighting conditions and reflections from white car bodies.

2. **Edge and Contour Extraction**:
   - Applies Gaussian blur smoothing and Canny Edge Detection (thresholds: 50, 150).
   - Uses morphological dilation (`cv2.dilate`) to bridge small gradient gaps in the plate boundary.

3. **Quadrilateral Localization and Perspective Rectification**:
   - Extracts external contours and approximates polygons using `cv2.approxPolyDP`.
   - Filters candidate contours based on convexity, aspect ratio, and bounding box dimensions.
   - Applies `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to warp angled plates into a canonical front-facing 200x80 bounding image.

4. **Adaptive Binarization**:
   - Converts rectified plates to grayscale and segments text using Otsu's Thresholding.
   - Applies upper border cropping to remove screw holes and plate border artifacts.

5. **OCR Recognition and Format Validation**:
   - Feeds the binarized image to Tesseract OCR configured with single-line segmentation (`--psm 7`) and an alphanumeric character whitelist (`A-Z, 0-9`).
   - Validates candidate strings against the target syntax (length of 7 characters: 3 alphabetic letters followed by 4 numeric digits).

---

## Tech Stack and Dependencies

- **Language**: Python 3.8+
- **Core Libraries**:
  - `opencv-python` (Computer Vision and Geometric Transformations)
  - `pytesseract` (Optical Character Recognition)
  - `numpy` (Numerical Computations and Array Manipulation)
  - `tkinter` (GUI directory selection)
- **External Dependency**:
  - Tesseract OCR Engine

---

## Installation and Usage

### 1. Prerequisites
Install the Tesseract OCR engine on your system:

- **Ubuntu / Debian**:
  ```bash
  sudo apt-get install tesseract-ocr


<img width="861" height="645" alt="屏幕截图 2024-12-28 200837" src="https://github.com/user-attachments/assets/1d916330-bfe5-445e-8ecd-b2057e23cb3d" />

<img width="1382" height="379" alt="屏幕截图 2024-12-29 170957" src="https://github.com/user-attachments/assets/2e00a4f6-06d1-4d78-bf61-eb58e40e6e3b" />

<img width="1350" height="799" alt="屏幕截图 2026-01-09 020716" src="https://github.com/user-attachments/assets/c03a51c1-50e6-42d6-b96e-a970c0056ff2" />

<img width="1297" height="779" alt="屏幕截图 2026-01-09 020949" src="https://github.com/user-attachments/assets/6057a5e8-4851-4ed6-8624-48d67b42f94d" />

<img width="1301" height="773" alt="屏幕截图 2026-01-09 020734" src="https://github.com/user-attachments/assets/0c9de3b8-221d-4b01-83f1-ba704e0ea94c" />

<img width="1318" height="724" alt="屏幕截图 2026-01-09 020749" src="https://github.com/user-attachments/assets/12d30f0f-d1ff-4ce6-8c4f-815016f496f0" />


