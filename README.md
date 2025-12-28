# AI Image Analyzer

A modular computer vision pipeline that analyzes an image to:
- Detect common objects (YOLOv8)
- Detect faces with privacy-safe tokens (no identity recognition)
- Generate a natural language scene description (BLIP)

Output is a clean JSON that can be consumed by an app/robot.

---

## Features
- ✅ Object detection (YOLOv8 via Ultralytics)
- ✅ Face detection + per-image tokens (`Face_1`, `Face_2`, …)
- ✅ Scene captioning (BLIP)
- ✅ Modular pipeline design (easy to swap models)

---

## Requirements
- **Python 3.11 (recommended)**
- Works on Windows/Linux/macOS
- NVIDIA GPU is recommended (CUDA), but CPU works too

---

## Installation

### 1) Clone
```bash
git clone https://github.com/mohammednafees1007-hub/ai-image-analyzer.git
cd ai-image-analyzer
