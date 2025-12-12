# AI Image Analyzer - Interface Spec (v1)

## Input
- image: JPG / PNG

## Output (JSON)
```json
{
  "objects": [
    { "label": "string", "confidence": 0.0 }
  ],
  "faces": [
    { "token": "Face_1", "bbox": [x1, y1, x2, y2], "confidence": 0.0 }
  ],
  "scene_description": "2 short lines max"
}
