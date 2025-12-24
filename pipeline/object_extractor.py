from ultralytics import YOLO
import numpy as np

_model = YOLO("yolov8n.pt")  # fast starter

# 1) Keep only what your app cares about (edit this list anytime)
ALLOWED_CLASSES = {
    "person",
    "bottle",
    "cup",
    "chair",
    "laptop",
    "cell phone",
    "book",
    "backpack",
    "handbag",
    "sports ball",
    "mouse",
    "keyboard",
    "tv",
}

# 2) Remap COCO names -> your app-friendly names
LABEL_MAP = {
    "cell phone": "phone",
    "sports ball": "ball",
    "tv": "screen",
}

def detect_objects(image, conf_threshold: float = 0.45):
    """
    Returns list:
      [{ "label": str, "confidence": float }]
    - filters to allowed classes
    - remaps labels
    - de-duplicates by keeping highest confidence per label
    """
    img = np.array(image)  # PIL -> numpy RGB
    results = _model.predict(img, verbose=False)[0]

    best_by_label: dict[str, float] = {}

    if results.boxes is None:
        return []

    for b in results.boxes:
        conf = float(b.conf.item())
        if conf < conf_threshold:
            continue

        cls_id = int(b.cls.item())
        coco_label = results.names[cls_id]

        # filter
        if coco_label not in ALLOWED_CLASSES:
            continue

        # remap
        label = LABEL_MAP.get(coco_label, coco_label)

        # dedupe: keep best confidence for each label
        prev = best_by_label.get(label, 0.0)
        if conf > prev:
            best_by_label[label] = conf

    # output sorted by confidence desc
    objects = [
        {"label": label, "confidence": round(conf, 3)}
        for label, conf in sorted(best_by_label.items(), key=lambda x: x[1], reverse=True)
    ]
    return objects
