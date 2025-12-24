from ultralytics import YOLO
import numpy as np

_model = YOLO("yolov8n.pt")  # fast starter

def detect_objects(image, conf_threshold: float = 0.25):
    img = np.array(image)  # PIL -> numpy RGB
    results = _model.predict(img, verbose=False)[0]

    objects = []
    if results.boxes is None:
        return objects

    for b in results.boxes:
        conf = float(b.conf.item())
        if conf < conf_threshold:
            continue
        cls_id = int(b.cls.item())
        label = results.names[cls_id]
        objects.append({"label": label, "confidence": round(conf, 3)})

    return objects
