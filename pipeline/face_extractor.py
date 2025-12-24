import cv2
import numpy as np

FACE_CASCADE_PATH = "models/face_detection/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def detect_faces(image):
    """
    Detect faces and assign Face tokens.
    """
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    detections = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    faces = []

    for idx, (x, y, w, h) in enumerate(detections, start=1):
        faces.append({
            "token": f"Face_{idx}",
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "confidence": None
        })

    return faces
