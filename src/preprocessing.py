"""
OpenCV 보조 전처리 — YOLO 추론 결과 시각화용
YOLO 자체 전처리(리사이즈·정규화)는 ultralytics 내부에서 처리됨
"""
import cv2
import numpy as np


def draw_results(img_bgr: np.ndarray, results) -> np.ndarray:
    """
    YOLO results 객체를 받아 바운딩박스 + 라벨을 그린 이미지 반환.
    클래스 이름은 results.names 딕셔너리에서 가져옴.
    """
    vis = img_bgr.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]

        color = (0, 0, 255) if cls_name != "normal" else (0, 200, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(vis, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def draw_detections(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    backend /diagnose가 반환한 JSON 결과(딕셔너리 리스트)를 받아 바운딩박스 +
    라벨을 그린 이미지 반환. draw_results()와 목적은 같지만, 프론트엔드가 더 이상
    ultralytics 객체에 직접 접근하지 못하고(HTTP로 backend를 호출하는 구조라서)
    JSON으로 받은 [{"part_en", "confidence", "bbox": [x1,y1,x2,y2]}, ...] 형태만
    다룰 수 있어서 별도로 추가함.
    """
    vis = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = float(det["confidence"])
        label_text = det.get("part_en", det.get("part", "?"))

        color = (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{label_text} {conf:.2f}"
        cv2.putText(vis, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def resize_for_display(img_bgr: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return img_bgr
