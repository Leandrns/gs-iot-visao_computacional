"""Tradutor de sinais EVA via MediaPipe Tasks"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_classifier import (
    GestureSmoother,
    classify_hand_gesture,
    is_air_pose,
    is_emergency_pose,
    is_ok_formal_pose,
    is_stop_pose,
)
from overlay import draw_hand_landmarks, draw_hud, draw_pose_landmarks

MODELS_DIR = Path(__file__).parent / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"


def build_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def build_pose_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    if not HAND_MODEL.exists() or not POSE_MODEL.exists():
        raise FileNotFoundError(
            "Modelos .task ausentes. Rode primeiro: python download_models.py"
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam (indice 0).")

    hand_landmarker = build_hand_landmarker()
    pose_landmarker = build_pose_landmarker()
    smoother = GestureSmoother(window=8, min_consensus=0.6)

    start = time.time()
    prev_time = start

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            raw_gesture = None

            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                draw_pose_landmarks(frame, landmarks)
                if is_emergency_pose(landmarks):
                    raw_gesture = "EMERGENCY"
                elif is_air_pose(landmarks):
                    raw_gesture = "AIR"
                elif is_stop_pose(landmarks):
                    raw_gesture = "STOP"
                elif is_ok_formal_pose(landmarks):
                    raw_gesture = "OK_FORMAL"

            if raw_gesture is None and hand_result.hand_landmarks:
                landmarks = hand_result.hand_landmarks[0]
                draw_hand_landmarks(frame, landmarks)
                g = classify_hand_gesture(landmarks)
                if g:
                    raw_gesture = g

            gesture = smoother.update(raw_gesture)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            draw_hud(frame, gesture, fps)
            cv2.imshow("Tradutor de Sinais EVA", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        pose_landmarker.close()


if __name__ == "__main__":
    main()
