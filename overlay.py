"""HUD + desenho de landmarks"""
import cv2

# Conexoes entre os 21 landmarks da mao
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Subset do esqueleto do pose: tronco + bracos (suficiente pro sinal "bracos cruzados")
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]
POSE_KEYPOINTS = (11, 12, 13, 14, 15, 16)

GESTURE_LABELS = {
    "OK": ("OK - Tudo bem", (0, 255, 0)),
    "OK_FORMAL": ("OK formal (mao na cabeca)", (0, 255, 0)),
    "WAIT": ("WAIT - Atencao, algo incomum", (0, 165, 255)),
    "STOP": ("STOP - Pare imediatamente", (0, 100, 255)),
    "THUMBS_UP": ("Status: BOM", (0, 255, 0)),
    "THUMBS_DOWN": ("Status: RUIM", (0, 100, 255)),
    "AIR": ("AIR - Checar oxigenio", (0, 200, 255)),
    "EMERGENCY": ("!! EMERGENCIA - levar a airlock !!", (0, 0, 255)),
}


def _to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_hand_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [_to_pixel(lm, w, h) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def draw_pose_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [_to_pixel(lm, w, h) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
    for i in POSE_KEYPOINTS:
        if i < len(pts):
            cv2.circle(frame, pts[i], 5, (0, 255, 255), -1)


def draw_hud(frame, gesture, fps=None):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.putText(frame, "Tradutor de sinais - EVA Communication",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    if gesture and gesture in GESTURE_LABELS:
        text, color = GESTURE_LABELS[gesture]
        cv2.putText(frame, text, (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    else:
        cv2.putText(frame, "Aguardando gesto...", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 1)

    if fps is not None:
        cv2.putText(frame, f"{fps:4.1f} FPS", (w - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.putText(frame, "Pressione [Q] para sair",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame
