"""Classificador dos sinais oficiais usados em EVA (NASA / MDRS)."""
import math
from collections import Counter, deque

# MediaPipe Hands - 21 landmarks
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# MediaPipe Pose - landmarks relevantes
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16


def _dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def _fingers_extended(lm):
    """Retorna [polegar, indicador, medio, anelar, mindinho] como booleanos."""
    thumb = _dist(lm[THUMB_TIP], lm[WRIST]) > _dist(lm[THUMB_IP], lm[WRIST]) * 1.1
    index = lm[INDEX_TIP].y < lm[INDEX_PIP].y
    middle = lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y
    ring = lm[RING_TIP].y < lm[RING_PIP].y
    pinky = lm[PINKY_TIP].y < lm[PINKY_PIP].y
    return [thumb, index, middle, ring, pinky]


def classify_hand_gesture(landmarks):
    """Sinais oficiais de mao: OK, WAIT, THUMBS UP/DOWN."""
    lm = landmarks
    fingers = _fingers_extended(lm)
    thumb, index, middle, ring, pinky = fingers

    palm_size = _dist(lm[WRIST], lm[MIDDLE_MCP])
    pinch = _dist(lm[THUMB_TIP], lm[INDEX_TIP])

    # OK informal: polegar e indicador unidos, demais estendidos
    if pinch < 0.4 * palm_size and middle and ring and pinky:
        return "OK"

    # WAIT / Perigo: punho fechado
    if not any(fingers):
        return "WAIT"

    # Polegar isolado: bom ou ruim
    if thumb and not index and not middle and not ring and not pinky:
        if lm[THUMB_TIP].y < lm[WRIST].y:
            return "THUMBS_UP"
        return "THUMBS_DOWN"

    return None


# ---------------------------------------------------------------------------
# Sinais de pose (corpo todo)
# ---------------------------------------------------------------------------

def _visible(landmark, threshold=0.5):
    return getattr(landmark, "visibility", 1.0) >= threshold


def is_emergency_pose(pose_landmarks):
    """AMBOS os bracos cruzados no peito: 'Emergencia, levar a airlock' (NASA)."""
    lm = pose_landmarks
    lw, rw = lm[POSE_LEFT_WRIST], lm[POSE_RIGHT_WRIST]
    ls, rs = lm[POSE_LEFT_SHOULDER], lm[POSE_RIGHT_SHOULDER]

    if not (_visible(ls) and _visible(rs)):
        return False
    # Quando cruzados, punhos podem ficar parcialmente ocultos atras do braco
    if not (_visible(lw, 0.3) and _visible(rw, 0.3)):
        return False

    shoulder_dist = _dist(ls, rs)
    if shoulder_dist < 0.05:
        return False

    threshold = 0.75 * shoulder_dist

    # Cada punho deve estar perto do ombro OPOSTO (independente de qual lado MediaPipe nomeou)
    lw_near_rs = _dist(lw, rs) < threshold
    rw_near_ls = _dist(rw, ls) < threshold

    return lw_near_rs and rw_near_ls


def is_air_pose(pose_landmarks):
    """UM braco cruzado no peito: 'Verificar oxigenio/ar' (MDRS)."""
    if is_emergency_pose(pose_landmarks):
        return False

    lm = pose_landmarks
    lw, rw = lm[POSE_LEFT_WRIST], lm[POSE_RIGHT_WRIST]
    ls, rs = lm[POSE_LEFT_SHOULDER], lm[POSE_RIGHT_SHOULDER]

    if not (_visible(ls) and _visible(rs)):
        return False

    shoulder_dist = _dist(ls, rs)
    if shoulder_dist < 0.05:
        return False

    threshold = 0.65 * shoulder_dist

    lw_crossed = _visible(lw, 0.3) and _dist(lw, rs) < threshold
    rw_crossed = _visible(rw, 0.3) and _dist(rw, ls) < threshold

    return lw_crossed != rw_crossed  # XOR: exatamente um


def is_stop_pose(pose_landmarks):
    """STOP oficial (MDRS): braco dobrado 90 graus para cima."""
    lm = pose_landmarks

    arms = (
        (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST),
        (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST),
    )

    for s_idx, e_idx, w_idx in arms:
        s, e, w = lm[s_idx], lm[e_idx], lm[w_idx]
        if not (_visible(s) and _visible(e) and _visible(w)):
            continue

        # Antebraco apontando para cima: punho bem acima do cotovelo
        wrist_above_elbow = w.y < e.y - 0.08
        # Cotovelo proximo da altura do ombro (caracteriza dobra 90 graus)
        elbow_at_shoulder = abs(e.y - s.y) < 0.15
        # Antebraco quase vertical (punho alinhado com cotovelo no eixo x)
        forearm_vertical = abs(w.x - e.x) < 0.10

        if wrist_above_elbow and elbow_at_shoulder and forearm_vertical:
            return True

    return False


def is_ok_formal_pose(pose_landmarks):
    """OK formal (MDRS): mao tocando o topo da cabeca."""
    lm = pose_landmarks
    nose = lm[POSE_NOSE]
    if not _visible(nose):
        return False

    for w_idx in (POSE_LEFT_WRIST, POSE_RIGHT_WRIST):
        w = lm[w_idx]
        if not _visible(w, 0.4):
            continue
        # Punho acima do nariz e proximo no eixo x
        if w.y < nose.y - 0.05 and abs(w.x - nose.x) < 0.20:
            return True

    return False


# ---------------------------------------------------------------------------
# Suavizacao temporal
# ---------------------------------------------------------------------------

class GestureSmoother:
    """Votacao majoritaria em janela deslizante para eliminar flicker."""

    def __init__(self, window=8, min_consensus=0.6):
        self.window = window
        self.threshold = int(window * min_consensus)
        self.history = deque(maxlen=window)

    def update(self, gesture):
        self.history.append(gesture)
        if len(self.history) < self.window:
            return None
        most_common, count = Counter(self.history).most_common(1)[0]
        return most_common if count >= self.threshold else None
