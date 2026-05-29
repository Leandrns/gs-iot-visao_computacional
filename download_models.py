"""Baixa os modelos .task usados pela API nova do MediaPipe."""
import urllib.request
from pathlib import Path

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

MODELS_DIR = Path(__file__).parent / "models"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        target = MODELS_DIR / name
        if target.exists():
            print(f"[OK] {name} ja existe.")
            continue
        print(f"[..] Baixando {name}...")
        urllib.request.urlretrieve(url, target)
        print(f"[OK] {name} salvo em {target}")


if __name__ == "__main__":
    main()
