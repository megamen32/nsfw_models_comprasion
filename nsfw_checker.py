import os
import time
import tempfile
import mimetypes
import cv2
import torch
from pathlib import Path
from typing import Union
from PIL import Image

# --- НАСТРОЙКИ ---
USE_NUDENET = True
NSFW_MODEL_NAME = "marqo"

NSFW_MODELS = {
    "marqo": {
        "type": "timm",
        "id": "hf_hub:marqo/nsfw-image-detection-384",
        "threshold": 0.8
    },
    "adamcodd": {
        "type": "hf",
        "id": "AdamCodd/vit-base-nsfw-detector",
        "threshold": 0.8
    },
    "falconsai": {
        "type": "hf",
        "id": "Falconsai/nsfw_image_detection",
        "threshold": 0.7
    }
}

# --- Переменные ---
_nsfw_model = None
_nsfw_processor = None
_nudenet = None

def _load_model():
    global _nsfw_model, _nsfw_processor, _nudenet

    if _nsfw_model is not None:
        return

    model_info = NSFW_MODELS[NSFW_MODEL_NAME]
    model_type = model_info["type"]

    if model_type == "timm":
        import timm
        from timm.data import resolve_model_data_config, create_transform

        _nsfw_model = timm.create_model(model_info["id"], pretrained=True)
        _nsfw_model = _nsfw_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
        config = resolve_model_data_config(_nsfw_model)
        _nsfw_processor = create_transform(**config, is_training=False)

    elif model_type == "hf":
        from transformers import AutoProcessor, AutoModelForImageClassification
        _nsfw_processor = AutoProcessor.from_pretrained(model_info["id"], use_fast=True)
        _nsfw_model = AutoModelForImageClassification.from_pretrained(model_info["id"])
        _nsfw_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if USE_NUDENET:
        from nudenet import NudeDetector
        _nudenet = NudeDetector()

def _run_nsfw_model(image: Image.Image):
    model_info = NSFW_MODELS[NSFW_MODEL_NAME]
    threshold = model_info["threshold"]

    if model_info["type"] == "timm":
        device = next(_nsfw_model.parameters()).device
        input_tensor = _nsfw_processor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = _nsfw_model(input_tensor).softmax(dim=-1).cpu()[0]
        score = float(probs[0])
    else:
        inputs = _nsfw_processor(images=image, return_tensors="pt").to(_nsfw_model.device)
        with torch.no_grad():
            logits = _nsfw_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
        score = float(probs[0][1])

    return score >= threshold, score
skip_class=[
    # "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
   # "BUTTOCKS_EXPOSED",
   # "FEMALE_BREAST_EXPOSED",
   # "FEMALE_GENITALIA_EXPOSED",
   # "MALE_BREAST_EXPOSED",
  #  "ANUS_EXPOSED",
   # "FEET_EXPOSED",
   # "BELLY_COVERED",
   # "FEET_COVERED",
  #  "ARMPITS_COVERED",
  #  "ARMPITS_EXPOSED",
    "FACE_MALE",
  #  "BELLY_EXPOSED",
   # "MALE_GENITALIA_EXPOSED",
  #  "ANUS_COVERED",
  #  "FEMALE_BREAST_COVERED",
  #  "BUTTOCKS_COVERED",
]
def _run_nudenet(image_path: str, threshold: float = 0.4):
    detections = _nudenet.detect(image_path)
    flagged = [d for d in detections if d["score"] >= threshold and d['class'] not in skip_class]
    score = max([d["score"] for d in flagged], default=0.0)
    return bool(flagged), score, flagged

def _check_image(image: Union[str, Image.Image]):
    _load_model()

    if isinstance(image, str):
        image_pil = Image.open(image).convert("RGB")
        image_path = image
    else:
        image_pil = image.convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_pil.save(tmp.name)
            image_path = tmp.name

    nsfw_flagged, nsfw_score = _run_nsfw_model(image_pil)

    if USE_NUDENET:
        nude_flagged, nude_score, nude_classes = _run_nudenet(image_path)
    else:
        nude_flagged, nude_score, nude_classes = False, 0.0, []

    is_nsfw = nsfw_flagged or nude_flagged
    max_score = max(nsfw_score, nude_score)

    return is_nsfw, max_score, {
        "nsfw_score": nsfw_score,
        "nude_score": nude_score,
        "nudenet_flagged": nude_flagged,
        "nude_classes": nude_classes,
        "nsfw_flagged": nsfw_flagged,
        "model": NSFW_MODEL_NAME
    }

def _extract_frame(path: str, last: bool = False) -> Image.Image:
    cap = cv2.VideoCapture(path)
    if last:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("❌ Не удалось извлечь кадр из видео.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def analyze_file(file_path: str):
    mime_type, _ = mimetypes.guess_type(file_path)
    is_video = mime_type and mime_type.startswith("video")

    if is_video:
        img_first = _extract_frame(file_path, last=False)
        img_last = _extract_frame(file_path, last=True)
        res_first = _check_image(img_first)
        res_last = _check_image(img_last)
        best = max([res_first, res_last], key=lambda x: x[1])
        return best[0], best[1], file_path, best[2]
    else:
        return *_check_image(file_path), file_path

def analyze_bytes(file_bytes: bytes, suffix: str = ".jpg"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return analyze_file(tmp_path)

async def check_avatar(bot, user_id: int):
    # 2) Если нет в кэше — грузим foto
    photos = await bot.get_user_profile_photos(user_id, limit=1)
    if not photos.total_count:
        result = (False, 0.0, "")
        return result

    file_id = photos.photos[0][0].file_id
    file = await bot.get_file(file_id)
    ext = Path(file.file_path).suffix.lower()
    file_bytes = (await bot.download_file(file.file_path)).read()

    # 3) Проверяем моделью
    is_nsfw, score, _, _ = analyze_bytes(file_bytes, suffix=ext)

    result = (is_nsfw, score, file_id)
    return result

if __name__ == "__main__":
    path = "antiwhorsebot/photo_2025-05-15_13-16-35.mp4"
    is_nsfw, score, src, debug = analyze_file(path)
    print("🔍 Проверка:", src)
    print("⚠️ NSFW:", is_nsfw)
    print("📈 Уверенность:", score)
    print("🧠 Детали:", debug)
