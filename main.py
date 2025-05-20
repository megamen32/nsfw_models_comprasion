import os
import time
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    AutoProcessor, AutoModelForImageClassification,
    ViTFeatureExtractor, ViTForImageClassification
)
import timm
import timm.data
from urllib.request import urlopen

FOLDER = "data"  # Папка с .jpg файлами

MODELS = {
    "adamcodd": "AdamCodd/vit-base-nsfw-detector",
    "falconsai": "Falconsai/nsfw_image_detection",
    "giacomoarienti": "giacomoarienti/nsfw-classifier",
    "LukeJacob2023": "LukeJacob2023/nsfw-image-detector",
    "Freepik": "Freepik/nsfw_image_detector",
    "marqo": "marqo/nsfw-image-detection-384",  # TIMM-модель
}

NSFW_KEYWORDS = [
    "nsfw", "porn", "xxx", "sexual", "sex", "nude", "nudes", "erotic", "explicit",
    "unsafe", "suggestive", "provocative", "boobs", "vagina", "penis","medium","high","low"
]
SENSITIVE_CLASSES = {
    "FEMALE_GENITALIA_COVERED", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED",
    "FEET_EXPOSED", "BELLY_COVERED", "ARMPITS_EXPOSED", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED"
}

def run_nudenet(dataset, threshold=0.4):
    from nudenet import NudeDetector
    detector = NudeDetector()
    y_true, y_pred = [], []

    start = time.time()

    for path, label in dataset:
        detections = detector.detect(path)
        flagged = [d for d in detections if d["class"] in SENSITIVE_CLASSES and d["score"] >= threshold]
        y_true.append(label)
        y_pred.append(1 if flagged else 0)
        if y_true[-1] != y_pred[-1]:
            missed = [d["class"] for d in detections if d["score"] >= threshold]
            print(f"❌ ошибка в {path} (  {'nsfw' if y_pred[-1] else 'sfw'} обнаружено: {missed})")

    end = time.time()
    duration = end - start

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "seconds": round(duration, 2)
    }


def get_dataset(folder):
    data = []
    for file in os.listdir(folder):
        if file.lower().endswith(".jpg"):
            label = 1 if "porn" in file.lower() else 0
            data.append((os.path.join(folder, file), label))
    return data

def run_model(model_id, dataset, is_timm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = [], []

    start = time.time()

    if is_timm:
        model = timm.create_model(f"hf_hub:{model_id}", pretrained=True).to(device).eval()
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        class_names = model.pretrained_cfg.get("label_names", [])

        for path, label in dataset:
            img = Image.open(path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor).softmax(dim=-1).cpu()[0]
                pred_class = output.argmax().item()
                pred_label = class_names[pred_class].lower() if class_names else "unknown"

            nsfw = (
                    any(tag in pred_label for tag in NSFW_KEYWORDS)
            )
            y_true.append(label)
            y_pred.append(1 if nsfw else 0)
            if y_true[-1] != y_pred[-1]:
                print(f'❌ ошибка в {path} ({pred_label})')

    else:

        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device)


        for path, label in dataset:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                logits = model(**inputs).logits
                pred_class = logits.argmax(dim=1).item()

            pred_label = model.config.id2label.get(pred_class, "").lower()
            nsfw = any(tag in pred_label for tag in NSFW_KEYWORDS)
            y_true.append(label)
            y_pred.append(1 if nsfw else 0)
            if y_true[-1] != y_pred[-1]:
                print(f'❌ ошибка в {path} ({pred_label})')

    end = time.time()
    duration = end - start

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "seconds": round(duration, 2)
    }

if __name__ == "__main__":
    dataset = get_dataset(FOLDER)
    all_results = {}

    print(f"\n🔍 Проверка модели: nudenet")
    all_results["nudenet"] = run_nudenet(dataset)

    for name, model_id in MODELS.items():
        print(f"\n🔍 Проверка модели: {name}")
        is_timm = model_id.startswith("marqo")
        all_results[name] = run_model(model_id, dataset, is_timm=is_timm)


    print("\n📊 Сравнение моделей:")
    for model_name, metrics in all_results.items():
        print(f"\n🧠 {model_name.upper()}")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

