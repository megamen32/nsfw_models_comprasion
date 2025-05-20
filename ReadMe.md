## 🔍 NSFW Model Benchmark

Этот проект — утилита для **сравнительного тестирования моделей детекции NSFW-контента** на изображениях. Поддерживает модели из Hugging Face, Timm и NudeNet, считает метрики (`accuracy`, `precision`, `recall`, `f1`) и замеряет скорость.

---

### 📦 Поддерживаемые модели

| Тип        | Модель ID                           | Источник         |
| ---------- | ----------------------------------- | ---------------- |
| 🤗 HF      | `AdamCodd/vit-base-nsfw-detector`   | transformers     |
| 🤗 HF      | `Falconsai/nsfw_image_detection`    | transformers     |
| 🤗 HF      | `giacomoarienti/nsfw-classifier`    | transformers     |
| 🤗 HF      | `LukeJacob2023/nsfw-image-detector` | transformers     |
| 🤗 HF      | `Freepik/nsfw_image_detector`       | transformers     |
| 🧠 Timm    | `marqo/nsfw-image-detection-384`    | timm / hf\_hub   |
| 👀 NudeNet | `NudeDetector`                      | local classifier |

---

### 🚀 Быстрый старт

1. Установи зависимости:

```bash
pip install -r requirements.txt
```

2. Положи изображения в папку `data/`.
   Файлы с `"porn"` в имени будут считаться `NSFW`, остальные — `SFW`.

3. Запусти скрипт:

```bash
python main.py
```

---

### 📈 Вывод

Скрипт выводит:

* ошибки моделей (если предсказание ≠ названию)
* метрики (`accuracy`, `precision`, `recall`, `f1`)
* скорость обработки (в секундах)

Пример:

```
🔍 Проверка модели: adamcodd
❌ ошибка в data/porn11.jpg (sfw)

🧠 ADAMCODD
accuracy: 0.923
precision: 1.000
recall: 0.909
f1: 0.952
seconds: 8.91
```

---

### ⚙️ Настройки

* `NSFW_KEYWORDS` — список ключевых слов, которые считаются NSFW.
* `SENSITIVE_CLASSES` — классы NudeNet, считающиеся откровенными.
* `threshold` — чувствительность моделей настраивается в коде.

---

### 📂 Структура проекта

```
.
├── main.py         # основной скрипт
├── data/                # изображения для проверки
├── requirements.txt
└── README.md
```

---
