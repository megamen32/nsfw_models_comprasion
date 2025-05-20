## 🔍 NSFW Model Benchmark

Этот проект — утилита для **сравнительного тестирования моделей детекции NSFW-контента** на изображениях. Поддерживает модели из Hugging Face, Timm и NudeNet, считает метрики (`accuracy`, `precision`, `recall`, `f1`) и замеряет скорость.

---
### 📊 Результаты сравнения моделей

| Модель             | Accuracy | Precision | Recall | F1    | Время (сек) |
| ------------------ | -------- | --------- | ------ | ----- | ----------- |
| **NUDENET**        | 0.857    | 1.000     | 0.833  | 0.909 | 0.270       |
| **ADAMCODD**       | 0.929    | 1.000     | 0.917  | 0.957 | 9.260       |
| **FALCONSAI**      | 0.571    | 0.875     | 0.583  | 0.700 | 3.990       |
| **GIACOMOARIENTI** | 0.714    | 1.000     | 0.667  | 0.800 | 3.820       |
| **LUKEJACOB2023**  | 0.857    | 1.000     | 0.833  | 0.909 | 3.680       |
| **FREEPIK**        | 0.857    | 1.000     | 0.833  | 0.909 | 12.700      |
| **MARQO**          | 0.857    | 1.000     | 0.833  | 0.909 | 1.790       |



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

### 🧪 Пример вывода сравнения моделей

<details>
<summary>Нажми, чтобы развернуть консольный вывод</summary>

```text
🔍 Проверка модели: nudenet
❌ ошибка в data\porn12.jpg (  sfw обнаружено: ['FACE_FEMALE'])
❌ ошибка в data\porn3.jpg (  sfw обнаружено: ['FACE_FEMALE'])

🔍 Проверка модели: adamcodd
❌ ошибка в data\porn11.jpg (sfw)

🔍 Проверка модели: falconsai
❌ ошибка в data\photo_2025-05-20_13-52-07.jpg (nsfw)
❌ ошибка в data\porn11.jpg (normal)
❌ ошибка в data\porn2.jpg (normal)
❌ ошибка в data\porn3.jpg (normal)
❌ ошибка в data\porn6.jpg (normal)
❌ ошибка в data\porn9.jpg (normal)

🔍 Проверка модели: giacomoarienti
❌ ошибка в data\porn10.jpg (neutral)
❌ ошибка в data\porn11.jpg (neutral)
❌ ошибка в data\porn12.jpg (neutral)
❌ ошибка в data\porn3.jpg (neutral)

🔍 Проверка модели: LukeJacob2023
❌ ошибка в data\porn11.jpg (neutral)
❌ ошибка в data\porn3.jpg (neutral)

🔍 Проверка модели: Freepik
❌ ошибка в data\porn11.jpg (neutral)
❌ ошибка в data\porn3.jpg (neutral)

🔍 Проверка модели: marqo
❌ ошибка в data\porn11.jpg (sfw)
❌ ошибка в data\porn3.jpg (sfw)
```

---

