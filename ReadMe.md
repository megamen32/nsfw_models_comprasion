# 🔍 NSFW Model Benchmark

Этот проект позволяет:

* 📊 **сравнивать точность и скорость** NSFW-моделей (из Hugging Face, Timm и NudeNet);
* 🧪 **проводить собственные тесты** на своих изображениях или видео;
* ⚙️ использовать готовую функцию `analyze_file` или интегрировать в Telegram-бота через `check_avatar`.

---

## 📊 Результаты сравнения моделей

| Модель             | Accuracy | Precision | Recall | F1    | Время (сек) |
| ------------------ | -------- | --------- | ------ | ----- | ----------- |
| **NUDENET**        | 0.857    | 1.000     | 0.833  | 0.909 | 0.270       |
| **ADAMCODD**       | 0.929    | 1.000     | 0.917  | 0.957 | 9.260       |
| **FALCONSAI**      | 0.571    | 0.875     | 0.583  | 0.700 | 3.990       |
| **GIACOMOARIENTI** | 0.714    | 1.000     | 0.667  | 0.800 | 3.820       |
| **LUKEJACOB2023**  | 0.857    | 1.000     | 0.833  | 0.909 | 3.680       |
| **FREEPIK**        | 0.857    | 1.000     | 0.833  | 0.909 | 12.700      |
| **MARQO**          | 0.857    | 1.000     | 0.833  | 0.909 | 1.790       |

---

## 📦 Поддерживаемые модели

| Тип        | Модель ID                           | Источник                     |
| ---------- | ----------------------------------- | ---------------------------- |
| 🤗 HF      | `AdamCodd/vit-base-nsfw-detector`   | transformers                 |
| 🤗 HF      | `Falconsai/nsfw_image_detection`    | transformers                 |
| 🤗 HF      | `giacomoarienti/nsfw-classifier`    | transformers                 |
| 🤗 HF      | `LukeJacob2023/nsfw-image-detector` | transformers                 |
| 🤗 HF      | `Freepik/nsfw_image_detector`       | transformers                 |
| 🧠 Timm    | `marqo/nsfw-image-detection-384`    | timm / hf\_hub               |
| 👀 NudeNet | `NudeDetector`                      | локальный детектор (nudenet) |

---

## 🚀 Быстрый старт

1. Установи зависимости:

```bash
pip install -r requirements.txt
```

2. Положи изображения в папку `data/`.
   Файлы, в названии которых есть `porn`, будут считаться `NSFW`, остальные — `SFW`.

3. Запусти сравнительный тест моделей:

```bash
python main.py
```

---

## 🧰 Проверка изображений и видео вручную

В проекте есть модуль [`nsfw_checker.py`](./nsfw_checker.py), который позволяет:

* проверять **любые изображения или видео** с помощью выбранной модели;
* использовать **встроенную функцию `analyze_file()`**;
* обрабатывать `.jpg`, `.png`, `.mp4`, `.mov` и пр.;
* получать результат: `is_nsfw`, `уверенность`, `debug-информацию`.

### Пример использования:

```python
from nsfw_checker import analyze_file

if __name__ == "__main__":
    path = "antiwhorebot/photo_2025-05-15_13-16-35.mp4"
    is_nsfw, score, src, debug = analyze_file(path)
    print("🔍 Проверка:", src)
    print("⚠️ NSFW:", is_nsfw)
    print("📈 Уверенность:", score)
    print("🧠 Детали:", debug)
```

---

## 🤖 Поддержка aiogram (Telegram-бот)

В `nsfw_checker.py` реализована функция `check_avatar(bot, user_id)`, которая:

* загружает аватарку пользователя;
* извлекает изображение;
* проверяет на NSFW;
* возвращает: `bool`, `уверенность`, `file_id`.

Пример:

```python
is_nsfw, score, file_id = await check_avatar(bot, user_id)
```

Идеально подходит для антиспам/антишлюх ботов.

---

## 📂 Структура проекта

```
.
├── main.py            # сравнение всех моделей
├── nsfw_checker.py    # API для проверки отдельных файлов
├── data/              # изображения и видео для теста
├── requirements.txt
└── README.md
```

---

## 🧪 Пример вывода ошибок

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

</details>

---
