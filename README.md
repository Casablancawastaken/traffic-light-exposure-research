# Traffic Light Exposure Research

Исследование зависимости качества обнаружения дорожных объектов и определения цвета светодиодного светофора от времени экспозиции, ШИМ и смаза движения.

**Состояние:** технический пилот. Основная исследовательская гипотеза не подтверждена; статья хранится отдельно.

## Структура

- `pilot.py`, `bdd_clean_control.py`, `run_*.py`, `export_gt_midpoint.py` — код синтеза, скачивания данных и проверок
- `config.json`, `requirements.txt` — параметры и зависимости
- [`docs/vision_exposure/`](docs/vision_exposure/README.md) — технические материалы для проверки преподавателем: манифест, SHA-256, GT, исходные confidence, предсказания, результаты CSV/JSON, примеры PNG и анализатор

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'huggingface_hub>=2,<3'
```

Исходные JPEG BDD100K, `.venv`, скачанные веса YOLO и полные временные артефакты остаются локальными и не включаются в репозиторий.

## Проверка сохранённых результатов (без YOLO)

```bash
cd docs/vision_exposure
python pilot/code/rebuild_result_zip.py --data pilot/results --out led_phase20_results.zip
python pilot/code/verify_led_phase20.py --zip led_phase20_results.zip --generator pilot/code/run_led_control_pilot.py --out audit_rerun.json
python protocol/criterion_example.py --config protocol/criterion_config.json --out protocol/criterion_example_report.json
```

Результат аудита арифметической согласованности не доказывает физическую точность симуляции.
