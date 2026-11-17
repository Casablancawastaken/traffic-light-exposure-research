# Воспроизводимые данные технического пилота

В папке представлены только исходный код технического пилота, результаты и проверочные материалы для преподавателя. Статья оформляется отдельно и в Git не включается.

**Статус:** технический пилот на 20 настроечных изображениях; основная гипотеза и независимая проверка критерия пока не завершены.

## Содержимое

- `pilot/code/`: исходный генератор, запуск четырёх режимов, анализатор и скрипты дополнительных проверок
- `pilot/reports/config.json`: параметры пилота и контрольная сумма весов YOLO
- `pilot/reports/source_manifest.json`: 20 исходных изображений и SHA-256; GT синтетического контрольного светофора
- `pilot/reports/predictions_with_sizegroups.json`: исходные confidence и размерные группы предсказаний
- `pilot/results/`: CSV/JSON и полные предсказания по каждому режиму, примеры четырёх вариантов изображений
- `pilot/gt_examples/`: пять примеров GT t/2 из предыдущего пилота; **не** полная разметка 20 сцен
- `pilot/reports/verification_320.json`: результат проверки 320 записей
- `protocol/`: пример правила A/R/U, статусов S/F/Q, доверительных границ и бюджетов ошибок

## Пересчёт таблиц 2–3 и проверка 320 записей

Из `docs/vision_exposure` выполните:

```bash
python pilot/code/rebuild_result_zip.py --data pilot/results --out led_phase20_results.zip
python pilot/code/verify_led_phase20.py --zip led_phase20_results.zip --generator pilot/code/run_led_control_pilot.py --out audit_rerun.json
python protocol/criterion_example.py --config protocol/criterion_config.json --out protocol/criterion_example_report.json
```

Ожидается `PASS`, `320` записей и `321` проверка. Исходные фотографии BDD100K и веса YOLO не публикуются. Запуск полного конвейера требует локального датасета и окружения Python.

## Дополнительные GT и 5 контрастных сцен

Эти проверки необходимо выполнить **локально**, результаты не входят в базовый пакет:

```bash
python export_gt_midpoint.py --project-root . --count 20
python run_multiscene_convergence.py --project-root . --count 5 --pool 20 --render-width 384 --yolo
cp -a outputs_bdd_control/gt_midpoint_export docs/vision_exposure/pilot/
cp -a outputs_bdd_control/multiscene_convergence docs/vision_exposure/pilot/
```

Итоговые JSON/CSV не следует объявлять полученными до успешного выполнения этих команд.

## Границы интерпретации

Добавленный крупный светофор не заменяет проверку естественных малых светофоров. Синтетический смаз и ШИМ добавлялись к JPEG с неизвестной исходной экспозицией. Критерий ещё не обучался и не проверялся на независимых сценах.
