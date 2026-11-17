import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

MODES = ("exp5_off", "exp5_weak", "exp5_partial", "exp10_period")
VARIANTS = ("clean", "pwm", "motion", "combined")
CLASSES = ("person", "car", "traffic light")


def audit(zip_path, generator_path):
    problems = []
    check_count = 0
    results = {}
    with ZipFile(zip_path) as z:
        manifest = json.loads(z.read("led_phase20/manifest.json"))
        standard_ids = None
        for mode in MODES:
            base = f"led_phase20/{mode}"
            report = json.loads(z.read(f"{base}/report.json"))
            records = json.loads(z.read(f"{base}/predictions.json"))
            if len(records) != 20 or report["scenes"] != 20:
                problems.append(f"{mode}: число сцен отличается от 20")
            ids = [(record["image"], record["source_sha256"], record["color"]) for record in records]
            if standard_ids is None:
                standard_ids = ids
            elif ids != standard_ids:
                problems.append(f"{mode}: список сцен или цветов отличается от первого режима")
            class_totals = {v: {c: {k: 0 for k in ("TP", "FP", "FN")} for c in CLASSES} for v in VARIANTS}
            color_totals = {v: {k: 0 for k in ("correct", "wrong_or_unknown", "missed")} for v in VARIANTS}
            for index, record in enumerate(records):
                for variant in VARIANTS:
                    entry = record["variants"][variant]
                    status = entry["end_to_end_color"]["status"]
                    if status not in color_totals[variant]:
                        problems.append(f"{mode}: неизвестный статус {status}")
                    else:
                        color_totals[variant][status] += 1
                    for cls in CLASSES:
                        for k in ("TP", "FP", "FN"):
                            class_totals[variant][cls][k] += entry["counts"][cls][k]
                    if index == 0:
                        filename = f"{base}/images/01_{variant}.png"
                        try:
                            hash_value = hashlib.sha256(z.read(filename)).hexdigest()
                            if hash_value != entry["sha256"]:
                                problems.append(f"{filename}: SHA-256 не совпал")
                        except KeyError:
                            problems.append(f"{filename}: отсутствует пример изображения")
                    check_count += 1
            for variant in VARIANTS:
                if color_totals[variant] != report["end_to_end_color"][variant]:
                    problems.append(f"{mode}/{variant}: не совпадает агрегат цветов")
                for cls in CLASSES:
                    ref = report["detector"][variant][cls]
                    if any(class_totals[variant][cls][k] != ref[k] for k in ("TP", "FP", "FN")):
                        problems.append(f"{mode}/{variant}/{cls}: не совпадает агрегат детектора")
            results[mode] = {
                "exposure_ms": report["parameters"]["exposure_ms"],
                "phase": report["parameters"]["phase"],
                "pwm_on_fraction": report["pwm_on_fraction"],
                "color_correct_pwm": color_totals["pwm"]["correct"],
                "color_correct_combined": color_totals["combined"]["correct"],
                "color_missed_combined": color_totals["combined"]["missed"],
                "detector_clean": class_totals["clean"],
                "detector_motion": class_totals["motion"],
            }
        source_path = Path(generator_path)
        source_verified = source_path.exists() and hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest["generator_sha256"]
        if not source_verified:
            problems.append("Невозможно подтвердить SHA-256 скрипта генератора")
        check_count += 1
        return {
            "status": "PASS" if not problems else "FAIL",
            "scenes_per_regime": 20,
            "regimes": len(MODES),
            "per_scene_variant_records_checked": 20 * len(MODES) * len(VARIANTS),
            "same_scene_ids_and_colors": standard_ids is not None and not any('список сцен' in p for p in problems),
            "generator_sha256_verified": source_verified,
            "checks_executed": check_count,
            "issues": problems,
            "results": results,
            "limitations": [
                "Не подтверждает физическую достоверность двумерного синтеза",
                "Режимы используют фиксированную фазу, а не независимые фазы ШИМ",
                "Примеры изображений в ZIP сохранены для одной сцены каждого режима",
                "Не проверяет логистический критерий выбора экспозиции на независимых сценах",
                "Не измеряет доверительные интервалы и не доказывает выполнение порога ошибок 5%",
            ],
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default='led_phase20_results.zip')
    parser.add_argument('--generator', default='run_led_control_pilot.py')
    parser.add_argument('--out', default='Проверка_пилота_20_сцен.json')
    args = parser.parse_args()
    result = audit(args.zip, args.generator)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, ensure_ascii=False, indent=2))
