import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SCENARIOS = [
    ("exp5_off", 5.0, 0.35),
    ("exp5_weak", 5.0, 0.175),
    ("exp5_partial", 5.0, 0.05),
    ("exp10_period", 10.0, 0.35),
]
VARIANTS = ("clean", "pwm", "motion", "combined")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_report(path, expected, count):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        p = d["parameters"]
        if d["scenes"] != count or d["inserted_leds"] != count:
            return None
        if abs(float(p["exposure_ms"]) - expected[1]) > 1e-7:
            return None
        if abs(float(p["phase"]) - expected[2]) > 1e-7:
            return None
        if abs(float(p["speed_kmh"]) - 90) > 1e-7 or abs(float(p["depth_m"]) - 15) > 1e-7:
            return None
        if abs(float(p["pwm_hz"]) - 100) > 1e-7 or abs(float(p["duty"]) - 0.2) > 1e-7:
            return None
        if "end_to_end_color" not in d:
            return None
        return d
    except (ValueError, KeyError, TypeError, OSError):
        return None


def write_output(root, result_dir, reports, count):
    rows = []
    for name, t, phase in SCENARIOS:
        d = reports[name]
        for variant in VARIANTS:
            color = d["end_to_end_color"][variant]
            for cls in ("person", "car", "traffic light"):
                metric = d["detector"][variant][cls]
                rows.append({
                    "regime": name,
                    "exposure_ms": t,
                    "phase": phase,
                    "on_fraction": d["pwm_on_fraction"],
                    "variant": variant,
                    "class": cls,
                    "tp": metric["TP"],
                    "fp": metric["FP"],
                    "fn": metric["FN"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "color_correct": color["correct"],
                    "color_wrong_or_unknown": color["wrong_or_unknown"],
                    "color_missed": color["missed"],
                })
    result_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "CONTROLLED_SYNTHETIC_LED_TECHNICAL_PILOT",
        "note": "Технический тест, не независимое доказательство безопасности камеры или гипотезы выбора экспозиции",
        "scenes_per_regime": count,
        "same_scene_ids_across_regimes": True,
        "regimes": {},
    }
    for name, t, phase in SCENARIOS:
        d = reports[name]
        summary["regimes"][name] = {
            "exposure_ms": t,
            "phase": phase,
            "pwm_on_fraction": d["pwm_on_fraction"],
            "color_by_variant": d["end_to_end_color"],
            "detector_by_variant": d["detector"],
        }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    script = root / "run_led_control_pilot.py"
    weights = root / "yolo11s.pt"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "count": count,
        "commands": [
            [sys.executable, str(script), "--count", str(count), "--exposure-ms", str(t),
             "--reference-exposure-ms", "5", "--speed-kmh", "90", "--depth-m", "15",
             "--phase", str(ph), "--yolo"]
            for _, t, ph in SCENARIOS
        ],
        "generator_sha256": file_sha256(script),
        "weights_sha256": file_sha256(weights) if weights.exists() else None,
        "note": "Сцены с синтетически добавленным крупным светодиодным светофором; результаты нельзя переносить на реальные маленькие светофоры без проверки",
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    archive = root / "led_phase20_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for filename in ("summary.csv", "summary.json", "manifest.json"):
            z.write(result_dir / filename, arcname=f"led_phase20/{filename}")
        for name, _, _ in SCENARIOS:
            base = result_dir / name
            for filename in ("report.json", "predictions.json"):
                z.write(base / filename, arcname=f"led_phase20/{name}/{filename}")
            for image_name in ("01_clean.png", "01_pwm.png", "01_motion.png", "01_combined.png", "01_combined_pred.png"):
                p = base / "images" / image_name
                if p.exists():
                    z.write(p, arcname=f"led_phase20/{name}/images/{image_name}")
    print("\nСводка:")
    for name, t, ph in SCENARIOS:
        d = reports[name]
        colors = d["end_to_end_color"]
        print(f"{name}: t={t:g} мс, фаза={ph:g}, доля свечения={d['pwm_on_fraction']:.3f}, "
              f"цвет PWM={colors['pwm']['correct']}/{count}, "
              f"совместно={colors['combined']['correct']}/{count}")
    print(f"\nАрхив для отправки: {archive}")
    print("Полные изображения остались в", result_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    generator = root / "run_led_control_pilot.py"
    if not generator.is_file():
        raise SystemExit("Не найден run_led_control_pilot.py. Поместите оба файла в одну папку проекта.")
    if args.count < 1:
        raise SystemExit("Количество сцен должно быть положительным.")
    result_dir = root / "outputs_bdd_control" / "led_phase20"
    output_dir = root / "outputs_bdd_control" / "led_control_pilot"
    reports = {}
    for i, scenario in enumerate(SCENARIOS, start=1):
        name, exposure, phase = scenario
        target = result_dir / name
        old = checked_report(target / "report.json", scenario, args.count)
        if old is not None and (target / "predictions.json").exists():
            print(f"{i}/{len(SCENARIOS)}: {name}: готовый результат найден, повторный запуск не нужен", flush=True)
            reports[name] = old
            continue
        cmd = [
            sys.executable, str(generator), "--count", str(args.count),
            "--exposure-ms", str(exposure), "--reference-exposure-ms", "5",
            "--speed-kmh", "90", "--depth-m", "15", "--phase", str(phase), "--yolo",
        ]
        print(f"{i}/{len(SCENARIOS)}: {name}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=root, check=True)
        report = checked_report(output_dir / "report.json", scenario, args.count)
        if report is None:
            raise SystemExit(f"Отчёт {name} отсутствует или его параметры не совпадают с заданными")
        if not (output_dir / "predictions.json").is_file():
            raise SystemExit(f"Не найдены предсказания для {name}")
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_dir / "report.json", target / "report.json")
        shutil.copy2(output_dir / "predictions.json", target / "predictions.json")
        if (output_dir / "images").exists():
            shutil.copytree(output_dir / "images", target / "images", dirs_exist_ok=True)
        reports[name] = report
    if args.dry_run:
        return
    first_files = json.loads((result_dir / SCENARIOS[0][0] / "predictions.json").read_text(encoding="utf-8"))
    first_ids = [x["image"] for x in first_files]
    if len(first_ids) != args.count:
        raise SystemExit("Число исходных сцен не совпадает с запрошенным")
    for name, _, _ in SCENARIOS[1:]:
        data = json.loads((result_dir / name / "predictions.json").read_text(encoding="utf-8"))
        if [x["image"] for x in data] != first_ids:
            raise SystemExit(f"В {name} использованы другие исходные сцены. Сравнение запрещено")
    write_output(root, result_dir, reports, args.count)


if __name__ == "__main__":
    main()
