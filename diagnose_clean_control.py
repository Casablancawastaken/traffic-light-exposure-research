import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

CLASSES = ("person", "car", "traffic light")
SIZES = ("small", "medium", "large", "all")
THRESHOLDS = (0.05, 0.10, 0.25)
RESOLUTIONS = (640, 960)


def size_group(area):
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def intersection_over_union(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def add(stat, klass, group, key):
    stat[(klass, group)][key] += 1
    stat[(klass, "all")][key] += 1


def evaluate_scene(ground_truth, predictions, width, height, threshold):
    stat = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    scale = (640 / max(width, height)) ** 2
    for klass in CLASSES:
        expected = [x for x in ground_truth if x["class"] == klass]
        detected = sorted((x for x in predictions if x["class"] == klass and x["confidence"] >= threshold), key=lambda x: -x["confidence"])
        matched = set()
        for obj in detected:
            candidate_index = None
            candidate_score = 0.5
            for i, gt in enumerate(expected):
                if i in matched:
                    continue
                score = intersection_over_union(obj["box"], gt["box"])
                if score >= candidate_score:
                    candidate_index = i
                    candidate_score = score
            if candidate_index is None:
                a = obj["box"]
                area = (a[2] - a[0]) * (a[3] - a[1]) * scale
                add(stat, klass, size_group(area), "FP")
            else:
                matched.add(candidate_index)
                gt = expected[candidate_index]
                add(stat, klass, size_group(gt["size_640"]), "TP")
        for i, gt in enumerate(expected):
            if i not in matched:
                add(stat, klass, size_group(gt["size_640"]), "FN")
    return stat


def merged_counts():
    return defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--source", default="outputs_bdd_control/predictions.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    input_file = root / args.source
    if not input_file.exists():
        raise FileNotFoundError(f"Нет отчёта с разметкой: {input_file}")
    entries = json.loads(input_file.read_text(encoding="utf-8"))[:args.count]
    if not entries:
        raise RuntimeError("В отчёте нет кадров")
    image_root = root / "bdd_sample/data/images_100k/100k/train/shard_000"
    paths = [(entry, image_root / entry["image"]) for entry in entries]
    missing = [str(path) for _, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Исходные изображения не найдены в кэше. Пример: " + missing[0])
    from ultralytics import YOLO
    weight_path = root / "yolo11n.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"Не найдена модель: {weight_path}")
    model = YOLO(str(weight_path))
    output_dir = root / "outputs_bdd_control/diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for resolution in RESOLUTIONS:
        outcomes[resolution] = {threshold: merged_counts() for threshold in THRESHOLDS}
        print(f"Разрешение {resolution}", flush=True)
        for index, (entry, path) in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Не удалось открыть {path}")
            height, width = image.shape[:2]
            result = model.predict(image, imgsz=resolution, conf=0.05, iou=0.7, device="cpu", verbose=False)[0]
            predicted = []
            for b in result.boxes:
                klass = model.names[int(b.cls.item())]
                if klass in CLASSES:
                    predicted.append({"class": klass, "confidence": float(b.conf.item()), "box": [float(x) for x in b.xyxy[0].cpu().tolist()]})
            for threshold in THRESHOLDS:
                scene_result = evaluate_scene(entry["objects_gt"], predicted, width, height, threshold)
                totals = outcomes[resolution][threshold]
                for group, counts in scene_result.items():
                    for key in ("TP", "FP", "FN"):
                        totals[group][key] += counts[key]
            print(f"  {index}/{len(paths)}: {entry['image']}", flush=True)
    table = []
    for resolution in RESOLUTIONS:
        for threshold in THRESHOLDS:
            totals = outcomes[resolution][threshold]
            for klass in CLASSES:
                for group in SIZES:
                    count = totals[(klass, group)]
                    tp, fp, fn = (count[key] for key in ("TP", "FP", "FN"))
                    table.append({"resolution": resolution, "confidence": threshold, "class": klass, "size": group, "GT": tp + fn, "TP": tp, "FP": fp, "FN": fn, "precision": round(tp / (tp + fp), 4) if tp + fp else None, "recall": round(tp / (tp + fn), 4) if tp + fn else None})
    csv_path = output_dir / "threshold_resolution_diagnostic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    report = {"status": "TUNING_PILOT_NOT_INDEPENDENT_TEST", "scenes": len(paths), "weight_sha256": weight_hash, "nms_iou": 0.7, "match_iou": 0.5, "rows": table, "warning": "Это настроечная выборка из отсортированного фрагмента BDD100K. Не использовать для окончательного вывода о качестве."}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nИТОГ ПО ВСЕМ РАЗМЕРАМ")
    print(f"{'imgsz':>5} {'conf':>5} {'класс':>14} {'GT':>4} {'P':>7} {'R':>7}")
    for row in table:
        if row["size"] == "all":
            p = "н/д" if row["precision"] is None else f'{row["precision"]:.3f}'
            r = "н/д" if row["recall"] is None else f'{row["recall"]:.3f}'
            print(f'{row["resolution"]:5} {row["confidence"]:5.2f} {row["class"]:>14} {row["GT"]:4} {p:>7} {r:>7}')
    print(f"\nОтчёт: {csv_path}")


if __name__ == "__main__":
    main()
