import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

CLASSES = ("person", "car", "traffic light")
CLASS_IDS = (0, 2, 9)
THRESHOLDS = (0.05, 0.10, 0.25)
SIZE_GROUPS = ("small", "medium", "large", "all")


def size_group(area):
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def tile_windows(width, height, overlap=0.15):
    tile_w = min(width, int(round(width / (2.0 - overlap))))
    tile_h = min(height, int(round(height / (2.0 - overlap))))
    xs = sorted(set([0, width - tile_w]))
    ys = sorted(set([0, height - tile_h]))
    return [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]


def predictions_from_results(result, model, offset_x=0, offset_y=0):
    out = []
    for b in result.boxes:
        cls = str(model.names[int(b.cls.item())])
        if cls not in CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        out.append({
            "class": cls,
            "confidence": float(b.conf.item()),
            "box": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
        })
    return out


def nms_by_class(predictions, threshold=0.55):
    out = []
    for cls in CLASSES:
        entries = [x for x in predictions if x["class"] == cls]
        if not entries:
            continue
        boxes = []
        scores = []
        for x in entries:
            a = x["box"]
            boxes.append([float(a[0]), float(a[1]), float(a[2] - a[0]), float(a[3] - a[1])])
            scores.append(float(x["confidence"]))
        retained = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=threshold)
        if retained is None or len(retained) == 0:
            continue
        for idx in retained:
            out.append(entries[int(idx)])
    return sorted(out, key=lambda x: -x["confidence"])


def infer_full(model, image, size):
    result = model.predict(image, imgsz=size, conf=0.05, iou=0.7, classes=list(CLASS_IDS), device="cpu", verbose=False)[0]
    return predictions_from_results(result, model)


def infer_tiled(model, image, size):
    h, w = image.shape[:2]
    windows = tile_windows(w, h)
    crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
    results = model.predict(crops, imgsz=size, conf=0.05, iou=0.7, classes=list(CLASS_IDS), device="cpu", verbose=False)
    predictions = []
    for result, (x1, y1, x2, y2) in zip(results, windows):
        predictions.extend(predictions_from_results(result, model, x1, y1))
    return nms_by_class(predictions)


def new_counts():
    return defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})


def add(counts, cls, group, key):
    counts[(cls, group)][key] += 1
    counts[(cls, "all")][key] += 1


def evaluate(gt, pred, width, height, confidence):
    counts = new_counts()
    factor = (640.0 / max(width, height)) ** 2
    for cls in CLASSES:
        expected = [x for x in gt if x["class"] == cls]
        found = sorted((x for x in pred if x["class"] == cls and x["confidence"] >= confidence), key=lambda x: -x["confidence"])
        used = set()
        for detection in found:
            best_index = None
            best_iou = 0.5
            for idx, item in enumerate(expected):
                if idx in used:
                    continue
                overlap = iou(detection["box"], item["box"])
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = idx
            if best_index is None:
                a = detection["box"]
                area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) * factor
                add(counts, cls, size_group(area), "FP")
            else:
                used.add(best_index)
                add(counts, cls, size_group(float(expected[best_index]["size_640"])), "TP")
        for idx, item in enumerate(expected):
            if idx not in used:
                add(counts, cls, size_group(float(item["size_640"])), "FN")
    return counts


def aggregate(global_counts, frame_counts):
    for key, values in frame_counts.items():
        for kind in ("TP", "FP", "FN"):
            global_counts[key][kind] += values[kind]


def annotate(image, gt, predictions, confidence):
    canvas = image.copy()
    for x in gt:
        a = [int(round(v)) for v in x["box"]]
        cv2.rectangle(canvas, (a[0], a[1]), (a[2], a[3]), (0, 220, 0), 2)
    for x in predictions:
        if x["confidence"] < confidence:
            continue
        a = [int(round(v)) for v in x["box"]]
        cv2.rectangle(canvas, (a[0], a[1]), (a[2], a[3]), (0, 0, 255), 2)
        cv2.putText(canvas, f'{x["class"]} {x["confidence"]:.2f}', (a[0], max(16, a[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = root / "outputs_bdd_control/predictions.json"
    model_file = root / "yolo11s.pt"
    if not manifest.exists():
        raise FileNotFoundError("Нет outputs_bdd_control/predictions.json — сначала выполнить проверку чистого контроля")
    if not model_file.exists():
        raise FileNotFoundError("Нет yolo11s.pt — используйте веса из предыдущего шага")
    entries = json.loads(manifest.read_text(encoding="utf-8"))[:args.count]
    if not entries:
        raise RuntimeError("В predictions.json нет сцен")
    image_dir = root / "bdd_sample/data/images_100k/100k/train/shard_000"
    from ultralytics import YOLO
    model = YOLO(str(model_file))
    output = root / "outputs_bdd_control/tiling_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    summary = {variant: {conf: new_counts() for conf in THRESHOLDS} for variant in ("full", "tiled")}
    by_scene = []
    for idx, entry in enumerate(entries, 1):
        image_path = image_dir / entry["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Не найдено изображение {image_path}")
        h, w = image.shape[:2]
        full = infer_full(model, image, args.imgsz)
        tiled = infer_tiled(model, image, args.imgsz)
        scene_record = {"image": entry["image"], "gt_count": len(entry["objects_gt"]), "predictions": {"full": full, "tiled": tiled}}
        by_scene.append(scene_record)
        for name, predictions in (("full", full), ("tiled", tiled)):
            for conf in THRESHOLDS:
                aggregate(summary[name][conf], evaluate(entry["objects_gt"], predictions, w, h, conf))
        if idx <= 4:
            left = annotate(image, entry["objects_gt"], full, 0.1)
            right = annotate(image, entry["objects_gt"], tiled, 0.1)
            cv2.putText(left, "Full frame, GT green, predictions red", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(right, "Tiled, GT green, predictions red", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            side = cv2.hconcat([left, right])
            if side.shape[1] > 1800:
                ratio = 1800 / side.shape[1]
                side = cv2.resize(side, None, fx=ratio, fy=ratio)
            cv2.imwrite(str(output / f"compare_{idx:02}.jpg"), side)
        print(f'{idx}/{len(entries)}: {entry["image"]}, whole={len(full)}, tiles={len(tiled)}', flush=True)
    rows = []
    for variant, by_threshold in summary.items():
        for conf, counts in by_threshold.items():
            for cls in CLASSES:
                for group in SIZE_GROUPS:
                    stat = counts[(cls, group)]
                    tp, fp, fn = (stat[k] for k in ("TP", "FP", "FN"))
                    rows.append({"variant": variant, "imgsz": args.imgsz, "conf": conf, "class": cls, "size": group, "GT": tp + fn, "TP": tp, "FP": fp, "FN": fn, "precision": round(tp / (tp + fp), 4) if tp + fp else None, "recall": round(tp / (tp + fn), 4) if tp + fn else None})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "TUNING_PILOT_ONLY_NOT_INDEPENDENT_TEST",
        "scenes": len(entries),
        "weight_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "resolution": args.imgsz,
        "confidence_thresholds": list(THRESHOLDS),
        "tile_grid": "2x2 with 15% overlap",
        "tile_nms_iou": 0.55,
        "single_scene_decision": "none, only detector diagnostics",
        "rows": rows,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps(by_scene, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nРЕЗУЛЬТАТ (все размеры)")
    print(f'{"variant":>8} {"conf":>6} {"class":>14} {"GT":>5} {"P":>7} {"R":>7}')
    for row in rows:
        if row["size"] != "all":
            continue
        p = "н/д" if row["precision"] is None else f'{row["precision"]:.3f}'
        r = "н/д" if row["recall"] is None else f'{row["recall"]:.3f}'
        print(f'{row["variant"]:>8} {row["conf"]:6.2f} {row["class"]:>14} {row["GT"]:5} {p:>7} {r:>7}')
    print(f"\nОтчёт: {output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
