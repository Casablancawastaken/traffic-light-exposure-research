import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
REPO = "dronefreak/BDD100K"
SPLIT = "train"
SHARD = "shard_000"
CLASSES = ("person", "car", "traffic light")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_annotation(path, width, height):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    objects = []
    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            category = obj.get("category")
            box = obj.get("box2d")
            if category not in CLASSES or not isinstance(box, dict):
                continue
            try:
                x1 = min(width, max(0.0, float(box["x1"])))
                y1 = min(height, max(0.0, float(box["y1"])))
                x2 = min(width, max(0.0, float(box["x2"])))
                y2 = min(height, max(0.0, float(box["y2"])))
            except (TypeError, KeyError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append({
                "class": category,
                "box": [x1, y1, x2, y2],
                "traffic_light_color": obj.get("attributes", {}).get("trafficLightColor") if category == "traffic light" else None,
                "size_640": round((x2 - x1) * (y2 - y1) * (640.0 / max(width, height)) ** 2, 2),
            })
        break
    return objects


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return overlap / max(1e-12, area_a + area_b - overlap)


def match_scene(gt, predictions, threshold=0.5):
    counts = {c: {"TP": 0, "FP": 0, "FN": 0} for c in CLASSES}
    matches = []
    for klass in CLASSES:
        expected = [g for g in gt if g["class"] == klass]
        predicted = sorted([p for p in predictions if p["class"] == klass], key=lambda p: -p["conf"])
        claimed = set()
        for p in predicted:
            best_idx = None
            best_iou = threshold
            for idx, g in enumerate(expected):
                if idx in claimed:
                    continue
                score = iou(p["box"], g["box"])
                if score >= best_iou:
                    best_idx = idx
                    best_iou = score
            if best_idx is None:
                counts[klass]["FP"] += 1
            else:
                counts[klass]["TP"] += 1
                claimed.add(best_idx)
                matches.append({"class": klass, "status": "TP", "iou": round(best_iou, 4)})
        counts[klass]["FN"] = len(expected) - len(claimed)
    return counts, matches


def visualize(image, gt, pred):
    output = image.copy()
    for obj in gt:
        a = tuple(round(v) for v in obj["box"])
        cv2.rectangle(output, a[:2], a[2:], (0, 220, 0), 2)
    for obj in pred:
        a = tuple(round(v) for v in obj["box"])
        cv2.rectangle(output, a[:2], a[2:], (0, 0, 255), 2)
        cv2.putText(output, f'{obj["class"]} {obj["conf"]:.2f}', (a[0], max(12, a[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--shard", default="shard_000")
    args = parser.parse_args()
    if args.count < 1 or args.max_candidates < args.count:
        parser.error("Количество кандидатов должно быть не меньше числа сцен")

    from huggingface_hub import HfApi, hf_hub_download
    from ultralytics import YOLO

    root = ROOT / "bdd_sample"
    output_dir = ROOT / "outputs_bdd_control"
    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revision = api.repo_info(repo_id=REPO, repo_type="dataset").sha
    label_folder = f"data/labels/100k/{SPLIT}/{args.shard}"
    image_folder = f"data/images_100k/100k/{SPLIT}/{args.shard}"
    print(f"Проверка структуры BDD100K: {label_folder}", flush=True)
    files = sorted(entry.path for entry in api.list_repo_tree(repo_id=REPO, repo_type="dataset", path_in_repo=label_folder, revision=revision) if entry.path.endswith(".json"))
    if not files:
        raise RuntimeError("Разметка JSON не найдена в выбранном разделе репозитория")
    print(f"Найдено файлов разметки: {len(files)}. Проверяем не более {args.max_candidates}.", flush=True)

    def fetch(repo_path):
        return Path(hf_hub_download(repo_id=REPO, repo_type="dataset", revision=revision, filename=repo_path, local_dir=root))

    weights = ROOT / "yolo11n.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Не найдены веса: {weights}")
    model = YOLO(str(weights))
    model_names = model.names
    totals = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    details = []
    selected = 0
    checked = 0
    failures = []

    for path in files[:args.max_candidates]:
        checked += 1
        try:
            label_file = fetch(path)
            data = json.loads(label_file.read_text(encoding="utf-8"))
            object_types = {obj.get("category") for frame in data.get("frames", []) for obj in frame.get("objects", []) if "box2d" in obj}
            if not ({"traffic light"} <= object_types and ({"person", "car"} & object_types)):
                continue
            image_name = Path(path).stem + ".jpg"
            image_path = fetch(f"{image_folder}/{image_name}")
            frame = cv2.imread(str(image_path))
            if frame is None:
                failures.append(f"Изображение не прочитано: {image_name}")
                continue
            h, w = frame.shape[:2]
            gt = read_annotation(label_file, w, h)
            results = model.predict(frame, imgsz=640, conf=0.25, iou=0.70, device="cpu", verbose=False)[0]
            pred = []
            for box in results.boxes:
                cls = model_names[int(box.cls.item())]
                if cls in CLASSES:
                    pred.append({"class": cls, "conf": round(float(box.conf.item()), 6), "box": [round(float(x), 3) for x in box.xyxy[0].cpu().tolist()]})
            counts, matches = match_scene(gt, pred)
        except Exception as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
            if len(failures) >= 10:
                raise RuntimeError("Много ошибок загрузки; первые примеры: " + " | ".join(failures[:3])) from exc
            continue

        for klass in CLASSES:
            for name in ("TP", "FP", "FN"):
                totals[klass][name] += counts[klass][name]
        selected += 1
        details.append({"image": image_name, "sha256_image": sha256(image_path), "sha256_label": sha256(label_file), "objects_gt": gt, "predictions": pred, "counts": counts, "matches": matches})
        if selected <= 5:
            cv2.imwrite(str(output_dir / f"sample_{selected:02}.jpg"), visualize(frame, gt, pred))
        print(f"{selected}/{args.count}: {image_name} | " + " | ".join(f"{klass}: TP={counts[klass]['TP']} FN={counts[klass]['FN']} FP={counts[klass]['FP']}" for klass in CLASSES), flush=True)
        if selected >= args.count:
            break

    report = {
        "status": "REAL_IMAGE_CLEAN_CONTROL_ONLY",
        "source": REPO,
        "source_layout": "raw BDD100K labels JSON and images_100k JPG",
        "repo_revision": revision,
        "split": f"{SPLIT}/{args.shard}",
        "requested_scenes": args.count,
        "evaluated_scenes": selected,
        "scanned_annotations": checked,
        "model": str(weights),
        "model_sha256": sha256(weights),
        "parameters": {"imgsz": 640, "confidence": 0.25, "nms_iou": 0.70, "match_iou": 0.50},
        "counts": dict(totals),
        "metrics": {},
        "retrieval_errors": failures[:10],
        "research_scope": "Только оценка обнаружения на исходных фотографиях; не моделирует ШИМ или экспозицию и не оценивает цвета светофора",
    }
    for klass in CLASSES:
        count = totals[klass]
        tp, fp, fn = count["TP"], count["FP"], count["FN"]
        report["metrics"][klass] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None, "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated_scenes": selected, "scanned_annotations": checked, "counts": report["counts"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    print(f"Отчёт: {output_dir / 'report.json'}")
    if selected < args.count:
        raise SystemExit("Подходящих кадров недостаточно в проверенных файлах. Можно повысить --max-candidates.")


if __name__ == "__main__":
    main()
