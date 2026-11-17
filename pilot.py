import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
KINDS = ("clean", "pwm", "motion", "combined")
CLASSES = {"person": 0, "car": 2, "traffic light": 9}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def on_integral(x, period, duty):
    q = math.floor(x / period)
    return q * duty * period + min(x - q * period, duty * period)


def duty_fraction(a, b, freq, duty, phase_fraction):
    if b <= a:
        raise ValueError("Invalid integration interval")
    period = 1.0 / freq
    phase = phase_fraction * period
    return np.clip((on_integral(b + phase, period, duty) - on_integral(a + phase, period, duty)) / (b - a), 0.0, 1.0)


def pzero(freq, duty, exp_s):
    return max(0.0, 1.0 - duty - exp_s * freq)


def srgb_to_linear(a):
    a = a.astype(np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(a):
    a = np.clip(a, 0.0, 1.0)
    return np.rint(255.0 * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)).astype(np.uint8)


def layer_from_bgr(img, alpha):
    rgb = srgb_to_linear(img)
    a = alpha.astype(np.float32) / 255.0
    return {"premul": rgb * a[..., None], "alpha": a}


def draw_scene(idx, cfg):
    n = cfg["image_size"]
    rng = np.random.default_rng(cfg["seed"] + idx * 104729)
    canvas = np.zeros((n, n, 3), dtype=np.uint8)
    canvas[:n // 2] = (185, 167, 133)
    canvas[n // 2:] = (107, 111, 107)
    horizon = 275 + int(rng.integers(-20, 21))
    cv2.rectangle(canvas, (0, horizon), (n, n), (96, 101, 101), -1)
    cv2.fillPoly(canvas, [np.array([[235, horizon], [405, horizon], [n, n], [0, n]], dtype=np.int32)], (82, 82, 76))
    cv2.line(canvas, (n // 2, horizon + 24), (n // 2 - 40, n), (189, 196, 194), 3)
    for k in range(14):
        x = int(rng.integers(0, n))
        y = int(rng.integers(80, 230))
        w = int(rng.integers(35, 105))
        h = int(rng.integers(50, 170))
        cv2.rectangle(canvas, (x, max(0, y - h)), (min(n - 1, x + w), y + 40), (100 + k % 5 * 7, 113, 120), -1)
    noise = rng.normal(0, 3, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    objs = []
    labels = {}
    colors = ((0, 0, 255), (0, 210, 240), (0, 240, 0))
    light_color = idx % 3
    hbox = [int(465 + rng.integers(-30, 30)), int(105 + rng.integers(-15, 15)), 38, 108]
    x, y, w, h = hbox
    blank = np.zeros_like(canvas)
    a = np.zeros((n, n), dtype=np.uint8)
    cv2.rectangle(blank, (x, y), (x + w, y + h), (27, 27, 27), -1)
    cv2.rectangle(a, (x, y), (x + w, y + h), 255, -1)
    for j in range(3):
        cv2.circle(blank, (x + w // 2, y + 20 + j * 33), 12, (7, 7, 7), -1)
    glow_mask = np.zeros((n, n), dtype=np.float32)
    cv2.circle(glow_mask, (x + w // 2, y + 20 + light_color * 33), 10, 1.0, -1)
    light = layer_from_bgr(blank, a)
    light.update({"name": "traffic light", "bbox": [x, y, x + w, y + h], "depth": 24.0 + int(idx % 4) * 8, "vx": 0.0, "vy": 0.0, "glow": glow_mask, "glow_color": srgb_to_linear(np.array(colors[light_color], dtype=np.uint8).reshape(1, 1, 3))[0, 0] * 1.7})
    objs.append(light)
    labels["traffic_light_color"] = ("red", "yellow", "green")[light_color]
    px = int(130 + rng.integers(-30, 30))
    py = int(363 + rng.integers(-15, 15))
    person = np.zeros_like(canvas)
    person_a = np.zeros((n, n), dtype=np.uint8)
    cv2.circle(person, (px, py - 37), 12, (130, 164, 187), -1)
    cv2.circle(person_a, (px, py - 37), 12, 255, -1)
    cv2.rectangle(person, (px - 13, py - 25), (px + 13, py + 21), (57, 80, 126), -1)
    cv2.rectangle(person_a, (px - 13, py - 25), (px + 13, py + 21), 255, -1)
    for direction in (-1, 1):
        cv2.line(person, (px + direction * 7, py + 21), (px + direction * 10, py + 57), (25, 26, 28), 7)
        cv2.line(person_a, (px + direction * 7, py + 21), (px + direction * 10, py + 57), 255, 7)
    person_obj = layer_from_bgr(person, person_a)
    person_obj.update({"name": "person", "bbox": [px - 15, py - 50, px + 16, py + 61], "depth": 17.0 + idx % 5 * 4, "vx": 0.5 + idx % 3 * 0.5, "vy": 0.0})
    objs.append(person_obj)
    cx = int(360 + rng.integers(-40, 40))
    cy = int(440 + rng.integers(-25, 25))
    car = np.zeros_like(canvas)
    car_a = np.zeros((n, n), dtype=np.uint8)
    cv2.rectangle(car, (cx - 64, cy - 25), (cx + 64, cy + 27), (59, 68, 169), -1)
    cv2.rectangle(car_a, (cx - 64, cy - 25), (cx + 64, cy + 27), 255, -1)
    cv2.fillPoly(car, [np.array([[cx - 36, cy - 23], [cx - 22, cy - 47], [cx + 27, cy - 47], [cx + 48, cy - 23]], dtype=np.int32)], (76, 90, 137))
    cv2.fillPoly(car_a, [np.array([[cx - 36, cy - 23], [cx - 22, cy - 47], [cx + 27, cy - 47], [cx + 48, cy - 23]], dtype=np.int32)], 255)
    for xw in (cx - 38, cx + 38):
        cv2.circle(car, (xw, cy + 25), 13, (17, 18, 20), -1)
        cv2.circle(car_a, (xw, cy + 25), 13, 255, -1)
    car_obj = layer_from_bgr(car, car_a)
    car_obj.update({"name": "car", "bbox": [cx - 67, cy - 49, cx + 67, cy + 39], "depth": 13.0 + idx % 4 * 7, "vx": -0.5, "vy": 0.0})
    objs.append(car_obj)
    objs.sort(key=lambda a: -a["depth"])
    return {"background": srgb_to_linear(canvas), "objects": objs, "labels": labels, "id": f"procedural_{idx:03}"}


def transform(obj, t_s, speed_kmh, cfg):
    n = cfg["image_size"]
    speed = speed_kmh / 3.6
    z0 = obj["depth"]
    z = max(0.5, z0 - speed * t_s)
    scale = z0 / z
    px, py = cfg["principal_point"]
    dx = cfg["focal_px"] * obj.get("vx", 0.0) * t_s / z
    dy = cfg["focal_px"] * obj.get("vy", 0.0) * t_s / z
    return np.array([[scale, 0, px * (1 - scale) + dx], [0, scale, py * (1 - scale) + dy]], dtype=np.float32)


def warp(arr, m, shape):
    return cv2.warpAffine(arr, m, shape, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def bbox_at(obj, moment, speed, cfg):
    m = transform(obj, moment, speed, cfg)
    x1, y1, x2, y2 = obj["bbox"]
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], dtype=np.float32)
    out = pts @ m.T
    return [float(out[:, 0].min()), float(out[:, 1].min()), float(out[:, 0].max()), float(out[:, 1].max())]


def make_image(scene, cfg, kind, steps=None):
    n = cfg["image_size"]
    steps = steps or cfg["integration_steps"]
    exp_s = cfg["exposure_ms"] / 1000.0
    speed = cfg["speed_kmh"] if kind in ("motion", "combined") else 0.0
    flicker = kind in ("pwm", "combined")
    result = np.zeros((n, n, 3), dtype=np.float32)
    for k in range(steps):
        a = exp_s * k / steps
        b = exp_s * (k + 1) / steps
        tau = (a + b) / 2
        duty = duty_fraction(a, b, cfg["pwm_hz"], cfg["duty_cycle"], cfg["phase_fraction"]) if flicker else 1.0
        out = scene["background"].copy()
        for obj in scene["objects"]:
            m = transform(obj, tau, speed, cfg)
            opacity = warp(obj["alpha"], m, (n, n))
            values = warp(obj["premul"], m, (n, n))
            out = out * (1.0 - opacity[..., None]) + values
            if obj["name"] == "traffic light":
                glow = warp(obj["glow"], m, (n, n))
                out += glow[..., None] * obj["glow_color"] * duty
        result += out / steps
    exposure_scale = exp_s * 1000.0 / cfg["exposure_reference_ms"]
    linear = np.clip(result * exposure_scale, 0, 1)
    return linear_to_srgb(linear), linear


def boxes_for_scene(scene, cfg, kind):
    t = cfg["exposure_ms"] / 2000.0
    speed = cfg["speed_kmh"] if kind in ("motion", "combined") else 0.0
    return [{"class": obj["name"], "bbox": bbox_at(obj, t, speed, cfg)} for obj in scene["objects"]]


def light_contrast(linear, gt, cfg):
    lamp = next((item for item in gt if item["class"] == "traffic light"), None)
    if lamp is None:
        return None
    x1, y1, x2, y2 = [int(round(c)) for c in lamp["bbox"]]
    n = cfg["image_size"]
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, n - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = linear[y1:y2, x1:x2]
    bright = float(np.quantile(np.max(patch, axis=2), 0.98))
    dark = float(np.quantile(np.max(patch, axis=2), 0.30))
    return (bright - dark) / max(dark, 0.05)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2-x1)*max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1])
    return intersection/max(1e-9, area_a+area_b-intersection)


def load_yolo(weights, use_model):
    if not use_model:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Установите зависимости: pip install -r requirements.txt") from e
    path = Path(weights)
    if not path.is_file():
        raise SystemExit(f"Нет весов: {weights}. Скачайте yolo11n.pt из официального выпуска v8.3.0")
    return YOLO(str(path))


def predict(model, image, cfg):
    if model is None:
        return []
    result = model.predict(source=image, imgsz=cfg["image_size"], conf=cfg["yolo_confidence"], iou=cfg["yolo_nms_iou"], device="cpu", verbose=False)[0]
    items = []
    for b in result.boxes:
        class_idx = int(b.cls[0])
        if class_idx in CLASSES.values():
            items.append({"class": next(k for k, v in CLASSES.items() if v == class_idx), "bbox": [float(x) for x in b.xyxy[0].tolist()], "confidence": float(b.conf[0])})
    return items


def predict_light_color(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 9 or y2 - y1 < 24:
        return "unknown"
    patch = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    scores = []
    for k, color in enumerate(("red", "yellow", "green")):
        top = int(k * patch.shape[0] / 3)
        bottom = int((k + 1) * patch.shape[0] / 3)
        region = hsv[top:bottom]
        base = (region[:, :, 1] >= 80) & (region[:, :, 2] >= 80)
        if color == "red":
            hue = (region[:, :, 0] <= 10) | (region[:, :, 0] >= 170)
        elif color == "yellow":
            hue = (region[:, :, 0] >= 18) & (region[:, :, 0] <= 38)
        else:
            hue = (region[:, :, 0] >= 40) & (region[:, :, 0] <= 85)
        scores.append(int(np.sum(base & hue)))
    ranking = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    if ranking[0][1] < 4 or ranking[0][1] < 1.2 * max(1, ranking[1][1]):
        return "unknown"
    return ("red", "yellow", "green")[ranking[0][0]]


def light_result(gt, predictions, image, expected, threshold):
    target = next(v for v in gt if v["class"] == "traffic light")
    options = [(iou(target["bbox"], p["bbox"]), p) for p in predictions if p["class"] == "traffic light"]
    if not options:
        return {"status": "missed", "predicted_color": None, "correct_color": False}
    score, candidate = max(options, key=lambda v: v[0])
    if score < threshold:
        return {"status": "missed", "predicted_color": None, "correct_color": False}
    color = predict_light_color(image, candidate["bbox"])
    return {"status": "found", "predicted_color": color, "correct_color": color == expected}


def evaluate(gt, pred, threshold):
    out = {}
    for name in CLASSES:
        reference = [x for x in gt if x["class"] == name]
        predicted = [x for x in pred if x["class"] == name]
        used = set()
        tp = 0
        for obj in reference:
            candidates = [(iou(obj["bbox"], p["bbox"]), j) for j, p in enumerate(predicted) if j not in used]
            if candidates:
                score, index = max(candidates)
                if score >= threshold:
                    used.add(index)
                    tp += 1
        out[name] = {"TP": tp, "FN": len(reference)-tp, "FP": len(predicted)-tp}
    return out


def demo(args, cfg):
    root = ROOT / "outputs" / "preview" if args.preview_size else ROOT / "outputs"
    root.mkdir(exist_ok=True, parents=True)
    model = load_yolo(ROOT / cfg["yolo_weights"], args.yolo)
    nscenes = args.scenes or cfg["scene_count"]
    manifest = []
    predictions = []
    counts = {v: {name: {"TP": 0, "FP": 0, "FN": 0} for name in CLASSES} for v in KINDS}
    per_scene = []
    for sid in range(nscenes):
        scene = draw_scene(sid, cfg)
        effective_cfg = cfg
        if args.preview_size:
            scene, effective_cfg = downsample_scene(scene, cfg, args.preview_size)
        for kind in KINDS:
            image, linear = make_image(scene, effective_cfg, kind)
            gt = boxes_for_scene(scene, effective_cfg, kind)
            out = root / f"scene_{sid:03}_{kind}.png"
            cv2.imwrite(str(out), image)
            record = {"scene": scene["id"], "variant": kind, "file": out.relative_to(ROOT).as_posix(), "sha256": sha256(out), "gt": gt, "signal": scene["labels"]["traffic_light_color"], "contrast": light_contrast(linear, gt, effective_cfg)}
            manifest.append(record)
            pred = predict(model, image, effective_cfg)
            if model is not None:
                signal_eval = light_result(gt, pred, image, scene["labels"]["traffic_light_color"], effective_cfg["match_iou"])
                metrics = evaluate(gt, pred, effective_cfg["match_iou"])
                for name in CLASSES:
                    for key in ("TP", "FP", "FN"):
                        counts[kind][name][key] += metrics[name][key]
                per_scene.append({"scene": scene["id"], "variant": kind, "counts": metrics})
            predictions.append({"scene": scene["id"], "variant": kind, "predictions": pred, "signal_evaluation": signal_eval if model is not None else None, "detector_executed": model is not None})
    with open(root / "manifest.json", "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(root / "predictions.json", "w", encoding="utf8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    with open(root / "by_scene.json", "w", encoding="utf8") as f:
        json.dump(per_scene, f, indent=2, ensure_ascii=False)
    with open(root / "manifest_sources.json", "w", encoding="utf8") as f:
        json.dump({"source_type": "procedural technical mock-up", "real_bdd100k_sources": [], "notes": "Эти изображения не являются проверкой качества детектора на реальных дорожных сценах", "config_sha256": sha256(ROOT / "config.json"), "weights_sha256": sha256(ROOT / cfg["yolo_weights"]) if model else None}, f, indent=2, ensure_ascii=False)
    report = {"status": "TECHNICAL_TEST_ONLY", "preview_size": args.preview_size, "scenes": nscenes, "variants": len(KINDS), "generated_images": len(manifest), "pwm_no_overlap_analytic": pzero(cfg["pwm_hz"], cfg["duty_cycle"], cfg["exposure_ms"] / 1000.0), "yolo_executed": model is not None, "class_counts": counts if model else None, "valid_for_research_claims": False, "reason": "Процедурные фигуры не являются BDD100K или набором реалистичных объектов. При отсутствии YOLO качество детекции также не измерено."}
    with open(root / "technical_report.json", "w", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def downsample_scene(scene, cfg, size):
    original = cfg["image_size"]
    if size == original:
        return scene, cfg
    ratio = size / original
    settings = dict(cfg)
    settings["image_size"] = size
    settings["focal_px"] = cfg["focal_px"] * ratio
    settings["principal_point"] = [p * ratio for p in cfg["principal_point"]]
    converted = {"background": cv2.resize(scene["background"], (size, size), interpolation=cv2.INTER_AREA), "labels": scene["labels"], "id": scene["id"], "objects": []}
    for obj in scene["objects"]:
        new = dict(obj)
        new["premul"] = cv2.resize(obj["premul"], (size, size), interpolation=cv2.INTER_AREA)
        new["alpha"] = cv2.resize(obj["alpha"], (size, size), interpolation=cv2.INTER_AREA)
        new["bbox"] = [x * ratio for x in obj["bbox"]]
        if "glow" in obj:
            new["glow"] = cv2.resize(obj["glow"], (size, size), interpolation=cv2.INTER_AREA)
        converted["objects"].append(new)
    return converted, settings


def converge(cfg, scenes):
    output = ROOT / "outputs" / "integration_convergence.csv"
    steps = cfg["convergence_steps"]
    rows = []
    for sid in range(scenes):
        scene = draw_scene(sid, cfg)
        scene, local_cfg = downsample_scene(scene, cfg, min(cfg["image_size"], cfg.get("convergence_preview_size", 256)))
        for label, velocity, exposure in (("short_slow", 0, 2), ("long_fast", 90, 10)):
            local = dict(local_cfg)
            local.update(speed_kmh=velocity, exposure_ms=exposure)
            data = []
            for count in steps:
                _, linear = make_image(scene, local, "combined", count)
                gt = boxes_for_scene(scene, local, "combined")
                data.append((count, linear, light_contrast(linear, gt, local)))
            for (a, im1, c1), (b, im2, c2) in zip(data, data[1:]):
                rows.append({"scene": sid, "case": label, "image_size": local["image_size"], "steps_a": a, "steps_b": b, "mean_abs_delta": float(np.abs(im1-im2).mean()), "contrast_delta": abs(c1-c2) if c1 is not None and c2 is not None else None})
    with open(output, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сходимость сохранена: {output}")
    print(f"Максимальная средняя ошибка 64→128: {max(r['mean_abs_delta'] for r in rows if r['steps_a']==64):.5f}")
    print(f"Максимальное изменение контраста 64→128: {max(r['contrast_delta'] for r in rows if r['steps_a']==64):.5f}")


def validate_formulas():
    checks = []
    for f in (100, 200):
        for d in (0.2, 0.5):
            for ms in (2, 4, 8, 10):
                phase_grid = np.linspace(0, 1, 5000, endpoint=False)
                numeric = np.mean([duty_fraction(0, ms/1000, f, d, float(phi)) < 1e-10 for phi in phase_grid])
                expected = pzero(f, d, ms/1000)
                if abs(numeric-expected) >= .002:
                    raise AssertionError((f, d, ms, numeric, expected))
                checks.append([f, d, ms, expected, numeric])
    out = ROOT / "outputs" / "formula_verification.csv"
    with open(out, "w", newline="", encoding="utf8") as file:
        writer = csv.writer(file)
        writer.writerow(["freq_hz", "duty", "exposure_ms", "analytic_p0", "numerical_p0"])
        writer.writerows(checks)
    print(f"Формулы проверены: {len(checks)} режимов, максимальное расхождение < 0.002")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("formula", "smoke", "pilot", "convergence"))
    parser.add_argument("--scenes", type=int)
    parser.add_argument("--preview-size", type=int, choices=[256,320,480])
    parser.add_argument("--yolo", action="store_true")
    args = parser.parse_args()
    if args.yolo and args.preview_size:
        parser.error("YOLO-пилот должен выполняться без --preview-size")
    with open(ROOT / "config.json", encoding="utf8") as f:
        cfg = json.load(f)
    (ROOT / "outputs").mkdir(exist_ok=True)
    if args.mode == "formula":
        validate_formulas()
    elif args.mode == "convergence":
        converge(cfg, args.scenes or cfg["convergence_scenes"])
    else:
        if args.mode == "smoke":
            args.scenes = args.scenes or 2
        else:
            args.scenes = args.scenes or cfg["scene_count"]
        demo(args, cfg)


if __name__ == "__main__":
    main()
