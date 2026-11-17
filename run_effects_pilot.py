import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ('person', 'car', 'traffic light')
CLASS_IDS = (0, 2, 9)
VARIANTS = ('clean', 'pwm', 'motion', 'combined')


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def linearize(bgr):
    a = bgr.astype(np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def encode(a):
    a = np.clip(a, 0, 1)
    return np.rint(255 * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)).astype(np.uint8)


def on_integral(value, period, duty):
    times = math.floor(value / period)
    return times * period * duty + min(value - times * period, period * duty)


def pwm_fraction(a, b, frequency, duty, phase):
    period = 1.0 / frequency
    shift = phase * period
    return max(0.0, min(1.0, (on_integral(b + shift, period, duty) - on_integral(a + shift, period, duty)) / (b - a)))


def match_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(right - left, 0) * max(bottom - top, 0)
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    return overlap / max(area_a + area_b - overlap, 1e-9)


def box_scale(box, sx, sy):
    return [float(box[0] * sx), float(box[1] * sy), float(box[2] * sx), float(box[3] * sy)]


def projection_scale(seconds, speed, depth):
    return depth / max(0.5, depth - (speed / 3.6) * seconds)


def scale_box(box, scale, width, height):
    cx, cy = width / 2, height / 2
    return [cx + (box[0] - cx) * scale, cy + (box[1] - cy) * scale, cx + (box[2] - cx) * scale, cy + (box[3] - cy) * scale]


def select_light(scene, sx, sy):
    options = []
    for obj in scene['objects_gt']:
        if obj['class'] != 'traffic light':
            continue
        color = str(obj.get('traffic_light_color') or '').lower()
        if color not in ('red', 'yellow', 'green'):
            continue
        box = box_scale(obj['box'], sx, sy)
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        options.append((area, box, color))
    return max(options, default=None, key=lambda x: x[0])


def prep_scene(scene, image, width):
    oh, ow = image.shape[:2]
    height = int(round(oh * width / ow))
    bgr = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    sx, sy = width / ow, height / oh
    gt = [{'class': entry['class'], 'box': box_scale(entry['box'], sx, sy), 'traffic_light_color': entry.get('traffic_light_color')} for entry in scene['objects_gt']]
    target = select_light(scene, sx, sy)
    signal = np.zeros_like(bgr, dtype=np.float32)
    background = bgr.copy()
    light_info = None
    light_mask = np.zeros((height, width), dtype=np.uint8)
    if target is not None:
        area, box, color = target
        x1, y1, x2, y2 = box
        ww, hh = max(0, x2 - x1), max(0, y2 - y1)
        if ww >= 5 and hh >= 12:
            j = ('red', 'yellow', 'green').index(color)
            center_x = int(round((x1 + x2) / 2))
            center_y = int(round(y1 + (j + 0.5) * hh / 3))
            rad = max(2, int(round(min(ww / 4.2, hh / 9))))
            cv2.circle(light_mask, (center_x, center_y), rad, 255, -1)
            background = cv2.inpaint(background, light_mask, 3, cv2.INPAINT_TELEA)
            palette = {'red': (0, 0, 255), 'yellow': (0, 225, 255), 'green': (0, 255, 0)}
            emission_bgr = np.array(palette[color], dtype=np.uint8).reshape(1, 1, 3)
            emission = linearize(emission_bgr)[0, 0] * 1.3
            signal = (light_mask.astype(np.float32) / 255.0)[..., None] * emission
            light_info = {'gt_bbox': box, 'color': color, 'center': [center_x, center_y], 'radius': rad, 'synthesized_active_signal': True}
    return {'bgr': bgr, 'base': linearize(background), 'emission': signal, 'gt': gt, 'light': light_info, 'mask': light_mask, 'size': [width, height]}


def warp(a, scale, width, height):
    if scale == 1:
        return a
    cx, cy = width / 2, height / 2
    matrix = np.array([[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)]], dtype=np.float32)
    return cv2.warpAffine(a, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def simulate(scene, kind, args, steps=None):
    width, height = scene['size']
    t = args.exposure_ms / 1000.0
    steps = steps or args.steps
    speed = args.speed_kmh if kind in ('motion', 'combined') else 0.0
    pwm = kind in ('pwm', 'combined')
    total = np.zeros_like(scene['base'], dtype=np.float32)
    for index in range(steps):
        a, b = t * index / steps, t * (index + 1) / steps
        mid = (a + b) / 2
        scale = projection_scale(mid, speed, args.depth_m)
        base = warp(scene['base'], scale, width, height)
        led = warp(scene['emission'], scale, width, height)
        fraction = pwm_fraction(a, b, args.pwm_hz, args.duty, args.phase) if pwm else 1.0
        total += (base + led * fraction) / steps
    total = np.clip(total * args.exposure_ms / args.reference_exposure_ms, 0.0, 1.0)
    scale_middle = projection_scale(t / 2, speed, args.depth_m)
    gt = [{**entry, 'box': scale_box(entry['box'], scale_middle, width, height)} for entry in scene['gt']]
    return encode(total), total, gt


def predictions_from_ultralytics(result, model, ox=0, oy=0):
    predictions = []
    for box in result.boxes:
        cls = model.names[int(box.cls.item())]
        if cls not in NAMES:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        predictions.append({'class': cls, 'score': float(box.conf.item()), 'box': [x1 + ox, y1 + oy, x2 + ox, y2 + oy]})
    return predictions


def tile_regions(width, height):
    tile_w = min(width, int(round(width / 1.85)))
    tile_h = min(height, int(round(height / 1.85)))
    return [(x, y, x + tile_w, y + tile_h) for y in sorted({0, height - tile_h}) for x in sorted({0, width - tile_w})]


def nms(predictions, threshold=0.55):
    if not predictions:
        return []
    boxes = [[p['box'][0], p['box'][1], p['box'][2] - p['box'][0], p['box'][3] - p['box'][1]] for p in predictions]
    scores = [p['score'] for p in predictions]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, threshold)
    return [predictions[int(i)] for i in indices] if indices is not None and len(indices) else []


def infer_hybrid(model, image, args):
    whole = model.predict(image, imgsz=args.imgsz, conf=0.05, iou=0.7, classes=list(CLASS_IDS), verbose=False, device='cpu')[0]
    persons_cars = [p for p in predictions_from_ultralytics(whole, model) if p['class'] in ('person', 'car') and p['score'] >= args.conf]
    height, width = image.shape[:2]
    windows = tile_regions(width, height)
    crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
    results = model.predict(crops, imgsz=args.imgsz, conf=0.05, iou=0.7, classes=list(CLASS_IDS), verbose=False, device='cpu')
    lights = []
    for result, (x1, y1, _, _) in zip(results, windows):
        lights.extend([p for p in predictions_from_ultralytics(result, model, x1, y1) if p['class'] == 'traffic light' and p['score'] >= args.conf])
    return persons_cars + nms(lights)


def evaluate(gt, pred):
    out = {}
    for cls in NAMES:
        expected = [g for g in gt if g['class'] == cls]
        found = sorted((p for p in pred if p['class'] == cls), key=lambda p: -p['score'])
        used = set()
        tp = 0
        for p in found:
            ranked = sorted(((match_iou(p['box'], g['box']), idx) for idx, g in enumerate(expected) if idx not in used), reverse=True)
            if ranked and ranked[0][0] >= 0.5:
                used.add(ranked[0][1])
                tp += 1
        out[cls] = {'TP': tp, 'FP': len(found) - tp, 'FN': len(expected) - tp}
    return out


def classify_color(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 5 or y2 - y1 < 12:
        return 'unknown'
    crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    scores = []
    for j, color in enumerate(('red', 'yellow', 'green')):
        a, b = int(j * crop.shape[0] / 3), int((j + 1) * crop.shape[0] / 3)
        section = crop[a:b]
        if len(section) == 0:
            scores.append(0)
            continue
        hue, sat, val = section[:, :, 0], section[:, :, 1], section[:, :, 2]
        if color == 'red':
            good = (hue <= 10) | (hue >= 170)
        elif color == 'yellow':
            good = (hue >= 18) & (hue <= 38)
        else:
            good = (hue >= 40) & (hue <= 85)
        scores.append(int(np.sum(good & (sat >= 80) & (val >= 80))))
    ranked = sorted(enumerate(scores), key=lambda row: -row[1])
    if ranked[0][1] < 4 or ranked[0][1] < 1.2 * max(1, ranked[1][1]):
        return 'unknown'
    return ('red', 'yellow', 'green')[ranked[0][0]]


def light_evaluation(image, gt, pred, light):
    if light is None:
        return {'status': 'signal_not_synthesized', 'matched': False, 'color_pred': None, 'color_gt': None}
    reference = max((x for x in gt if x['class'] == 'traffic light' and x.get('traffic_light_color') == light['color']), default=None, key=lambda v: match_iou(light['gt_bbox'], v['box']))
    if reference is None:
        return {'status': 'gt_missing', 'matched': False, 'color_pred': None, 'color_gt': light['color']}
    candidates = sorted(((match_iou(reference['box'], p['box']), p) for p in pred if p['class'] == 'traffic light'), key=lambda x: -x[0])
    if not candidates or candidates[0][0] < 0.5:
        return {'status': 'missed', 'matched': False, 'color_pred': None, 'color_gt': light['color']}
    predicted = classify_color(image, candidates[0][1]['box'])
    return {'status': 'correct' if predicted == light['color'] else 'wrong_or_unknown', 'matched': True, 'color_pred': predicted, 'color_gt': light['color']}


def draw_predictions(image, gt, pred):
    out = image.copy()
    for obj in gt:
        xy = [int(round(v)) for v in obj['box']]
        cv2.rectangle(out, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 1)
    for obj in pred:
        xy = [int(round(v)) for v in obj['box']]
        cv2.rectangle(out, (xy[0], xy[1]), (xy[2], xy[3]), (0, 0, 255), 1)
    return out


def scene_image(scene_record, root):
    image_path = root / 'bdd_sample/data/images_100k/100k/train/shard_000' / scene_record['image']
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Отсутствует скачанный кадр: {image_path}')
    return image_path, image


def convergence(args, entries, root, output):
    model = None
    if args.yolo:
        from ultralytics import YOLO
        weights = root / 'yolo11s.pt'
        if not weights.exists():
            raise FileNotFoundError(f'Нет модели {weights}')
        model = YOLO(str(weights))
    rows = []
    for scene_id, record in enumerate(entries[:args.count], 1):
        _, photo = scene_image(record, root)
        prepared = prep_scene(record, photo, args.render_width)
        for name, speed, exp in [('short_slow', 0, 2), ('long_fast', 90, 10), ('medium_fast', 60, 8)]:
            local = argparse.Namespace(**vars(args))
            local.speed_kmh, local.exposure_ms = speed, exp
            results = {}
            detection = {}
            for step in (16, 32, 64, 128):
                rendered, lin, gt = simulate(prepared, 'combined', local, steps=step)
                results[step] = lin
                if model is not None:
                    detection[step] = evaluate(gt, infer_hybrid(model, rendered, args))
            for left, right in ((16, 32), (32, 64), (64, 128)):
                rows.append({'scene': record['image'], 'condition': name, 'width': args.render_width, 'steps': f'{left}-{right}', 'mean_abs_pixel_diff': round(float(np.mean(np.abs(results[right] - results[left]))), 8), 'max_abs_pixel_diff': round(float(np.max(np.abs(results[right] - results[left]))), 8), 'detection_counts_changed': int(detection[left] != detection[right]) if model is not None else ''})
        print(f'Сходимость: {scene_id}/{min(args.count, len(entries))}', flush=True)
    path = output / 'integration_convergence.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {'status': 'NUMERICAL_CONVERGENCE_ONLY', 'scenes': min(args.count, len(entries)), 'render_width': args.render_width, 'mean_diff_64_128_max': max(r['mean_abs_pixel_diff'] for r in rows if r['steps'] == '64-128'), 'max_diff_64_128_max': max(r['max_abs_pixel_diff'] for r in rows if r['steps'] == '64-128'), 'yolo_outputs_compared': bool(model), 'scope': 'Тест численной сходимости дополнительного воздействия на готовые фотографии, не проверка физической достоверности экспозиции'}
    (output / 'convergence_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def experiment(args, entries, root, output):
    model = None
    model_path = root / 'yolo11s.pt'
    if args.yolo:
        if not model_path.exists():
            raise FileNotFoundError(f'Нет модели {model_path}')
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    samples = []
    totals = {kind: {cls: {'TP': 0, 'FP': 0, 'FN': 0} for cls in NAMES} for kind in VARIANTS}
    signal_stats = {kind: {'correct': 0, 'wrong_or_unknown': 0, 'missed': 0, 'not_synthesized': 0} for kind in VARIANTS}
    output_imgs = output / 'images'
    output_imgs.mkdir(exist_ok=True)
    for idx, record in enumerate(entries[:args.count], 1):
        image_path, image = scene_image(record, root)
        prepared = prep_scene(record, image, args.render_width)
        scene_out = {'image': record['image'], 'source_sha256': checksum(image_path), 'original_source_sha256': record.get('sha256_image'), 'source_label_sha256': record.get('sha256_label'), 'light': prepared['light'], 'variants': {}}
        for kind in VARIANTS:
            altered, lin, gt = simulate(prepared, kind, args)
            target = output_imgs / f'{idx:02d}_{kind}.png'
            cv2.imwrite(str(target), altered)
            pred = infer_hybrid(model, altered, args) if model is not None else []
            counts = evaluate(gt, pred) if model is not None else None
            color_eval = light_evaluation(altered, gt, pred, prepared['light']) if model is not None else None
            if counts is not None:
                for cls in NAMES:
                    for key in ('TP', 'FP', 'FN'):
                        totals[kind][cls][key] += counts[cls][key]
                if prepared['light'] is None:
                    signal_stats[kind]['not_synthesized'] += 1
                else:
                    signal_stats[kind][color_eval['status']] += 1
                if idx <= 3:
                    cv2.imwrite(str(output_imgs / f'{idx:02d}_{kind}_pred.png'), draw_predictions(altered, gt, pred))
            scene_out['variants'][kind] = {'image': target.name, 'sha256': checksum(target), 'gt': gt, 'predictions': pred, 'counts': counts, 'light_prediction': color_eval}
        samples.append(scene_out)
        print(f'{idx}/{min(args.count, len(entries))}: {record["image"]}, light_synthesized={prepared["light"] is not None}, yolo={bool(model)}', flush=True)
    report = {'status': 'TECHNICAL_PILOT_NOT_INDEPENDENT_RESEARCH', 'count': len(samples), 'model': str(model_path) if model else None, 'weights_sha256': checksum(model_path) if model else None, 'source': 'BDD100K photographs with approximate simulated extra effects', 'render_width': args.render_width, 'mode': 'uniform assigned depth, no per-object optical motion, no physical source exposure', 'parameters': {'exposure_ms': args.exposure_ms, 'speed_kmh': args.speed_kmh, 'depth_m': args.depth_m, 'phase': args.phase, 'frequency': args.pwm_hz, 'duty': args.duty, 'integration_steps': args.steps, 'model_imgsz': args.imgsz, 'confidence': args.conf}, 'four_variants': list(VARIANTS), 'config_file_sha256': checksum(root / args.config) if args.config and (root / args.config).exists() else None, 'totals': totals if model else None, 'color_totals': signal_stats if model else None, 'limitations': ['Готовые JPEG уже прошли неизвестную экспозицию и обработку', 'Использована единая условная глубина и поступательное движение камеры', 'Светофорный импульс синтезируется по вручную не проверенной области GT', 'Нет раздельных реальных глубин и движения пешеходов/автомобилей', 'Результаты не являются проверкой основной гипотезы']}
    for kind in VARIANTS:
        if model:
            for cls in NAMES:
                c = report['totals'][kind][cls]
                c['precision'] = round(c['TP'] / (c['TP'] + c['FP']), 4) if c['TP'] + c['FP'] else None
                c['recall'] = round(c['TP'] / (c['TP'] + c['FN']), 4) if c['TP'] + c['FN'] else None
    (output / 'pilot_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'predictions_manifest.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(output / 'pilot_report.json'), 'scenes': len(samples), 'yolo': bool(model), 'color_totals': report['color_totals'], 'totals': report['totals']}, ensure_ascii=False, indent=2))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='next_pilot_config.json')
    preliminary, _ = pre.parse_known_args()
    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument('--mode', choices=('convergence', 'effects'), default='effects')
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--yolo', action='store_true')
    parser.add_argument('--render-width', type=int, default=640)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--exposure-ms', type=float, default=8.0)
    parser.add_argument('--reference-exposure-ms', type=float, default=8.0)
    parser.add_argument('--speed-kmh', type=float, default=60.0)
    parser.add_argument('--depth-m', type=float, default=40.0)
    parser.add_argument('--pwm-hz', type=float, default=100.0)
    parser.add_argument('--duty', type=float, default=0.2)
    parser.add_argument('--phase', type=float, default=0.13)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.10)
    settings_path = Path(__file__).resolve().parent / preliminary.config
    if settings_path.exists():
        data = json.loads(settings_path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            parser.error('Конфигурация должна быть JSON-объектом')
        valid = {act.dest for act in parser._actions}
        unknown = sorted(set(data) - valid)
        if unknown:
            parser.error(f'Неизвестные параметры в конфигурации: {unknown}')
        parser.set_defaults(**data)
    args = parser.parse_args()
    if not 0 < args.duty <= 1 or not 0 <= args.phase < 1 or args.count < 1 or args.steps < 1 or args.render_width < 128:
        parser.error('Некорректные параметры генерации')
    root = Path(__file__).resolve().parent
    manifest = root / 'outputs_bdd_control/predictions.json'
    if not manifest.exists():
        raise FileNotFoundError('Отсутствует outputs_bdd_control/predictions.json')
    entries = json.loads(manifest.read_text(encoding='utf-8'))
    if not entries:
        raise RuntimeError('Пустой список сцен')
    output = root / 'outputs_bdd_control/effects_pilot'
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'convergence':
        convergence(args, entries, root, output)
    else:
        experiment(args, entries, root, output)


if __name__ == '__main__':
    main()
