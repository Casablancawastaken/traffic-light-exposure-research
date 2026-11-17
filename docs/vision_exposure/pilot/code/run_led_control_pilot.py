import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import run_effects_pilot as core

COLORS = ('red', 'yellow', 'green')
VARIANTS = ('clean', 'pwm', 'motion', 'combined')
CLASSES = ('person', 'car', 'traffic light')


def pick_placement(gt, width, height, bw, bh):
    candidates = []
    for xf in (0.79, 0.12, 0.62, 0.45):
        for yf in (0.08, 0.22, 0.36):
            left = int(width * xf)
            top = int(height * yf)
            left = max(3, min(left, width - bw - 3))
            top = max(3, min(top, height - bh - 3))
            box = [left, top, left + bw, top + bh]
            penalty = sum(core.match_iou(box, obj['box']) * (10 if obj['class'] == 'traffic light' else 1) for obj in gt)
            candidates.append((penalty, box))
    return min(candidates, key=lambda p: p[0])[1]


def controlled_scene(record, image, width, color):
    oh, ow = image.shape[:2]
    height = int(round(oh * width / ow))
    bgr = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    sx, sy = width / ow, height / oh
    gt = [{'class': obj['class'], 'box': core.box_scale(obj['box'], sx, sy), 'traffic_light_color': obj.get('traffic_light_color')} for obj in record['objects_gt']]
    housing_w = max(26, int(round(width * 0.058)))
    housing_h = min(height - 20, int(round(housing_w * 2.75)))
    left, top, right, bottom = pick_placement(gt, width, height, housing_w, housing_h)
    cv2.rectangle(bgr, (left - 2, top - 2), (right + 2, bottom + 2), (70, 78, 83), -1)
    cv2.rectangle(bgr, (left, top), (right, bottom), (17, 20, 23), -1)
    cv2.rectangle(bgr, (left + 2, top + 2), (right - 2, bottom - 2), (24, 27, 29), 1)
    radius = max(5, int(round(housing_w * 0.30)))
    center_x = (left + right) // 2
    centers = [(center_x, int(round(top + (j + 0.5) * housing_h / 3))) for j in range(3)]
    for j, center in enumerate(centers):
        cv2.circle(bgr, center, radius + 2, (4, 5, 6), -1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, center, radius, (35, 32, 35), -1, lineType=cv2.LINE_AA)
    mask = np.zeros((height, width), dtype=np.uint8)
    index = COLORS.index(color)
    cv2.circle(mask, centers[index], radius, 255, -1, lineType=cv2.LINE_AA)
    bright = {'red': (0, 0, 245), 'yellow': (0, 225, 250), 'green': (0, 235, 0)}
    emission_bgr = np.array(bright[color], dtype=np.uint8).reshape(1, 1, 3)
    emission = (mask.astype(np.float32) / 255.0)[..., None] * core.linearize(emission_bgr)[0, 0] * 1.4
    box = [float(left), float(top), float(right), float(bottom)]
    gt.append({'class': 'traffic light', 'box': box, 'traffic_light_color': color})
    light = {'gt_bbox': box, 'color': color, 'center': centers[index], 'radius': radius, 'synthesized_active_signal': True, 'source': '2D controlled overlay, not a real measured LED'}
    return {'bgr': bgr, 'base': core.linearize(bgr), 'emission': emission, 'gt': gt, 'light': light, 'mask': mask, 'size': [width, height]}


def summary_counts(variant_totals):
    results = {}
    for variant, classes in variant_totals.items():
        results[variant] = {}
        for name, counts in classes.items():
            tp, fp, fn = (counts[k] for k in ('TP', 'FP', 'FN'))
            results[variant][name] = {**counts, 'precision': round(tp / (tp + fp), 4) if tp + fp else None, 'recall': round(tp / (tp + fn), 4) if tp + fn else None}
    return results


def load_model(root, yolo):
    if not yolo:
        return None
    from ultralytics import YOLO
    path = root / 'yolo11s.pt'
    if not path.exists():
        raise FileNotFoundError(f'Не найдены веса: {path}')
    return YOLO(str(path))


def run_effects(args, records, root, output):
    model = load_model(root, args.yolo)
    images = output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    totals = {variant: {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in CLASSES} for variant in VARIANTS}
    color_totals = {variant: {'correct': 0, 'wrong_or_unknown': 0, 'missed': 0} for variant in VARIANTS}
    oracle = {variant: {'correct': 0, 'wrong_or_unknown': 0} for variant in VARIANTS}
    per_scene = []
    for idx, record in enumerate(records[:args.count], 1):
        image_path, photo = core.scene_image(record, root)
        color = COLORS[(idx - 1) % len(COLORS)]
        prepared = controlled_scene(record, photo, args.render_width, color)
        item = {'image': record['image'], 'source_sha256': core.checksum(image_path), 'color': color, 'controlled_light_bbox': prepared['light']['gt_bbox'], 'variants': {}}
        for variant in VARIANTS:
            img, _, gt = core.simulate(prepared, variant, args)
            file = images / f'{idx:02d}_{variant}.png'
            cv2.imwrite(str(file), img)
            pred = core.infer_hybrid(model, img, args) if model is not None else []
            metrics = core.evaluate(gt, pred) if model is not None else None
            color_pred = core.light_evaluation(img, gt, pred, prepared['light']) if model is not None else None
            oracle_pred = core.classify_color(img, gt[-1]['box'])
            oracle[variant]['correct' if oracle_pred == color else 'wrong_or_unknown'] += 1
            if model is not None:
                for name in CLASSES:
                    for key in ('TP', 'FP', 'FN'):
                        totals[variant][name][key] += metrics[name][key]
                color_totals[variant][color_pred['status']] += 1
            item['variants'][variant] = {'file': file.name, 'sha256': core.checksum(file), 'predictions': pred, 'counts': metrics, 'end_to_end_color': color_pred, 'oracle_box_color_diagnostic': oracle_pred}
            if idx <= 3 and model is not None:
                cv2.imwrite(str(images / f'{idx:02d}_{variant}_pred.png'), core.draw_predictions(img, gt, pred))
        per_scene.append(item)
        print(f'{idx}/{min(args.count,len(records))}: {record["image"]} | inserted={color} | yolo={bool(model)}', flush=True)
    report = {'status': 'CONTROLLED_SYNTHETIC_LED_TECHNICAL_PILOT', 'notes': ['Один искусственный светофор добавлен на каждую дорожную фотографию', 'Успех цветового алгоритма по эталонной рамке (oracle) является только отдельным техническим тестом', 'Основная оценка цвета использует только обнаруженные YOLO рамки', 'Двумерная накладка не является физически достоверным изображением реального светофора', 'Результаты не являются независимой проверкой гипотезы выбора экспозиции'], 'scenes': len(per_scene), 'inserted_leds': len(per_scene), 'color_distribution': {c: sum(row['color'] == c for row in per_scene) for c in COLORS}, 'source': 'Existing BDD100K images already cached locally', 'parameters': {'render_width': args.render_width, 'exposure_ms': args.exposure_ms, 'speed_kmh': args.speed_kmh, 'depth_m': args.depth_m, 'pwm_hz': args.pwm_hz, 'duty': args.duty, 'phase': args.phase, 'num_steps': args.steps, 'imgsz': args.imgsz, 'conf': args.conf}, 'pwm_on_fraction': core.pwm_fraction(0, args.exposure_ms / 1000, args.pwm_hz, args.duty, args.phase), 'gt_box_color_diagnostic': oracle, 'end_to_end_color': color_totals if model is not None else None, 'detector': summary_counts(totals) if model is not None else None, 'model_sha256': core.checksum(root / 'yolo11s.pt') if model is not None else None}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'predictions.json').write_text(json.dumps(per_scene, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(output/'report.json'), 'color_distribution': report['color_distribution'], 'on_fraction': report['pwm_on_fraction'], 'colors': report['end_to_end_color'], 'oracle': report['gt_box_color_diagnostic']}, ensure_ascii=False, indent=2))


def run_convergence(args, records, root, output):
    model = load_model(root, args.yolo)
    rows = []
    for idx, record in enumerate(records[:args.count], 1):
        _, photo = core.scene_image(record, root)
        prepared = controlled_scene(record, photo, args.render_width, COLORS[(idx - 1) % 3])
        for name, speed, exp, phase in (('short_led_off', 0, 2.0, 0.35), ('long_fast', 90, 10.0, 0.13), ('medium_fast', 60, 8.0, 0.13)):
            case = argparse.Namespace(**vars(args))
            case.speed_kmh, case.exposure_ms, case.phase = speed, exp, phase
            outputs = {}
            for nsteps in (16, 32, 64, 128):
                img, linear, gt = core.simulate(prepared, 'combined', case, steps=nsteps)
                d = {'linear': linear, 'oracle_color': core.classify_color(img, gt[-1]['box'])}
                if model is not None:
                    detections = core.infer_hybrid(model, img, args)
                    d['detection_counts'] = core.evaluate(gt, detections)
                    d['end_to_end_color'] = core.light_evaluation(img, gt, detections, prepared['light'])['status']
                outputs[nsteps] = d
            for a, b in ((16, 32), (32, 64), (64, 128)):
                err = abs(outputs[a]['linear'] - outputs[b]['linear'])
                rows.append({'scene': record['image'], 'case': name, 'steps': f'{a}-{b}', 'mean_absolute_linear_pixel_difference': f'{err.mean():.9f}', 'max_absolute_linear_pixel_difference': f'{err.max():.9f}', 'oracle_color_changed': int(outputs[a]['oracle_color'] != outputs[b]['oracle_color']), 'detector_counts_changed': int(outputs[a].get('detection_counts') != outputs[b].get('detection_counts')) if model is not None else 'NOT_TESTED', 'end_to_end_color_changed': int(outputs[a].get('end_to_end_color') != outputs[b].get('end_to_end_color')) if model is not None else 'NOT_TESTED'})
        print(f'{idx}/{min(args.count,len(records))}: convergence', flush=True)
    with (output / 'convergence.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'scenes':min(args.count,len(records)), 'tests': len(rows), 'max_mean_64_to_128': max(float(x['mean_absolute_linear_pixel_difference']) for x in rows if x['steps']=='64-128'), 'yolo_compared': bool(model), 'csv':str(output/'convergence.csv')}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('effects', 'convergence'), default='effects')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--yolo', action='store_true')
    parser.add_argument('--render-width', type=int, default=640)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--exposure-ms', type=float, default=8.0)
    parser.add_argument('--speed-kmh', type=float, default=60.0)
    parser.add_argument('--depth-m', type=float, default=40.0)
    parser.add_argument('--pwm-hz', type=float, default=100.0)
    parser.add_argument('--duty', type=float, default=0.2)
    parser.add_argument('--phase', type=float, default=0.13)
    parser.add_argument('--reference-exposure-ms', type=float, default=8.0)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.1)
    args = parser.parse_args()
    if args.count < 1 or args.render_width < 256 or args.steps < 1 or args.exposure_ms <= 0 or args.reference_exposure_ms <= 0 or args.depth_m <= 0 or args.pwm_hz <= 0 or not (0 < args.duty <= 1) or not (0 <= args.phase < 1):
        parser.error('Некорректные параметры')
    root = Path(__file__).resolve().parent
    manifest_path = root / 'outputs_bdd_control/predictions.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Отсутствует {manifest_path}')
    records = json.loads(manifest_path.read_text(encoding='utf-8'))
    output = root / 'outputs_bdd_control/led_control_pilot'
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'effects':
        run_effects(args, records, root, output)
    else:
        run_convergence(args, records, root, output)


if __name__ == '__main__':
    main()
