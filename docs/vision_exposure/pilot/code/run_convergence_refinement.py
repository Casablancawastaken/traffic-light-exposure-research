import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_effects_pilot as core
import run_led_control_pilot as led
import run_multiscene_convergence as original


def pair_metrics(a, b):
    diff = np.abs(a['linear'] - b['linear'])
    ac, bc = a['contrast'], b['contrast']
    delta_contrast = abs(ac - bc) if ac is not None and bc is not None else None
    gt_shift = max(abs(x - y) for ga, gb in zip(a['gt'], b['gt']) for x, y in zip(ga['box'], gb['box']))
    changed_counts = a['counts'] != b['counts']
    changed_color = a['end'] != b['end']
    pass_numeric = float(diff.mean()) <= 0.01 and delta_contrast is not None and delta_contrast <= 0.02 and gt_shift <= 1e-8
    passed = pass_numeric and a['oracle'] == b['oracle'] and not changed_counts and not changed_color
    return {
        'mean_abs_linear_diff': float(diff.mean()),
        'max_abs_linear_diff': float(diff.max()),
        'abs_contrast_diff': delta_contrast,
        'gt_midbox_max_coord_shift': float(gt_shift),
        'oracle_color_change': a['oracle'] != b['oracle'],
        'detector_counts_change': changed_counts,
        'detector_color_change': changed_color,
        'meets_original_convergence_rule': bool(passed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--base-report', default='docs/vision_exposure/pilot/multiscene_convergence/report.json')
    parser.add_argument('--output', default='docs/vision_exposure/pilot/convergence_refinement')
    parser.add_argument('--render-width', type=int, default=384)
    parser.add_argument('--steps', default='128,256,512')
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    base_report = Path(args.base_report).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    steps = tuple(int(x) for x in args.steps.split(','))
    if len(steps) < 2 or any(b <= a for a, b in zip(steps, steps[1:])):
        parser.error('Шаги должны быть возрастающими и содержать как минимум два значения')

    old_report = json.loads(base_report.read_text(encoding='utf-8'))
    failures = old_report.get('failures_64_128', [])
    if not failures:
        raise RuntimeError('В исходном отчёте нет случаев, требующих повторной проверки')
    records = json.loads((root / 'outputs_bdd_control/predictions.json').read_text(encoding='utf-8'))
    by_scene = {rec['image']: (i, rec) for i, rec in enumerate(records)}
    by_case = {item[0]: item[1:] for item in original.CASES}
    model = led.load_model(root, True)
    rows = []
    outputs = []

    for index, failure in enumerate(failures, 1):
        name = failure['scene']
        case = failure['case']
        if name not in by_scene or case not in by_case:
            raise RuntimeError(f'Нет сцены или параметров: {name} / {case}')
        source_index, rec = by_scene[name]
        exposure, speed, phase = by_case[case]
        image_path, photo = core.scene_image(rec, root)
        prepared = led.controlled_scene(rec, photo, args.render_width, led.COLORS[source_index % 3])
        params = argparse.Namespace(steps=steps[0], exposure_ms=exposure, reference_exposure_ms=5.0, speed_kmh=speed, depth_m=15.0, pwm_hz=100.0, duty=0.2, phase=phase, imgsz=960, conf=0.1)
        per_step = {}
        serial = {}

        for step in steps:
            rendered, linear, gt = core.simulate(prepared, 'combined', params, steps=step)
            detections = core.infer_hybrid(model, rendered, params)
            counts = core.evaluate(gt, detections)
            end = core.light_evaluation(rendered, gt, detections, prepared['light'])['status']
            oracle = core.classify_color(rendered, gt[-1]['box'])
            scale = core.projection_scale(exposure / 2000.0, speed, 15.0)
            contrast = original.contrast_value(linear, prepared, gt[-1]['box'], scale)
            per_step[step] = {'linear': linear, 'gt': gt, 'counts': counts, 'end': end, 'oracle': oracle, 'contrast': contrast}
            serial[str(step)] = {
                'counts': counts,
                'end_to_end_color_status': end,
                'oracle_box_color': oracle,
                'contrast': contrast,
                'gt_mid_exposure': gt,
                'predictions': detections,
            }

        for low, high in zip(steps, steps[1:]):
            result = pair_metrics(per_step[low], per_step[high])
            rows.append({'scene': name, 'case': case, 'steps': f'{low}-{high}', **result})
        outputs.append({
            'scene': name,
            'case': case,
            'exposure_ms': exposure,
            'speed_kmh': speed,
            'phase': phase,
            'source_image_sha256': core.checksum(image_path),
            'steps': serial,
        })
        print(f'{index}/{len(failures)}: {name} | {case}', flush=True)

    with (output / 'comparisons.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'predictions.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding='utf-8')
    last_pair = f'{steps[-2]}-{steps[-1]}'
    last_rows = [r for r in rows if r['steps'] == last_pair]
    not_converged = [r for r in last_rows if not r['meets_original_convergence_rule']]
    report = {
        'status': 'FOLLOW_UP_CONVERGED_ON_FAILED_CASES' if not not_converged else 'FOLLOW_UP_NOT_CONVERGED',
        'source_multiscene_status': old_report.get('status'),
        'cases_reexamined': len(failures),
        'step_levels': list(steps),
        'last_pair': last_pair,
        'last_pair_failures': not_converged,
        'all_pair_comparisons': len(rows),
        'scope': 'Только ранее не прошедшие случаи, техническая сходимость двумерной модели; не подтверждает пригодность чистого контроля, физику датчика или гипотезу',
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'last_pair_failures'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
