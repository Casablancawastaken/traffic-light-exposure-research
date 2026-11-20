import argparse
import json
import math
from pathlib import Path

from statistical_contract import above, below, classify_intervals, finite_number, paired_cluster_bootstrap, classify_mode, test_specs

ROOT = Path(__file__).resolve().parent


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def empirical_upper_residual(observed, predicted, quantile=0.95, min_support=30):
    if len(observed) != len(predicted) or len(observed) < min_support:
        return None
    if not finite_number(quantile) or not 0 < quantile < 1:
        return None
    if any(not finite_number(a) or not finite_number(b) for a, b in zip(observed, predicted)):
        return None
    residuals = sorted(a - b for a, b in zip(observed, predicted))
    rank = min(len(residuals) - 1, max(0, math.ceil(quantile * len(residuals)) - 1))
    return max(0.0, residuals[rank])


def calibration_adjustment(observed, predicted, config):
    min_support = config['calibration']['minimum_valid_scenes_per_mode_and_metric']
    value = empirical_upper_residual(observed, predicted, min_support=min_support)
    return {'estimate': value, 'valid_scenes': len(observed),
            'status': 'ESTIMATED_ZERO' if value == 0.0 else ('ESTIMATED_POSITIVE' if value is not None else 'UNDEFINED')}


def calibrate_by_mode_scene_records(records, config):
    buckets = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Calibration record must be a dict')
        mode, metric, scene_id = (record.get(k) for k in ('mode', 'metric', 'scene_id'))
        if mode is None or metric is None or scene_id is None:
            raise ValueError('Missing mode, metric or scene_id')
        key = (str(mode), str(metric))
        buckets.setdefault(key, []).append(record)
    result = {}
    for (mode, metric), rows in buckets.items():
        unique = {str(row['scene_id']) for row in rows}
        if len(unique) != len(rows) or any(not finite_number(row.get('observed')) or not finite_number(row.get('predicted')) for row in rows):
            value = {'estimate': None, 'valid_scenes': 0, 'status': 'UNDEFINED'}
        else:
            value = calibration_adjustment([row['observed'] for row in rows],
                                           [row['predicted'] for row in rows], config)
        result.setdefault(mode, {})[metric] = value
    return result


def explicit_zero_adjustments(scenes):
    result = {name: {'estimate': 0.0, 'status': 'ESTIMATED_ZERO'} for name in
              ('light_miss_prob', 'color_given_found_prob', 'light_fp_count')}
    for scene in scenes:
        for klass, groups in scene.get('objects', {}).items():
            for size in groups:
                result[f'{klass}_{size}_miss_prob'] = {'estimate': 0.0, 'status': 'ESTIMATED_ZERO'}
                result[f'{klass}_{size}_fp_count'] = {'estimate': 0.0, 'status': 'ESTIMATED_ZERO'}
    return result


def aggregate_forecasts(scenes, adjustment, calibration_required=True):
    def unknown():
        return {'traffic_light': {'gt': None, 'miss_rate': None, 'recall': None, 'precision': None,
                                  'color_error_given_found': None}, 'size_groups': {}, 'calibration_status': 'U'}

    if not isinstance(adjustment, dict):
        return unknown()

    def correction(name):
        if name not in adjustment:
            return None if calibration_required else 0.0
        value = adjustment[name]
        if isinstance(value, dict):
            value = value.get('estimate') if value.get('status') in ('ESTIMATED_ZERO', 'ESTIMATED_POSITIVE') else None
        return value if finite_number(value) and value >= 0 else None

    missed = found = wrong_color = fp_light = 0.0
    total_gt_light = 0
    grouped = {}
    for scene in scenes:
        n_light = scene.get('n_light')
        if not finite_number(n_light) or n_light < 0:
            return unknown()
        a0 = scene.get('p_light_miss')
        c0 = scene.get('p_color_error_given_found')
        f0 = scene.get('fp_light_expected', 0.0)
        am = correction('light_miss_prob')
        ac = correction('color_given_found_prob')
        af = correction('light_fp_count')
        if any(not finite_number(x) for x in (a0, c0, f0, am, ac, af)):
            return unknown()
        a = min(1.0, max(0.0, a0 + am))
        c = min(1.0, max(0.0, c0 + ac))
        total_gt_light += n_light
        missed += n_light * a
        found += n_light * (1 - a)
        wrong_color += n_light * (1 - a) * c
        fp_light += max(0.0, f0 + af)
        for cls, sizegroups in scene.get('objects', {}).items():
            for size, cell in sizegroups.items():
                name = f'{cls}_{size}'
                n, p0, fp = cell.get('n_gt'), cell.get('p_miss'), cell.get('fp_expected')
                adj_p = correction(f'{name}_miss_prob')
                adj_f = correction(f'{name}_fp_count')
                if any(not finite_number(x) for x in (n, p0, fp, adj_p, adj_f)):
                    return unknown()
                if min(n, fp, adj_p, adj_f) < 0 or not 0 <= p0 <= 1:
                    return unknown()
                if name not in grouped:
                    grouped[name] = {'tp': 0.0, 'fn': 0.0, 'fp': 0.0}
                prob = min(1.0, max(0.0, p0 + adj_p))
                lam = max(0.0, fp + adj_f)
                grouped[name]['tp'] += n * (1 - prob)
                grouped[name]['fn'] += n * prob
                grouped[name]['fp'] += lam
    out = {'traffic_light': {'gt': total_gt_light, 'expected_missed': missed,
                             'expected_found': found, 'expected_color_errors_among_found': wrong_color,
                             'expected_false_positives': fp_light,
                             'miss_rate': safe_ratio(missed, total_gt_light),
                             'recall': safe_ratio(found, total_gt_light),
                             'precision': safe_ratio(found, found + fp_light),
                             'color_error_given_found': safe_ratio(wrong_color, found)},
           'size_groups': {}, 'calibration_status': 'DEFINED'}
    for name, cnt in grouped.items():
        out['size_groups'][name] = {**cnt,
            'precision': safe_ratio(cnt['tp'], cnt['tp'] + cnt['fp']),
            'recall': safe_ratio(cnt['tp'], cnt['tp'] + cnt['fn'])}
    return out


def decide_exposure(affected, clean, p0, blur_pixels, config):
    limits = config['limits']
    violated = False
    unknown = False

    def check_max(value, threshold):
        nonlocal violated, unknown
        if not finite_number(value):
            unknown = True
        elif above(value, threshold):
            violated = True

    def check_min(value, threshold):
        nonlocal violated, unknown
        if not finite_number(value):
            unknown = True
        elif below(value, threshold):
            violated = True

    check_max(p0, limits['p0_max'])
    check_max(blur_pixels, limits['blur_max_pixels'])
    if affected.get('calibration_status') != 'DEFINED' or clean.get('calibration_status') != 'DEFINED':
        unknown = True
    a_light = affected.get('traffic_light', {})
    c_light = clean.get('traffic_light', {})
    check_max(a_light.get('miss_rate'), limits['traffic_light_miss_max'])
    check_max(a_light.get('color_error_given_found'), limits['traffic_light_color_given_found_max'])
    check_min(c_light.get('recall'), limits['clean_traffic_light_detection_and_color_min'])
    check_min(c_light.get('precision'), limits['clean_traffic_light_detection_and_color_min'])
    color = c_light.get('color_error_given_found')
    check_min(1 - color if finite_number(color) else None, limits['clean_traffic_light_detection_and_color_min'])

    required = set(config['required_size_groups'])
    a_groups = affected.get('size_groups', {})
    c_groups = clean.get('size_groups', {})
    if not required.issubset(a_groups) or not required.issubset(c_groups):
        unknown = True
    if set(a_groups) != set(c_groups):
        unknown = True

    for name in sorted(required):
        a = a_groups.get(name, {})
        c = c_groups.get(name, {})
        for metric in ('precision', 'recall'):
            left = c.get(metric)
            right = a.get(metric)
            check_max(left - right if finite_number(left) and finite_number(right) else None,
                      limits['precision_recall_drop_max_pp'])

    for klass in ('person', 'car'):
        for metric in ('precision', 'recall'):
            tp = fp = fn = 0.0
            complete = True
            for size in ('small', 'medium', 'large'):
                record = c_groups.get(f'{klass}_{size}', {})
                values = [record.get(k) for k in ('tp', 'fp', 'fn')]
                if any(not finite_number(x) or x < 0 for x in values):
                    complete = False
                    break
                tp += values[0]
                fp += values[1]
                fn += values[2]
            result = safe_ratio(tp, tp + (fp if metric == 'precision' else fn)) if complete else None
            check_min(result, limits['clean_person_car_precision_recall_min'])

    return 'R' if violated else ('U' if unknown else 'A')


def classify_truth_from_intervals(checks, check_specs):
    intervals = {}
    for name, data in checks.items():
        if isinstance(data, dict):
            intervals[name] = [data.get('lower'), data.get('upper')]
        else:
            intervals[name] = data
    return classify_intervals(intervals, check_specs)


def losses(method, truth):
    if truth == 'Q':
        return 1
    if method == 'U':
        return 1
    return 0 if (method == 'A' and truth == 'S') or (method == 'R' and truth == 'F') else 1


def decision_report(rows):
    all_records = list(rows)
    if not all_records:
        raise ValueError('No regimes')
    for row in all_records:
        if row['truth'] not in ('S', 'F', 'Q') or row['proposed'] not in ('A', 'R', 'U') or row['analytic'] not in ('A', 'R'):
            raise ValueError('Invalid truth or decision')
    a = [r for r in all_records if r['proposed'] == 'A']
    s = [r for r in all_records if r['truth'] == 'S']
    f = [r for r in all_records if r['truth'] == 'F']
    q = [r for r in all_records if r['truth'] == 'Q']
    av = len(a)
    true_allow = sum(r['truth'] == 'S' for r in a)
    false_allow = sum(r['truth'] == 'F' for r in a)
    uncertain_truth_allow = sum(r['truth'] == 'Q' for r in a)
    false_reject = sum(r['truth'] == 'S' and r['proposed'] == 'R' for r in all_records)
    uncertain_decision = sum(r['proposed'] == 'U' for r in all_records)
    uncertain_decision_on_true_allowed = sum(r['truth'] == 'S' and r['proposed'] == 'U' for r in all_records)
    n = len(all_records)
    improvement = sum(losses(r['analytic'],r['truth']) - losses(r['proposed'],r['truth']) for r in all_records) / n
    against_reject_all = sum(losses('R',r['truth']) - losses(r['proposed'],r['truth']) for r in all_records) / n
    return {
        'n_regimes_all_including_Q': n,
        'truth': {'S': len(s), 'F': len(f), 'Q': len(q)},
        'proposed_decisions': {key: sum(r['proposed'] == key for r in all_records) for key in ('A','R','U')},
        'correct_allow': true_allow,
        'false_allow_confirmed': false_allow,
        'false_reject_confirmed': false_reject,
        'uncertain_decisions': uncertain_decision,
        'uncertain_decision_on_S': uncertain_decision_on_true_allowed,
        'allowed_with_uncertain_truth_Q': uncertain_truth_allow,
        'false_allow_conservative_upper': safe_ratio(false_allow + uncertain_truth_allow, av),
        'false_allow_confirmed_F_over_A': safe_ratio(false_allow, av),
        'false_allow_among_confirmed_F': safe_ratio(false_allow, len(f)),
        'admissible_coverage_with_U_in_denominator': safe_ratio(true_allow, len(s)),
        'improvement_analytic_minus_proposed_with_U_penalty': improvement,
        'improvement_reject_all_minus_proposed_with_U_penalty': against_reject_all,
        'mean_loss_analytic': sum(losses(r['analytic'],r['truth']) for r in all_records) / n,
        'mean_loss_proposed': sum(losses(r['proposed'],r['truth']) for r in all_records) / n,
        'loss_rule': 'U is a loss on known S/F; Q is charged equally to both rules on the full N',
        'conditional_determined_coverage': safe_ratio(len(s)+len(f), n)
    }


def binomial_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def exact_binomial_upper(k, n, alpha):
    if n <= 0 or not 0 <= k <= n or not 0 < alpha < 1:
        return None
    if k == n:
        return 1.0
    left, right = 0.0, 1.0
    for _ in range(75):
        mid = (left + right) / 2
        if binomial_cdf(k, n, mid) > alpha:
            left = mid
        else:
            right = mid
    return (left + right) / 2


def example(config):
    rows = [
        {'mode': 'r1','truth':'F','analytic':'A','proposed':'A'},
        {'mode': 'r2','truth':'S','analytic':'R','proposed':'A'},
        {'mode': 'r3','truth':'S','analytic':'A','proposed':'U'},
        {'mode': 'r4','truth':'F','analytic':'A','proposed':'R'},
        {'mode': 'r5','truth':'Q','analytic':'R','proposed':'A'},
        {'mode': 'r6','truth':'Q','analytic':'A','proposed':'U'},
        {'mode': 'r7','truth':'F','analytic':'R','proposed':'R'},
        {'mode': 'r8','truth':'S','analytic':'R','proposed':'R'},
        {'mode': 'r9','truth':'F','analytic':'A','proposed':'U'},
        {'mode': 'r10','truth':'F','analytic':'R','proposed':'A'}
    ]
    sample_scenes = [
        {'n_light':1,'p_light_miss':0.1,'p_color_error_given_found':0.2,
         'objects': {'person': {'small': {'n_gt':2, 'p_miss':0.25,'fp_expected':0.5}},
                     'car': {'small': {'n_gt':1,'p_miss':0.4,'fp_expected':0.25}}}},
        {'n_light':1,'p_light_miss':0.6,'p_color_error_given_found':0.8,
         'objects': {'person': {'small': {'n_gt':1, 'p_miss':0.5,'fp_expected':0.3}},
                     'car': {'small': {'n_gt':2,'p_miss':0.2,'fp_expected':0.1}}}}
    ]
    specs = {'risk': {'threshold':0.05,'direction':'max'}}
    good_group = {size:{'n_gt':1,'p_miss':0.01,'fp_expected':0.01} for size in ('small','medium','large')}
    good_scene = {'n_light':1,'p_light_miss':0.01,'p_color_error_given_found':0.01,'fp_light_expected':0.01,'objects': {'person':good_group, 'car':good_group}}
    good = aggregate_forecasts([good_scene],explicit_zero_adjustments([good_scene]))
    no_light = json.loads(json.dumps(good))
    no_light['traffic_light']['color_error_given_found'] = None
    return {
        'forecast_decision_examples': {'A':decide_exposure(good,good,0.0,1.0,config),'R':decide_exposure(good,good,0.5,1.0,config),'U':decide_exposure(no_light,good,0.0,1.0,config)},
        'status_interval_examples': {
            'confirmed_S': classify_truth_from_intervals({'risk':{'lower':0.01,'upper':0.03}}, specs),
            'confirmed_F': classify_truth_from_intervals({'risk':{'lower':0.07,'upper':0.11}}, specs),
            'uncertain_Q': classify_truth_from_intervals({'risk':{'lower':0.02,'upper':0.10}}, specs)
        },
        'decision_example': decision_report(rows),
        'empty_allowed_example': decision_report([{'truth':'F','analytic':'R','proposed':'R'} for _ in range(20)]),
        'zero_errors_binomial_example_independent_trials_only': {
            'errors':0, 'n_independent':20, 'alpha_one_sided':0.05,
            'upper':exact_binomial_upper(0,20,0.05),
            'caveat':'Not valid treating dependent phases/modes as 20 independent trials'
        },
        'residuals': {
            'unit':'one_calibration_scene with all phases for one fixed mode',
            'color_without_detected_light':'skip conditional residual, do not impute zero',
            'example_miss_rate_positive_95_quantile':empirical_upper_residual(([0.0,0.1,0.3,0.2] * 8)[:30], [0.1] * 30, min_support=config['calibration']['minimum_valid_scenes_per_mode_and_metric']),
            'example_fp_count_positive_95_quantile':empirical_upper_residual(([1,2,0,3] * 8)[:30], ([0.7,0.9,0.5,1.1] * 8)[:30], min_support=config['calibration']['minimum_valid_scenes_per_mode_and_metric']),
            'warning':'The empirical quantile is a calibration heuristic, not a 95 percent risk confidence bound'
        },
        'calibration_mode_example': calibrate_by_mode_scene_records(
            [{'mode': 'r_1', 'metric': 'traffic_light_miss', 'scene_id': i, 'observed': 0.0, 'predicted': 0.0} for i in range(30)] +
            [{'mode': 'r_2', 'metric': 'traffic_light_miss', 'scene_id': i, 'observed': 0.0, 'predicted': 0.0} for i in range(29)], config),
        'forecast_example':aggregate_forecasts(sample_scenes,{**explicit_zero_adjustments(sample_scenes),'person_small_fp_count':0.1})
    }


def synthetic_scene(mode):
    size_groups = {part: {'tp': 10, 'fp': 0, 'fn': 0} for part in ('small', 'medium', 'large')}
    clean = {'traffic_light': {'tp': 10, 'fp': 0, 'fn': 0, 'color_errors': 0, 'correct_color': 10},
             'person': size_groups, 'car': size_groups}
    affected = json.loads(json.dumps(clean))
    if mode == 'unsafe':
        affected['traffic_light'].update({'tp': 0, 'fn': 10, 'color_errors': 0, 'correct_color': 0})
    if mode == 'undefined':
        del affected['traffic_light']['correct_color']
    return {'clean': clean, 'affected': affected, 'physical': {'p0': 0, 'blur_pixels': 1}}


def statistical_example(config):
    scenes = [{'modes': {mode: synthetic_scene(mode) for mode in ('safe', 'unsafe', 'undefined')}} for _ in range(24)]
    decisions = {'safe': {'analytic': 'A', 'proposed': 'A'},
                 'unsafe': {'analytic': 'A', 'proposed': 'R'},
                 'undefined': {'analytic': 'R', 'proposed': 'U'}}
    by_mode = {mode: classify_mode([row['modes'][mode] for row in scenes], config)
               for mode in decisions}
    bootstrap = paired_cluster_bootstrap(scenes, decisions, config, repetitions=60)
    return {'unit': 'independent synthetic scene with all phases and paired modes',
            'synthetic_scenes': len(scenes), 'mode_results': by_mode,
            'paired_delta': bootstrap, 'not_independent_empirical_claim': True}


def selftest(report, config):
    u = report['decision_example']
    assert u['n_regimes_all_including_Q'] == 10
    assert u['truth'] == {'S': 3, 'F': 5, 'Q': 2}
    assert u['proposed_decisions'] == {'A': 4, 'R': 3, 'U': 3}
    assert u['false_allow_conservative_upper'] == 0.75
    assert u['admissible_coverage_with_U_in_denominator'] == 1 / 3
    assert report['empty_allowed_example']['false_allow_conservative_upper'] is None
    assert abs(report['zero_errors_binomial_example_independent_trials_only']['upper'] - (1 - 0.05 ** (1 / 20))) < 1e-10
    light = report['forecast_example']['traffic_light']
    assert abs(light['expected_found'] - 1.3) < 1e-9
    assert abs(light['color_error_given_found'] - (0.9 * 0.2 + 0.4 * 0.8) / 1.3) < 1e-9
    assert report['status_interval_examples'] == {'confirmed_S': 'S', 'confirmed_F': 'F', 'uncertain_Q': 'Q'}
    assert report['forecast_decision_examples'] == {'A': 'A', 'R': 'R', 'U': 'U'}
    expected = config['alpha']['status_budget'] / (config['max_regimes'] * 2 * len(config['status_tests']))
    assert abs(expected - config['alpha']['status_tail_budget_each']) < 1e-16
    assert len(config['status_tests']) == 21
    assert report['statistical_example']['mode_results']['unsafe']['status'] == 'F'
    assert report['statistical_example']['mode_results']['undefined']['status'] == 'Q'
    assert report['statistical_example']['paired_delta']['status_recomputed_in_each_bootstrap'] is True
    assert report['calibration_mode_example']['r_1']['traffic_light_miss']['status'] == 'ESTIMATED_ZERO'
    assert report['calibration_mode_example']['r_2']['traffic_light_miss']['status'] == 'UNDEFINED'
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(ROOT/'criterion_config.json'))
    ap.add_argument('--out',default=str(ROOT/'criterion_example_report.json'))
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    result=example(config)
    result['statistical_example'] = statistical_example(config)
    result['selftest_pass']=selftest(result,config)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selftest_pass':result['selftest_pass'],'decision_example':result['decision_example'],'zero_errors_upper_95':result['zero_errors_binomial_example_independent_trials_only']['upper']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
