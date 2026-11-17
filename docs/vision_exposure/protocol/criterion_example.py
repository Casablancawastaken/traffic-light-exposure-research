import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def empirical_upper_residual(observed, predicted, quantile=0.95):
    if len(observed) != len(predicted) or not observed:
        return None
    residuals = sorted(a - b for a, b in zip(observed, predicted))
    rank = min(len(residuals) - 1, max(0, math.ceil(quantile * len(residuals)) - 1))
    return max(0.0, residuals[rank])


def aggregate_forecasts(scenes, adjustment):
    missed = 0.0
    found = 0.0
    wrong_color = 0.0
    total_gt_light = 0
    fp_light = 0.0
    grouped = {}
    for scene in scenes:
        n_light = scene['n_light']
        a = min(1.0, max(0.0, scene['p_light_miss'] + adjustment.get('light_miss_prob', 0.0)))
        c = min(1.0, max(0.0, scene['p_color_error_given_found'] + adjustment.get('color_given_found_prob', 0.0)))
        total_gt_light += n_light
        missed += n_light * a
        found += n_light * (1 - a)
        wrong_color += n_light * (1 - a) * c
        fp_light += max(0.0, scene.get('fp_light_expected', 0.0) + adjustment.get('light_fp_count', 0.0))
        for cls, sizegroups in scene['objects'].items():
            for size, cell in sizegroups.items():
                name = f'{cls}_{size}'
                if name not in grouped:
                    grouped[name] = {'tp': 0.0, 'fn': 0.0, 'fp': 0.0}
                n = cell['n_gt']
                p = min(1.0, max(0.0, cell['p_miss'] + adjustment.get(f'{name}_miss_prob', 0.0)))
                lam = max(0.0, cell['fp_expected'] + adjustment.get(f'{name}_fp_count', 0.0))
                grouped[name]['tp'] += n * (1 - p)
                grouped[name]['fn'] += n * p
                grouped[name]['fp'] += lam
    output = {
        'traffic_light': {
            'gt': total_gt_light,
            'expected_missed': missed,
            'expected_found': found,
            'expected_color_errors_among_found': wrong_color,
            'expected_false_positives': fp_light,
            'miss_rate': safe_ratio(missed, total_gt_light),
            'recall': safe_ratio(found, total_gt_light),
            'precision': safe_ratio(found, found + fp_light),
            'color_error_given_found': safe_ratio(wrong_color, found)
        },
        'size_groups': {}
    }
    for name, cnt in grouped.items():
        output['size_groups'][name] = {**cnt,
            'precision': safe_ratio(cnt['tp'], cnt['tp'] + cnt['fp']),
            'recall': safe_ratio(cnt['tp'], cnt['tp'] + cnt['fn'])}
    return output



def decide_exposure(affected, clean, p0, blur_pixels, config):
    limits = config['limits']
    if p0 > limits['p0_max'] or blur_pixels > limits['blur_max_pixels']:
        return 'R'
    light = affected['traffic_light']
    clean_light = clean['traffic_light']
    require = [light['miss_rate'], light['color_error_given_found'],clean_light['recall'],clean_light['precision'],clean_light['color_error_given_found']]
    if any(x is None for x in require):
        return 'U'
    if light['miss_rate'] > limits['traffic_light_miss_max'] or light['color_error_given_found'] > limits['traffic_light_color_given_found_max']:
        return 'R'
    if clean_light['recall'] < limits['clean_traffic_light_detection_and_color_min'] or clean_light['precision'] < limits['clean_traffic_light_detection_and_color_min'] or 1 - clean_light['color_error_given_found'] < limits['clean_traffic_light_detection_and_color_min']:
        return 'R'
    required = set(config['required_size_groups'])
    if not required.issubset(clean['size_groups']) or not required.issubset(affected['size_groups']):
        return 'U'
    if set(clean['size_groups']) != set(affected['size_groups']):
        return 'U'
    for name, group in affected['size_groups'].items():
        baseline = clean['size_groups'][name]
        if any(group[k] is None or baseline[k] is None for k in ('recall','precision')):
            return 'U'
        if baseline['recall'] < limits['clean_person_car_precision_recall_min'] or baseline['precision'] < limits['clean_person_car_precision_recall_min']:
            return 'R'
        if baseline['recall'] - group['recall'] > limits['precision_recall_drop_max_pp'] or baseline['precision'] - group['precision'] > limits['precision_recall_drop_max_pp']:
            return 'R'
    return 'A'


def classify_truth_from_intervals(checks, check_specs):
    if not checks:
        return 'Q'
    unknown = False
    for name, data in checks.items():
        if name not in check_specs or data is None:
            return 'Q'
        spec = check_specs[name]
        low = data.get('lower')
        high = data.get('upper')
        if low is None or high is None or low > high:
            return 'Q'
        limit = spec['threshold']
        direction = spec['direction']
        if direction == 'max':
            if low > limit:
                return 'F'
            if high > limit:
                unknown = True
        elif direction == 'min':
            if high < limit:
                return 'F'
            if low < limit:
                unknown = True
        else:
            raise ValueError(direction)
    if set(checks) != set(check_specs):
        return 'Q'
    return 'Q' if unknown else 'S'


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
    good = aggregate_forecasts([good_scene],{})
    no_light = aggregate_forecasts([{'n_light':1,'p_light_miss':1.0,'p_color_error_given_found':0.01,'objects':{}}],{})
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
            'example_miss_rate_positive_95_quantile':empirical_upper_residual([0.0,0.1,0.3,0.2],[0.1,0.1,0.1,0.1]),
            'example_fp_count_positive_95_quantile':empirical_upper_residual([1,2,0,3],[0.7,0.9,0.5,1.1]),
            'warning':'The empirical quantile is a calibration heuristic, not a 95 percent risk confidence bound'
        },
        'forecast_example':aggregate_forecasts(sample_scenes,{'light_miss_prob':0.0,'color_given_found_prob':0.0,'person_small_fp_count':0.1})
    }


def selftest(report, config):
    u = report['decision_example']
    assert u['n_regimes_all_including_Q'] == 10
    assert u['truth']=={'S':3,'F':5,'Q':2}
    assert u['proposed_decisions']=={'A':4,'R':3,'U':3}
    assert u['false_allow_conservative_upper']==0.75
    assert u['admissible_coverage_with_U_in_denominator']==1/3
    assert report['empty_allowed_example']['false_allow_conservative_upper'] is None
    assert abs(report['zero_errors_binomial_example_independent_trials_only']['upper'] - (1-0.05**(1/20)))<1e-10
    a=report['forecast_example']['traffic_light']
    assert abs(a['expected_found']-1.3)<1e-9
    assert abs(a['color_error_given_found']-(0.9*0.2+0.4*0.8)/1.3)<1e-9
    assert report['status_interval_examples']=={'confirmed_S':'S','confirmed_F':'F','uncertain_Q':'Q'}
    assert report['forecast_decision_examples']=={'A':'A','R':'R','U':'U'}
    expected=config['alpha']['status_budget']/(config['max_regimes']*2*len(config['status_tests']))
    assert abs(expected - config['alpha']['status_tail_budget_each']) < 1e-16
    assert len(config['status_tests'])==21
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(ROOT/'criterion_config.json'))
    ap.add_argument('--out',default=str(ROOT/'criterion_example_report.json'))
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    result=example(config)
    result['selftest_pass']=selftest(result,config)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selftest_pass':result['selftest_pass'],'decision_example':result['decision_example'],'zero_errors_upper_95':result['zero_errors_binomial_example_independent_trials_only']['upper']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
