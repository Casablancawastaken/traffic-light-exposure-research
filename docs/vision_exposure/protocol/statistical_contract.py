import math
import random
from statistics import fmean


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def above(value, threshold, tol=1e-12):
    return value > threshold + tol


def below(value, threshold, tol=1e-12):
    return value < threshold - tol


def exact_zero_upper(n, alpha):
    if not isinstance(n, int) or n < 1 or not finite_number(alpha) or not 0 < alpha < 1:
        return None
    return -math.expm1(math.log(alpha) / n)


def mean_interval(values, cap, alpha):
    if not values or not finite_number(cap) or cap <= 0 or not finite_number(alpha) or not 0 < alpha < 1:
        return None
    if any(not finite_number(value) or value < 0 or value > cap for value in values):
        return None
    n = len(values)
    if all(x == 0 for x in values):
        return [0.0, cap * exact_zero_upper(n, alpha / 2)]
    if all(x == cap for x in values):
        return [cap * (1 - exact_zero_upper(n, alpha / 2)), cap]
    radius = cap * math.sqrt(math.log(2 / alpha) / (2 * n))
    avg = fmean(values)
    return [max(0.0, avg - radius), min(cap, avg + radius)]


def counter_for_scene(row, variant, klass, size, key):
    branch = row.get(variant)
    if not isinstance(branch, dict) or klass not in branch:
        return None
    data = branch[klass]
    if klass in ('person', 'car'):
        if size == 'all':
            if not isinstance(data, dict) or any(x not in data for x in ('small', 'medium', 'large')):
                return None
            values = [data[part].get(key) for part in ('small', 'medium', 'large')]
            return sum(values) if all(finite_number(x) and x >= 0 for x in values) else None
        data = data.get(size)
    value = data.get(key) if isinstance(data, dict) else None
    return value if finite_number(value) and value >= 0 else None


def counter_limit(config, klass, size, key):
    caps = config['statistical_contract']['max_scene_counts']
    if klass in ('person', 'car') and size == 'all':
        return 3 * caps[key]
    return caps[key]


def ratio_bounds(scene_rows, variant, klass, size, numerator, second, alpha, config):
    if not scene_rows:
        return None
    xs = [counter_for_scene(row, variant, klass, size, numerator) for row in scene_rows]
    ys = [counter_for_scene(row, variant, klass, size, second) for row in scene_rows]
    if any(x is None for x in xs + ys):
        return None
    if sum(xs) + sum(ys) == 0:
        return None
    xcap = counter_limit(config, klass, size, numerator)
    ycap = counter_limit(config, klass, size, second)
    xi = mean_interval(xs, xcap, alpha / 2)
    yi = mean_interval(ys, ycap, alpha / 2)
    if xi is None or yi is None:
        return None
    low = xi[0] / (xi[0] + yi[1]) if xi[0] + yi[1] > 0 else 0.0
    high = xi[1] / (xi[1] + yi[0]) if xi[1] + yi[0] > 0 else 1.0
    return [max(0.0, low), min(1.0, high)]


def color_bounds(scene_rows, variant, alpha, config):
    errs = [counter_for_scene(row, variant, 'traffic_light', 'all', 'color_errors') for row in scene_rows]
    oks = [counter_for_scene(row, variant, 'traffic_light', 'all', 'correct_color') for row in scene_rows]
    if any(x is None for x in errs + oks) or sum(errs) + sum(oks) == 0:
        return None
    ei = mean_interval(errs, counter_limit(config, 'traffic_light', 'all', 'color_errors'), alpha / 2)
    oi = mean_interval(oks, counter_limit(config, 'traffic_light', 'all', 'correct_color'), alpha / 2)
    if ei is None or oi is None:
        return None
    lower = ei[0] / (ei[0] + oi[1]) if ei[0] + oi[1] > 0 else 0.0
    upper = ei[1] / (ei[1] + oi[0]) if ei[1] + oi[0] > 0 else 1.0
    return [max(0.0, lower), min(1.0, upper)]


def interval_for_test(rows, name, alpha, config):
    if name == 'traffic_light_miss':
        return ratio_bounds(rows, 'affected', 'traffic_light', 'all', 'fn', 'tp', alpha, config)
    if name == 'traffic_light_color_error_given_detected':
        return color_bounds(rows, 'affected', alpha, config)
    if name.startswith('clean_'):
        suffix = name[6:]
        for klass in ('traffic_light', 'person', 'car'):
            if suffix.startswith(klass + '_'):
                metric = suffix[len(klass) + 1:]
                if metric == 'precision_abs':
                    return ratio_bounds(rows, 'clean', klass, 'all', 'tp', 'fp', alpha, config)
                if metric == 'recall_abs':
                    return ratio_bounds(rows, 'clean', klass, 'all', 'tp', 'fn', alpha, config)
                if metric == 'color_abs' and klass == 'traffic_light':
                    interval = color_bounds(rows, 'clean', alpha, config)
                    return [1 - interval[1], 1 - interval[0]] if interval is not None else None
    for klass in ('person', 'car'):
        if name.startswith(klass + '_'):
            for size in ('small', 'medium', 'large'):
                prefix = f'{klass}_{size}_'
                if name.startswith(prefix):
                    metric = name[len(prefix):]
                    pair = {'precision_drop': ('tp', 'fp'), 'recall_drop': ('tp', 'fn')}.get(metric)
                    if pair is None:
                        return None
                    ci_clean = ratio_bounds(rows, 'clean', klass, size, pair[0], pair[1], alpha / 2, config)
                    ci_affected = ratio_bounds(rows, 'affected', klass, size, pair[0], pair[1], alpha / 2, config)
                    if ci_clean is None or ci_affected is None:
                        return None
                    return [ci_clean[0] - ci_affected[1], ci_clean[1] - ci_affected[0]]
    return None


def test_specs(config):
    limits = config['limits']
    specs = {}
    for name in config['status_tests']:
        if name == 'traffic_light_miss':
            specs[name] = {'direction': 'max', 'threshold': limits['traffic_light_miss_max']}
        elif name == 'traffic_light_color_error_given_detected':
            specs[name] = {'direction': 'max', 'threshold': limits['traffic_light_color_given_found_max']}
        elif name.endswith('_drop'):
            specs[name] = {'direction': 'max', 'threshold': limits['precision_recall_drop_max_pp']}
        elif name.startswith('clean_traffic_light_'):
            specs[name] = {'direction': 'min', 'threshold': limits['clean_traffic_light_detection_and_color_min']}
        elif name.startswith('clean_'):
            specs[name] = {'direction': 'min', 'threshold': limits['clean_person_car_precision_recall_min']}
        else:
            raise ValueError(name)
    return specs


def classify_intervals(checks, specs):
    violated = False
    unknown = set(checks) != set(specs)
    for name, spec in specs.items():
        interval = checks.get(name)
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            unknown = True
            continue
        low, high = interval
        if not finite_number(low) or not finite_number(high) or low > high or not finite_number(spec['threshold']):
            unknown = True
            continue
        cutoff = spec['threshold']
        if spec['direction'] == 'max':
            if above(low, cutoff):
                violated = True
            elif above(high, cutoff):
                unknown = True
        elif spec['direction'] == 'min':
            if below(high, cutoff):
                violated = True
            elif below(low, cutoff):
                unknown = True
        else:
            raise ValueError(spec['direction'])
    return 'F' if violated else ('Q' if unknown else 'S')


def classify_mode(scene_rows, config):
    alpha = config['alpha']['status_tail_budget_each']
    specs = test_specs(config)
    checks = {name: interval_for_test(scene_rows, name, alpha, config) for name in specs}
    status = classify_intervals(checks, specs)
    violated = status == 'F'
    unknown = status == 'Q'
    for row in scene_rows:
        physical = row.get('physical')
        if not isinstance(physical, dict):
            unknown = True
            continue
        for key, bound in (('p0', config['limits']['p0_max']), ('blur_pixels', config['limits']['blur_max_pixels'])):
            value = physical.get(key)
            if not finite_number(value):
                unknown = True
            elif above(value, bound):
                violated = True
    status = 'F' if violated else ('Q' if unknown else 'S')
    return {'status': status, 'intervals': {name: {'lower': value[0], 'upper': value[1]} if value else None for name, value in checks.items()}}


def loss(decision, status):
    if status == 'Q' or decision == 'U':
        return 1
    return 0 if (decision == 'A' and status == 'S') or (decision == 'R' and status == 'F') else 1


def delta_from_statuses(statuses, decisions):
    if not statuses or set(statuses) != set(decisions):
        return None
    return sum(loss(decisions[k]['analytic'], statuses[k]) - loss(decisions[k]['proposed'], statuses[k]) for k in statuses) / len(statuses)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def paired_cluster_bootstrap(scene_rows, decisions, config, repetitions=None, seed=20261120):
    if not scene_rows or not decisions:
        return {'point': None, 'lower': None, 'status': 'INSUFFICIENT'}
    if any(set(row['modes']) != set(decisions) for row in scene_rows):
        return {'point': None, 'lower': None, 'status': 'INCOMPLETE_MODES'}
    b = repetitions or config['statistical_contract']['bootstrap_repetitions']
    statuses = {mode: classify_mode([row['modes'][mode] for row in scene_rows], config)['status'] for mode in decisions}
    point = delta_from_statuses(statuses, decisions)
    rng = random.Random(seed)
    draws = []
    for _ in range(b):
        indices = [rng.randrange(len(scene_rows)) for _ in scene_rows]
        new_statuses = {mode: classify_mode([scene_rows[i]['modes'][mode] for i in indices], config)['status'] for mode in decisions}
        draws.append(delta_from_statuses(new_statuses, decisions))
    q = config['alpha']['paired_improvement_budget']
    lower = percentile(draws, q)
    return {'point': point, 'lower': lower, 'alpha': q, 'replicates': b, 'statuses': statuses,
            'status_recomputed_in_each_bootstrap': True,
            'interpretation': 'Percentile cluster bootstrap is an approximate interval; coverage requires validation, especially with discrete S/F/Q transitions.'}
