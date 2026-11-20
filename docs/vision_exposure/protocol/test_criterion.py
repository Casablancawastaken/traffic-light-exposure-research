import copy
import json
import math
import unittest
from pathlib import Path

from criterion_example import aggregate_forecasts, calibration_adjustment, decide_exposure, empirical_upper_residual, example, explicit_zero_adjustments, calibrate_by_mode_scene_records
from statistical_contract import classify_intervals, classify_mode, exact_zero_upper, paired_cluster_bootstrap, test_specs


ROOT = Path(__file__).resolve().parent


class CriterionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'criterion_config.json').read_text(encoding='utf-8'))
        group = {size: {'n_gt': 1, 'p_miss': 0.01, 'fp_expected': 0.01} for size in ('small', 'medium', 'large')}
        scene = {'n_light': 1, 'p_light_miss': 0.01, 'p_color_error_given_found': 0.01,
                 'fp_light_expected': 0.01, 'objects': {'person': group, 'car': group}}
        cls.good = aggregate_forecasts([scene], explicit_zero_adjustments([scene]))

    def test_nan_p0_cannot_allow(self):
        self.assertEqual(decide_exposure(self.good, self.good, float('nan'), 1, self.config), 'U')
        self.assertEqual(decide_exposure(self.good, self.good, float('inf'), 1, self.config), 'U')

    def test_known_violation_precedes_unknown(self):
        incomplete = copy.deepcopy(self.good)
        incomplete['traffic_light']['color_error_given_found'] = None
        self.assertEqual(decide_exposure(incomplete, self.good, 0.8, 1, self.config), 'R')
        self.assertEqual(decide_exposure(incomplete, self.good, float('nan'), 1, self.config), 'U')

    def test_inclusive_boundary_and_floating_error(self):
        a = copy.deepcopy(self.good)
        c = copy.deepcopy(self.good)
        c['size_groups']['person_small']['recall'] = 0.9
        a['size_groups']['person_small']['recall'] = 0.85
        self.assertEqual(decide_exposure(a, c, 0.05, 3, self.config), 'A')

    def test_absolute_clean_per_class_not_each_size(self):
        a = copy.deepcopy(self.good)
        c = copy.deepcopy(self.good)
        for obj in (a, c):
            obj['size_groups']['person_medium'].update({'tp': 1, 'fn': 0, 'fp': 1, 'recall': 1.0, 'precision': 0.5})
            obj['size_groups']['person_small'].update({'tp': 3, 'fn': 0, 'fp': 1, 'recall': 1.0, 'precision': 0.75})
            obj['size_groups']['person_large'].update({'tp': 3, 'fn': 0, 'fp': 0, 'recall': 1.0, 'precision': 1.0})
        self.assertEqual(decide_exposure(a, c, 0, 1, self.config), 'A')

    def test_calibration_missing_vs_zero(self):
        self.assertIsNone(empirical_upper_residual([0], [0], min_support=30))
        self.assertIsNone(empirical_upper_residual([float('nan')] * 30, [0] * 30, min_support=30))
        self.assertEqual(calibration_adjustment([0] * 30, [0] * 30, self.config)['status'], 'ESTIMATED_ZERO')
        self.assertEqual(calibration_adjustment([0] * 29, [0] * 29, self.config)['status'], 'UNDEFINED')
        invalid = copy.deepcopy(self.good)
        invalid['calibration_status'] = 'U'
        self.assertEqual(decide_exposure(invalid, self.good, 0, 1, self.config), 'U')

    def test_invalid_calibration_adjustment(self):
        group = {'small': {'n_gt': 1, 'p_miss': 0.01, 'fp_expected': 0.01}}
        scene = {'n_light': 1, 'p_light_miss': 0.01, 'p_color_error_given_found': 0.01,
                 'objects': {'person': group}}
        result = aggregate_forecasts([scene], {'light_miss_prob': None})
        self.assertEqual(result['calibration_status'], 'U')
        result = aggregate_forecasts([scene], {'light_miss_prob': float('nan')})
        self.assertEqual(result['calibration_status'], 'U')
        result = aggregate_forecasts([scene], {}, calibration_required=True)
        self.assertEqual(result['calibration_status'], 'U')

    def test_calibration_never_pools_modes_or_duplicate_scenes(self):
        entries = ([{'mode': 'r1', 'metric': 'miss', 'scene_id': i, 'observed': 0, 'predicted': 0} for i in range(30)] +
                   [{'mode': 'r2', 'metric': 'miss', 'scene_id': i, 'observed': 0, 'predicted': 0} for i in range(29)])
        out = calibrate_by_mode_scene_records(entries, self.config)
        self.assertEqual(out['r1']['miss']['status'], 'ESTIMATED_ZERO')
        self.assertEqual(out['r2']['miss']['status'], 'UNDEFINED')
        out2 = calibrate_by_mode_scene_records(entries[:30] + [entries[0]], self.config)
        self.assertEqual(out2['r1']['miss']['status'], 'UNDEFINED')

    def test_nan_interval_and_priority_and_field_order(self):
        specs = {'unknown': {'direction': 'max', 'threshold': 0.05},
                 'violated': {'direction': 'max', 'threshold': 0.05}}
        checks = {'unknown': [float('nan'), float('nan')], 'violated': [0.2, 0.3]}
        self.assertEqual(classify_intervals(checks, specs), 'F')
        self.assertEqual(classify_intervals(dict(reversed(list(checks.items()))), specs), 'F')
        self.assertEqual(classify_intervals({'unknown': [float('nan'), 0.01], 'violated': [0, 0.01]}, specs), 'Q')
        self.assertEqual(classify_intervals({'unknown': [0.05, 0.05], 'violated': [0.05, 0.05]}, specs), 'S')

    def test_registry_21_aggregate_limits(self):
        specs = test_specs(self.config)
        self.assertEqual(len(specs), 21)
        self.assertEqual(len([x for x in specs if x.endswith('_drop')]), 12)
        self.assertEqual(len([x for x in specs if x.startswith('clean_person_') or x.startswith('clean_car_')]), 4)
        self.assertEqual(len([x for x in specs if x.startswith('clean_traffic_light_')]), 3)
        expected = .025 / (96 * 21 * 2)
        self.assertAlmostEqual(self.config['alpha']['status_tail_budget_each'], expected)

    def test_zero_error_risk_not_zero(self):
        self.assertAlmostEqual(exact_zero_upper(20, .05), 1 - .05 ** (1 / 20))
        self.assertGreater(exact_zero_upper(20, .05), .13)

    def test_scene_count_to_interval_and_bootstrap(self):
        report = example(self.config)
        self.assertEqual(report['status_interval_examples']['confirmed_F'], 'F')
        self.assertTrue(report['decision_example']['false_allow_conservative_upper'] == 0.75)
        from criterion_example import synthetic_scene
        scenes = [{'modes': {k: synthetic_scene(k) for k in ('safe', 'unsafe', 'undefined')}} for _ in range(12)]
        status = classify_mode([item['modes']['unsafe'] for item in scenes], self.config)
        self.assertEqual(status['status'], 'F')
        decisions = {'safe': {'analytic': 'A', 'proposed': 'A'},
                     'unsafe': {'analytic': 'A', 'proposed': 'R'},
                     'undefined': {'analytic': 'R', 'proposed': 'U'}}
        result = paired_cluster_bootstrap(scenes, decisions, self.config, repetitions=15)
        self.assertTrue(result['status_recomputed_in_each_bootstrap'])
        self.assertTrue(math.isfinite(result['lower']))


if __name__ == '__main__':
    unittest.main()
