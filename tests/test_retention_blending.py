import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from scripts.train_candidate import select_retained_candidate


class RetentionBlendingTests(unittest.TestCase):
    def test_selects_first_passing_fraction_without_changing_anchor(self):
        candidate = SimpleNamespace(p={'w': np.array([10.])})
        anchor = {'w': np.array([2.])}
        failed = {'accepted': False, 'reasons': ['old answer lost']}
        passed = {'accepted': True, 'reasons': []}
        with patch('scripts.train_candidate.evaluate', side_effect=[{'raw': 1}, {'blended': 1}]), patch('scripts.train_candidate.decide', side_effect=[failed, passed]):
            after, decision, fraction, attempts = select_retained_candidate(candidate, anchor, {}, [])
        self.assertEqual(fraction, .75)
        self.assertTrue(decision['accepted'])
        self.assertEqual(len(attempts), 2)
        np.testing.assert_array_equal(candidate.p['w'], [8.])
        np.testing.assert_array_equal(anchor['w'], [2.])

    def test_rejection_restores_raw_candidate_and_original_reasons(self):
        candidate = SimpleNamespace(p={'w': np.array([10.])})
        failed = {'accepted': False, 'reasons': ['old answer lost']}
        with patch('scripts.train_candidate.evaluate', return_value={}), patch('scripts.train_candidate.decide', return_value=failed):
            _, decision, fraction, attempts = select_retained_candidate(candidate, {'w': np.array([2.])}, {}, [])
        self.assertFalse(decision['accepted'])
        self.assertEqual(decision['reasons'], failed['reasons'])
        self.assertEqual(len(attempts), 5)
        np.testing.assert_array_equal(candidate.p['w'], [10.])

    def test_retraining_never_blends_unrelated_initial_weights(self):
        candidate = SimpleNamespace(p={'w': np.array([10.])})
        failed = {'accepted': False, 'reasons': ['not learned']}
        with patch('scripts.train_candidate.evaluate', return_value={}), patch('scripts.train_candidate.decide', return_value=failed):
            _, _, _, attempts = select_retained_candidate(candidate, {'w': np.array([2.])}, {}, [], blend=False)
        self.assertEqual(len(attempts), 1)


if __name__ == '__main__':
    unittest.main()
