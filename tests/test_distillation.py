import copy
import unittest
import numpy as np

from tiny_model import build_from_pairs, softmax


class DistillationTests(unittest.TestCase):
    def test_identical_teacher_scales_supervised_gradient_on_old_rows(self):
        model = build_from_pairs([('質問', '回答です。')], d_model=4, n_layers=1)
        x = np.array([[0, 1, 2]])
        y = np.array([[1, 2, 3]])
        weights = np.ones_like(x, dtype=float)
        _, plain = model.loss_and_grads(x, y, weights)
        probabilities = softmax(model.forward(x)[0])
        _, retained = model.loss_and_grads(x, y, weights, probabilities, [True])
        for name in plain:
            np.testing.assert_allclose(retained[name], .1 * plain[name], atol=1e-12)

    def test_mixed_objective_gradient_matches_finite_difference(self):
        model = build_from_pairs([('質問', '回答です。')], d_model=4, n_layers=1)
        teacher = copy.deepcopy(model)
        teacher.p['bout'][0] += .5
        x = np.array([[0, 1], [2, 3]])
        y = np.array([[1, 2], [3, 4]])
        weights = np.array([[1., .5], [1., 1.]])
        probabilities = softmax(teacher.forward(x[:1])[0])
        loss, gradients = model.loss_and_grads(x, y, weights, probabilities, [True, False])
        eps = 1e-5
        model.p['bout'][0] += eps
        plus, _ = model.loss_and_grads(x, y, weights, probabilities, [True, False])
        model.p['bout'][0] -= 2 * eps
        minus, _ = model.loss_and_grads(x, y, weights, probabilities, [True, False])
        self.assertAlmostEqual(gradients['bout'][0], (plus - minus) / (2 * eps), places=7)

    def test_training_does_not_mutate_teacher_and_rejects_misaligned_vocab(self):
        pairs = [('古い質問', '既存の回答です。'), ('新しい質問', '追加の回答です。')]
        model = build_from_pairs(pairs, d_model=4, n_layers=1)
        teacher = copy.deepcopy(model)
        original = {k: v.copy() for k, v in teacher.p.items()}
        model.train(pairs, steps=3, batch=2, teacher=teacher, new_users={pairs[1][0]}, log=lambda _: None)
        for name in original:
            np.testing.assert_array_equal(teacher.p[name], original[name])
        teacher.chars = list(reversed(teacher.chars))
        with self.assertRaises(ValueError):
            model.train(pairs, steps=1, teacher=teacher)


if __name__ == '__main__':
    unittest.main()
