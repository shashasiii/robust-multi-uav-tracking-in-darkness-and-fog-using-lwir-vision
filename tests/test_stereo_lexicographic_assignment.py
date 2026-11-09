import unittest

import numpy as np

from stereo.tracking_stereo_distance_online import solve_lexicographic_assignment


class StereoLexicographicAssignmentTests(unittest.TestCase):
    def test_maximises_cardinality_before_cost(self):
        cost = np.full((8, 8), np.inf, dtype=np.float64)
        for index in range(7):
            cost[index, index] = 0.0
        cost[7, 0] = 160.0
        cost[0, 7] = 160.0

        pairs = solve_lexicographic_assignment(cost, cost_limit=160.0)

        self.assertEqual(len(pairs), 8)
        self.assertEqual(sum(value for _, _, value in pairs), 320.0)

    def test_minimises_cost_at_fixed_cardinality(self):
        cost = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)

        pairs = solve_lexicographic_assignment(cost, cost_limit=10.0)

        self.assertEqual({(row, col) for row, col, _ in pairs}, {(0, 1), (1, 0)})
        self.assertEqual(sum(value for _, _, value in pairs), 3.0)


if __name__ == "__main__":
    unittest.main()
