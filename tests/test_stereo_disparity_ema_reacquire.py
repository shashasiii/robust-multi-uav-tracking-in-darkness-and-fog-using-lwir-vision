import unittest

from stereo.tracking_stereo_distance_online import (
    CSV_FIELDS,
    StereoPairState,
    smooth_disparity_with_reacquire,
)


def update(state, frame, disparity, *, changed=False):
    depth = None if disparity is None else 36.8 / disparity
    return smooth_disparity_with_reacquire(
        depth,
        disparity,
        state,
        frame,
        changed,
        alpha=0.35,
        jump_ratio=0.50,
        jump_abs_px=2.0,
        reacquire_samples=3,
        stability_ratio=0.10,
        stability_abs_px=0.50,
        max_frame_gap=2,
    )


class DisparityEmaReacquireTests(unittest.TestCase):
    def make_state(self):
        return StereoPairState(1, 0, 10, 20, 0)

    def test_current_csv_schema_uses_filtered_depth_name(self):
        self.assertIn("filtered_depth_m", CSV_FIELDS)
        self.assertNotIn("distance_m", CSV_FIELDS)

    def test_normal_disparity_is_smoothed(self):
        state = self.make_state()
        first = update(state, 1, 8.0)
        second = update(state, 2, 9.0)
        self.assertTrue(first[1])
        self.assertTrue(second[1])
        self.assertAlmostEqual(second[4], 8.35)
        self.assertFalse(second[6])

    def test_three_stable_jump_samples_reacquire(self):
        state = self.make_state()
        update(state, 1, 20.0)
        one = update(state, 2, 4.0, changed=True)
        two = update(state, 3, 4.2)
        three = update(state, 4, 4.1)
        self.assertFalse(one[1])
        self.assertFalse(two[1])
        self.assertTrue(three[1])
        self.assertEqual(three[3], "disparity_ema_reacquired")
        self.assertAlmostEqual(three[4], 4.1)
        self.assertTrue(three[6])

    def test_id_change_without_geometry_jump_keeps_continuity(self):
        state = self.make_state()
        update(state, 1, 8.0)
        result = update(state, 2, 8.2, changed=True)
        self.assertTrue(result[1])
        self.assertEqual(result[3], "ok")
        self.assertFalse(result[6])

    def test_invalid_input_does_not_replace_ema(self):
        state = self.make_state()
        update(state, 1, 8.0)
        result = update(state, 2, None)
        self.assertFalse(result[1])
        self.assertEqual(result[3], "disparity_input_invalid")
        self.assertEqual(state.disparity_ema, 8.0)


if __name__ == "__main__":
    unittest.main()
