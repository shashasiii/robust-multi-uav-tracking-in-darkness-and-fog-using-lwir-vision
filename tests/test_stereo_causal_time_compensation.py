from pathlib import Path
import unittest

from stereo.tracking_stereo_distance_online import (
    RectifiedMotionSample,
    RectifiedMotionState,
    StereoFramePair,
    compensate_rectified_centers,
    fit_causal_rectified_velocity,
)


def motion_state(points):
    return RectifiedMotionState(
        samples=[RectifiedMotionSample(timestamp_ns=t, x=x, y=y) for t, x, y in points],
        last_seen_frame=len(points) - 1,
    )


class CausalTimeCompensationTests(unittest.TestCase):
    def setUp(self):
        self.frame_pair = StereoFramePair(
            Path("left.png"), Path("right.png"), 1_000_000_000, 1_020_000_000
        )

    def compensate(self, left_state, right_state, **overrides):
        kwargs = dict(
            frame_pair=self.frame_pair,
            left_raw=(100.0, 50.0),
            right_raw=(80.0, 50.0),
            left_state=left_state,
            right_state=right_state,
            enabled=True,
            min_samples=2,
            decay=0.70,
            max_gap_ms=120.0,
            max_shift_px=12.0,
            bound_margin_px=16.0,
            frame_size=(640, 512),
            min_disparity_px=2.0,
            max_disparity_px=260.0,
        )
        kwargs.update(overrides)
        return compensate_rectified_centers(**kwargs)

    def test_signed_delta_and_midpoint(self):
        self.assertEqual(self.frame_pair.stereo_delta_ns, 20_000_000)
        self.assertEqual(self.frame_pair.common_timestamp_ns, 1_010_000_000)
        reversed_pair = StereoFramePair(
            Path("left.png"), Path("right.png"), 1_020_000_000, 1_000_000_000
        )
        self.assertEqual(reversed_pair.stereo_delta_ns, -20_000_000)
        self.assertEqual(reversed_pair.common_timestamp_ns, 1_010_000_000)

    def test_weighted_fit_recovers_constant_velocity(self):
        state = motion_state([
            (900_000_000, 10.0, 20.0),
            (950_000_000, 11.0, 19.5),
            (1_000_000_000, 12.0, 19.0),
        ])
        estimate, reason = fit_causal_rectified_velocity(state, 1_000_000_000, 2, 0.70, 120.0)
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.vx_px_s, 20.0, places=6)
        self.assertAlmostEqual(estimate.vy_px_s, -10.0, places=6)

    def test_predicts_both_eyes_to_common_time(self):
        left = motion_state([(950_000_000, 99.0, 50.0), (1_000_000_000, 100.0, 50.0)])
        right = motion_state([(970_000_000, 79.0, 50.0), (1_020_000_000, 80.0, 50.0)])
        result = self.compensate(left, right)
        self.assertTrue(result.valid)
        self.assertTrue(result.applied)
        self.assertAlmostEqual(result.left_compensated[0], 100.2, places=6)
        self.assertAlmostEqual(result.right_compensated[0], 79.8, places=6)
        self.assertAlmostEqual(result.disparity_compensated_px, 20.4, places=6)

    def test_insufficient_history_is_invalid(self):
        left = motion_state([(1_000_000_000, 100.0, 50.0)])
        right = motion_state([(1_020_000_000, 80.0, 50.0)])
        result = self.compensate(left, right)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "left_temporal_insufficient_history")

    def test_shift_gate_rejects_prediction(self):
        left = motion_state([(990_000_000, 80.0, 50.0), (1_000_000_000, 100.0, 50.0)])
        right = motion_state([(1_010_000_000, 60.0, 50.0), (1_020_000_000, 80.0, 50.0)])
        result = self.compensate(left, right, max_shift_px=5.0)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "temporal_left_shift_exceeded")

    def test_disabled_mode_keeps_raw_contract(self):
        result = self.compensate(None, None, enabled=False)
        self.assertTrue(result.valid)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "disabled")
        self.assertIsNone(result.disparity_compensated_px)
        self.assertEqual(result.disparity_raw_px, 20.0)


if __name__ == "__main__":
    unittest.main()
