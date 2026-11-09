from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stereo.tracking_stereo_distance_online import collect_stereo_pairs


def image_name(prefix: str, timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return f"thermal_{prefix}_image_{seconds}_{nanoseconds:09d}.png"


def write_frames(directory: Path, prefix: str, timestamps_ns: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for timestamp_ns in timestamps_ns:
        (directory / image_name(prefix, timestamp_ns)).touch()


class StereoTimestampPairingTests(unittest.TestCase):
    def test_searches_entire_window_when_nearest_frames_are_consumed(self):
        base = 1_700_000_000_000_000_000
        right = [base, base + 2_000_000, base + 4_000_000]
        left = [base + 2_000_000, base + 2_050_000, base + 2_100_000]

        with TemporaryDirectory() as temp:
            root = Path(temp)
            left_dir = root / "left"
            right_dir = root / "right"
            write_frames(left_dir, "inferno", left)
            write_frames(right_dir, "grayscale", right)
            pairs = collect_stereo_pairs(
                left_dir, right_dir, 0, 20.0, sync_phase_tolerance_ms=0.0
            )
            self.assertEqual(len(pairs), 3)
            self.assertEqual({pair.right_timestamp_ns for pair in pairs}, set(right))

    def test_searches_past_phase_invalid_nearest_candidate(self):
        base = 1_700_000_000_000_000_000
        period = 40_000_000
        left = [base + index * period for index in range(6)]
        right = [base + index * period + 7_000_000 for index in range(6)]
        right.append(base + 5 * period)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            left_dir = root / "left"
            right_dir = root / "right"
            write_frames(left_dir, "inferno", left)
            write_frames(right_dir, "grayscale", right)
            pairs = collect_stereo_pairs(
                left_dir,
                right_dir,
                0,
                20.0,
                sync_phase_tolerance_ms=5.0,
                sync_phase_window=31,
                sync_phase_min_samples=5,
            )
            self.assertEqual(len(pairs), 6)
            self.assertTrue(all(pair.stereo_delta_ns == 7_000_000 for pair in pairs))

    def test_phase_guard_prevents_frame_period_cascade(self):
        base = 1_700_000_000_000_000_000
        period = 40_000_000
        right = [base + index * period for index in range(11)]
        normal_left = [timestamp + 7_000_000 for timestamp in right]
        conflict_left = right[5] - 8_000_000
        left = sorted(normal_left + [conflict_left])

        with TemporaryDirectory() as temp:
            root = Path(temp)
            left_dir = root / "left"
            right_dir = root / "right"
            write_frames(left_dir, "inferno", left)
            write_frames(right_dir, "grayscale", right)

            unsafe = collect_stereo_pairs(
                left_dir, right_dir, 0, 35.0, sync_phase_tolerance_ms=0.0
            )
            self.assertTrue(any(pair.stereo_delta_ns >= 30_000_000 for pair in unsafe))

            audit = {}
            fixed = collect_stereo_pairs(
                left_dir,
                right_dir,
                0,
                20.0,
                sync_phase_tolerance_ms=5.0,
                sync_phase_window=31,
                sync_phase_min_samples=5,
                audit=audit,
            )
            self.assertEqual(len(fixed), len(normal_left))
            self.assertTrue(all(pair.stereo_delta_ns == -7_000_000 for pair in fixed))
            self.assertEqual(audit["phase_rejected_left"], 1)
            self.assertEqual(audit["no_candidate_left"], 0)

    def test_positive_v22_style_phase_is_preserved(self):
        base = 1_700_000_000_000_000_000
        period = 40_000_000
        left = [base + index * period for index in range(20)]
        right = [timestamp + 17_000_000 for timestamp in left]

        with TemporaryDirectory() as temp:
            root = Path(temp)
            left_dir = root / "left"
            right_dir = root / "right"
            write_frames(left_dir, "inferno", left)
            write_frames(right_dir, "grayscale", right)
            audit = {}
            pairs = collect_stereo_pairs(
                left_dir,
                right_dir,
                0,
                20.0,
                sync_phase_tolerance_ms=5.0,
                sync_phase_window=31,
                sync_phase_min_samples=5,
                audit=audit,
            )
            self.assertEqual(len(pairs), 20)
            self.assertTrue(all(pair.stereo_delta_ns == 17_000_000 for pair in pairs))
            self.assertEqual(audit["phase_rejected_left"], 0)


if __name__ == "__main__":
    unittest.main()
