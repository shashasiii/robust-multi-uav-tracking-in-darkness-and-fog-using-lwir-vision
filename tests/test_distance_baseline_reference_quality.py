import csv
from pathlib import Path
import tempfile
import unittest

from distance_baseline.compare_realsense_reference import (
    OUTPUT_FIELDS,
    apply_reference_quality,
    comparison_key,
)


class ReferenceQualityTests(unittest.TestCase):
    def test_comparison_schema_uses_filtered_depth_names(self):
        self.assertIn("stereo_filtered_depth_m", OUTPUT_FIELDS)
        self.assertIn("filtered_error_m", OUTPUT_FIELDS)
        self.assertNotIn("stereo_distance_m", OUTPUT_FIELDS)
        self.assertNotIn("ema_error_m", OUTPUT_FIELDS)

    def test_multi_target_keys_and_temporal_windows_are_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "comparison.csv"
            rows = []
            for frame_idx in range(3):
                for pair_label, depth in (
                    ("D-D-1", (4.0, 10.0, 4.0)[frame_idx]),
                    ("D-D-2", 10.0),
                ):
                    row = {field: "" for field in OUTPUT_FIELDS}
                    row.update(
                        frame_idx=str(frame_idx),
                        image_name_left=f"frame_{frame_idx}.png",
                        class_id="0",
                        class_name="drone",
                        pair_label=pair_label,
                        reference_valid="1",
                        reference_reason="ok",
                        rs_depth_m=str(depth),
                        rs_depth_iqr_m="0.1",
                        rs_valid_pixels="30",
                    )
                    rows.append(row)
            self.assertEqual(len({comparison_key(row) for row in rows}), len(rows))
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            apply_reference_quality(path, max_iqr_m=0.20, min_valid_pixels=25)

            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                result = list(csv.DictReader(fh))
            by_target_frame = {
                (row["pair_label"], int(row["frame_idx"])): row for row in result
            }
            self.assertEqual(
                by_target_frame[("D-D-1", 1)]["reference_temporal_warning"], "1"
            )
            self.assertEqual(
                by_target_frame[("D-D-2", 1)]["reference_temporal_warning"], "0"
            )

    def test_quality_gate_and_temporal_warning_do_not_rewrite_depth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "comparison.csv"
            rows = []
            for frame_idx, depth, iqr, pixels in (
                (0, 4.0, 0.10, 30),
                (1, 10.0, 0.10, 30),
                (2, 4.1, 0.10, 30),
                (3, 4.2, 0.30, 20),
            ):
                row = {field: "" for field in OUTPUT_FIELDS}
                row.update(
                    frame_idx=str(frame_idx),
                    image_name_left=f"frame_{frame_idx}.png",
                    reference_valid="1",
                    reference_reason="ok",
                    rs_depth_m=str(depth),
                    rs_depth_iqr_m=str(iqr),
                    rs_valid_pixels=str(pixels),
                )
                rows.append(row)
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            apply_reference_quality(path, max_iqr_m=0.20, min_valid_pixels=25)

            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                result = list(csv.DictReader(fh))
            self.assertEqual([row["rs_depth_m"] for row in result], ["4.0", "10.0", "4.1", "4.2"])
            self.assertEqual(result[0]["reference_quality_valid"], "1")
            self.assertEqual(result[1]["reference_temporal_warning"], "1")
            self.assertEqual(result[3]["reference_quality_valid"], "0")
            self.assertIn("quality_depth_iqr_above_limit", result[3]["reference_quality_reason"])
            self.assertIn("quality_insufficient_valid_depth_pixels", result[3]["reference_quality_reason"])


if __name__ == "__main__":
    unittest.main()
