import tempfile
import unittest
from collections import Counter
from pathlib import Path

from prepare.split_manifest import (
    assert_disjoint,
    get_design_family,
    read_manifest,
    split_train_validation,
)


def make_rows(
    family: str,
    count: int,
    start_index: int,
) -> list[tuple[str, str]]:
    rows = []

    for offset in range(count):
        sample_id = start_index + offset
        filename = (
            f"{sample_id}-{family}-a-1-c2-u0.8-m1-p1-f0.npy"
        )
        rows.append(
            (
                f"feature/{filename}",
                f"label/{filename}",
            )
        )

    return rows


class TestSplitManifest(unittest.TestCase):
    def setUp(self):
        self.rows = (
            make_rows("RISCY", 20, 0)
            + make_rows("RISCY-FPU", 20, 100)
        )

    def test_split_is_deterministic_and_disjoint(self):
        first_train, first_validation = split_train_validation(
            self.rows,
            validation_ratio=0.2,
            seed=7,
        )
        second_train, second_validation = split_train_validation(
            self.rows,
            validation_ratio=0.2,
            seed=7,
        )

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)

        self.assertEqual(len(first_train), 32)
        self.assertEqual(len(first_validation), 8)

        self.assertTrue(
            set(first_train).isdisjoint(first_validation)
        )
        self.assertEqual(
            set(first_train) | set(first_validation),
            set(self.rows),
        )

        validation_families = Counter(
            get_design_family(row)
            for row in first_validation
        )
        self.assertEqual(validation_families["RISCY"], 4)
        self.assertEqual(validation_families["RISCY-FPU"], 4)

    def test_different_seed_changes_validation_membership(self):
        _, first_validation = split_train_validation(
            self.rows,
            validation_ratio=0.2,
            seed=7,
        )
        _, second_validation = split_train_validation(
            self.rows,
            validation_ratio=0.2,
            seed=8,
        )

        self.assertNotEqual(
            set(first_validation),
            set(second_validation),
        )

    def test_read_manifest_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.csv"
            path.write_text(
                "feature/sample.npy\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_read_manifest_rejects_duplicate_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text(
                "feature/sample.npy,label/sample.npy\n"
                "feature/sample.npy,label/sample.npy\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_assert_disjoint_rejects_overlap(self):
        repeated_row = self.rows[0]

        with self.assertRaises(ValueError):
            assert_disjoint(
                "train",
                [repeated_row],
                "validation",
                [repeated_row],
            )


if __name__ == "__main__":
    unittest.main()