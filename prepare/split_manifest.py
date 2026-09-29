from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path


DESIGN_PATTERN = re.compile(
    r"^\d+-(RISCY-FPU|RISCY|zero-riscy)-"
)


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for line_number, fields in enumerate(reader, start=1):
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected two columns, "
                    f"got {len(fields)}"
                )

            feature_path, label_path = fields
            rows.append((feature_path, label_path))

    feature_paths = [feature for feature, _ in rows]

    if len(feature_paths) != len(set(feature_paths)):
        raise ValueError(f"{path}: duplicate feature paths detected")

    return rows


def get_design_family(row: tuple[str, str]) -> str:
    feature_path, _ = row
    filename = Path(feature_path).name
    match = DESIGN_PATTERN.match(filename)

    if match is None:
        raise ValueError(
            f"Cannot determine design family from {feature_path}"
        )

    return match.group(1)


def split_train_validation(
    rows: list[tuple[str, str]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    groups = defaultdict(list)

    for row in rows:
        groups[get_design_family(row)].append(row)

    random_generator = random.Random(seed)
    train_rows = []
    validation_rows = []

    for family in sorted(groups):
        family_rows = sorted(groups[family])
        random_generator.shuffle(family_rows)

        validation_count = max(
            1,
            int(len(family_rows) * validation_ratio + 0.5),
        )

        validation_rows.extend(family_rows[:validation_count])
        train_rows.extend(family_rows[validation_count:])

    return sorted(train_rows), sorted(validation_rows)


def assert_disjoint(
    first_name: str,
    first_rows: list[tuple[str, str]],
    second_name: str,
    second_rows: list[tuple[str, str]],
) -> None:
    first_features = {feature for feature, _ in first_rows}
    second_features = {feature for feature, _ in second_rows}
    overlap = first_features & second_features

    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(
            f"{first_name} and {second_name} overlap: {example}"
        )


def write_manifest(
    path: Path,
    rows: list[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(rows)


def print_summary(
    name: str,
    rows: list[tuple[str, str]],
) -> None:
    counts = defaultdict(int)

    for row in rows:
        counts[get_design_family(row)] += 1

    details = ", ".join(
        f"{family}={counts[family]}"
        for family in sorted(counts)
    )
    print(f"{name}: total={len(rows)} ({details})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic validation split."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--validation-out", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source_rows = read_manifest(args.source)
    test_rows = read_manifest(args.test_manifest)

    train_rows, validation_rows = split_train_validation(
        source_rows,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    assert_disjoint("train", train_rows, "validation", validation_rows)
    assert_disjoint("train", train_rows, "test", test_rows)
    assert_disjoint("validation", validation_rows, "test", test_rows)

    if len(train_rows) + len(validation_rows) != len(source_rows):
        raise RuntimeError("Split changed the source sample count")

    write_manifest(args.train_out, train_rows)
    write_manifest(args.validation_out, validation_rows)

    print_summary("train", train_rows)
    print_summary("validation", validation_rows)
    print_summary("test", test_rows)
    print("split validation: OK")


if __name__ == "__main__":
    main()