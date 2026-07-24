from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hatexplain_sparsemax.config import ExperimentConfig  # noqa: E402


def slug(value: object) -> str:
    text = str(value).lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_-]+", "-", text).strip("-")


def axis_combinations(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    keys = list(axes)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(axes[key] for key in keys))
    ]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def materialize(matrix_path: Path, output_root: Path) -> dict[str, Any]:
    specification = read_json(matrix_path)
    experiment = specification["experiment"]
    base_path = (matrix_path.parent / specification["base_config"]).resolve()
    base = read_json(base_path)
    fixed = specification.get("overrides", {})
    conditions = specification.get("conditions", [{"name": "", "overrides": {}}])
    combinations = axis_combinations(specification.get("axes", {}))

    destination = output_root / experiment
    destination.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    for condition in conditions:
        for combination in combinations:
            payload = base | fixed | condition.get("overrides", {}) | combination
            parts = [experiment]
            if condition.get("name"):
                parts.append(slug(condition["name"]))
            parts.extend(f"{slug(key)}-{slug(value)}" for key, value in combination.items())
            variant = "__".join(parts)
            payload["variant"] = variant
            payload["output_dir"] = f"runs/{experiment}/{variant}"

            config = ExperimentConfig(**payload)
            config.validate()
            output_path = destination / f"{variant}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            generated.append(str(output_path.relative_to(REPOSITORY_ROOT)))

    index = {
        "experiment": experiment,
        "matrix": str(matrix_path.relative_to(REPOSITORY_ROOT)),
        "count": len(generated),
        "configs": generated,
    }
    index_path = destination / "index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize checked-in experiment matrices")
    parser.add_argument("matrices", nargs="*", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        help="materialize every experiments/*/matrix.json manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "generated",
    )
    args = parser.parse_args()

    matrices = list(args.matrices)
    if args.all:
        matrices.extend(sorted(Path(__file__).parent.glob("*/matrix.json")))
    matrices = list(dict.fromkeys(path.resolve() for path in matrices))
    if not matrices:
        parser.error("pass at least one matrix path or use --all")

    indices = [materialize(path, args.output.resolve()) for path in matrices]
    print(json.dumps(indices, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
