from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError


EXPECTED_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(EXPECTED_CLASSES)}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg"}
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_ALGORITHM = "sha256_rank_per_class_v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Caminho inseguro no ZIP: {member.filename}")
        archive.extractall(destination)


def find_dataset_dir(root: Path) -> Path | None:
    expected = set(EXPECTED_CLASSES)
    for current_path, directory_names, _ in os.walk(root):
        if expected.issubset(set(directory_names)):
            return Path(current_path)
    return None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_image_id(relative_path: str) -> str:
    return sha256_bytes(relative_path.encode("utf-8"))


def split_rank(relative_path: str) -> str:
    return sha256_bytes(f"{SEED}|{relative_path}".encode("utf-8"))


def inventory_dataset(dataset_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    unexpected_paths: list[str] = []
    for path in sorted(item for item in dataset_dir.rglob("*") if item.is_file()):
        relative_path = path.relative_to(dataset_dir).as_posix()
        parts = Path(relative_path).parts
        class_name = parts[0] if len(parts) >= 2 else ""
        extension = path.suffix.lower()
        expected_location = class_name in CLASS_TO_INDEX and extension in ALLOWED_EXTENSIONS
        row = {
            "image_id": stable_image_id(relative_path),
            "relative_path": relative_path,
            "class_name": class_name,
            "class_index": CLASS_TO_INDEX.get(class_name, ""),
            "file_size_bytes": path.stat().st_size,
            "file_sha256": "",
            "pixel_sha256": "",
            "width": "",
            "height": "",
            "mode": "",
            "channels": "",
            "format": "",
            "readable": False,
            "eligible": False,
            "exclusion_reason": "",
            "canonical_relative_path": "",
        }
        if not expected_location:
            row["exclusion_reason"] = "unexpected_path_or_extension"
            unexpected_paths.append(relative_path)
            rows.append(row)
            continue
        file_bytes = path.read_bytes()
        row["file_sha256"] = sha256_bytes(file_bytes)
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                image_format = image.format or ""
                pixel_payload = (
                    f"{mode}|{width}|{height}|".encode("utf-8") + image.tobytes()
                )
                row.update(
                    {
                        "pixel_sha256": sha256_bytes(pixel_payload),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "channels": len(image.getbands()),
                        "format": image_format,
                        "readable": True,
                        "eligible": True,
                    }
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            row["exclusion_reason"] = f"unreadable:{type(exc).__name__}"
        rows.append(row)
    return rows, unexpected_paths


def apply_duplicate_policy(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    readable_rows = [row for row in rows if row["readable"]]
    groups_by_file: dict[str, list[dict]] = defaultdict(list)
    groups_by_pixel: dict[str, list[dict]] = defaultdict(list)
    for row in readable_rows:
        groups_by_file[row["file_sha256"]].append(row)
        groups_by_pixel[row["pixel_sha256"]].append(row)

    byte_duplicate_groups = [group for group in groups_by_file.values() if len(group) > 1]
    pixel_duplicate_groups = [group for group in groups_by_pixel.values() if len(group) > 1]
    conflicts = []
    for duplicate_type, groups in (
        ("file_sha256", byte_duplicate_groups),
        ("pixel_sha256", pixel_duplicate_groups),
    ):
        for group in groups:
            labels = sorted({row["class_name"] for row in group})
            if len(labels) > 1:
                conflicts.append(
                    {
                        "duplicate_type": duplicate_type,
                        "labels": labels,
                        "paths": sorted(row["relative_path"] for row in group),
                    }
                )
    if conflicts:
        raise RuntimeError(
            "Duplicatas com rótulos conflitantes encontradas; divisão bloqueada: "
            + json.dumps(conflicts[:5], ensure_ascii=False)
        )

    duplicate_records: list[dict] = []
    seen_pixel_hashes: set[str] = set()
    for group in sorted(
        pixel_duplicate_groups,
        key=lambda values: min(row["relative_path"] for row in values),
    ):
        ordered = sorted(group, key=lambda row: row["relative_path"])
        canonical = ordered[0]
        seen_pixel_hashes.add(canonical["pixel_sha256"])
        for row in ordered:
            row["canonical_relative_path"] = canonical["relative_path"]
            duplicate_records.append(
                {
                    "pixel_sha256": row["pixel_sha256"],
                    "file_sha256": row["file_sha256"],
                    "class_name": row["class_name"],
                    "relative_path": row["relative_path"],
                    "canonical_relative_path": canonical["relative_path"],
                    "is_canonical": row is canonical,
                    "same_bytes_as_canonical": row["file_sha256"]
                    == canonical["file_sha256"],
                }
            )
            if row is not canonical:
                row["eligible"] = False
                row["exclusion_reason"] = "duplicate_pixels_same_label"

    for row in readable_rows:
        if row["pixel_sha256"] not in seen_pixel_hashes:
            row["canonical_relative_path"] = row["relative_path"]

    return byte_duplicate_groups, pixel_duplicate_groups, duplicate_records


def make_split(rows: list[dict]) -> list[dict]:
    eligible = [row for row in rows if row["eligible"]]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_class[row["class_name"]].append(row)

    missing_classes = [name for name in EXPECTED_CLASSES if not by_class[name]]
    if missing_classes:
        raise RuntimeError(f"Classes ausentes após auditoria: {missing_classes}")

    split_rows: list[dict] = []
    for class_name in EXPECTED_CLASSES:
        class_rows = sorted(
            by_class[class_name],
            key=lambda row: (split_rank(row["relative_path"]), row["relative_path"]),
        )
        total = len(class_rows)
        train_count = int(total * SPLIT_RATIOS["train"])
        validation_count = int(total * SPLIT_RATIOS["validation"])
        boundaries = {
            "train": (0, train_count),
            "validation": (train_count, train_count + validation_count),
            "test": (train_count + validation_count, total),
        }
        for split_name, (start, end) in boundaries.items():
            for row in class_rows[start:end]:
                split_rows.append(
                    {
                        "image_id": row["image_id"],
                        "relative_path": row["relative_path"],
                        "class_name": row["class_name"],
                        "class_index": row["class_index"],
                        "file_sha256": row["file_sha256"],
                        "pixel_sha256": row["pixel_sha256"],
                        "split": split_name,
                        "split_rank_sha256": split_rank(row["relative_path"]),
                    }
                )
    return sorted(split_rows, key=lambda row: row["relative_path"])


def verify_split(rows: list[dict], split_rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["eligible"]]
    eligible_ids = {row["image_id"] for row in eligible}
    split_ids = {row["image_id"] for row in split_rows}
    if eligible_ids != split_ids:
        raise RuntimeError("A divisão não cobre exatamente o inventário elegível.")

    ids_by_split = {
        name: {row["image_id"] for row in split_rows if row["split"] == name}
        for name in SPLIT_RATIOS
    }
    hashes_by_split = {
        name: {row["pixel_sha256"] for row in split_rows if row["split"] == name}
        for name in SPLIT_RATIOS
    }
    intersections = {}
    names = list(SPLIT_RATIOS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left}__{right}"
            intersections[key] = {
                "image_ids": len(ids_by_split[left] & ids_by_split[right]),
                "pixel_hashes": len(hashes_by_split[left] & hashes_by_split[right]),
            }
    if any(value for pair in intersections.values() for value in pair.values()):
        raise RuntimeError(f"Sobreposição entre conjuntos: {intersections}")

    reconstructed = make_split(rows)
    reproducible = reconstructed == split_rows
    if not reproducible:
        raise RuntimeError("A reprodução determinística da divisão divergiu.")

    return {
        "eligible_inventory_covered": True,
        "intersections": intersections,
        "reproduced_identically": True,
    }


def create_examples_figure(dataset_dir: Path, split_rows: list[dict], output_path: Path) -> list[dict]:
    selected = []
    for class_name in EXPECTED_CLASSES:
        candidates = sorted(
            (
                row
                for row in split_rows
                if row["split"] == "train" and row["class_name"] == class_name
            ),
            key=lambda row: (row["split_rank_sha256"], row["relative_path"]),
        )
        selected.append(candidates[0])

    figure, axes = plt.subplots(2, 5, figsize=(15, 6))
    for axis, row in zip(axes.ravel(), selected):
        with Image.open(dataset_dir / row["relative_path"]) as image:
            axis.imshow(image)
        axis.set_title(row["class_name"], fontsize=9)
        axis.axis("off")
    figure.suptitle("Exemplo determinístico de treino por classe", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return [
        {
            "class_name": row["class_name"],
            "relative_path": row["relative_path"],
            "selection_rule": "lowest_split_rank_sha256_in_train",
        }
        for row in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--extract-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    zip_path = args.zip.resolve()
    extract_root = args.extract_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Diretório de saída já existe: {output_dir}")
    output_dir.mkdir(parents=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = find_dataset_dir(extract_root)
    if dataset_dir is None:
        safe_extract(zip_path, extract_root)
        dataset_dir = find_dataset_dir(extract_root)
    if dataset_dir is None:
        raise RuntimeError("Raiz do EuroSAT RGB não localizada após extração.")

    rows, unexpected_paths = inventory_dataset(dataset_dir)
    byte_groups, pixel_groups, duplicate_records = apply_duplicate_policy(rows)
    split_rows = make_split(rows)
    verification = verify_split(rows, split_rows)

    inventory_fields = list(rows[0].keys())
    split_fields = list(split_rows[0].keys())
    exclusion_rows = [row for row in rows if not row["eligible"]]
    duplicate_fields = [
        "pixel_sha256",
        "file_sha256",
        "class_name",
        "relative_path",
        "canonical_relative_path",
        "is_canonical",
        "same_bytes_as_canonical",
    ]
    write_csv(output_dir / "inventory.csv", rows, inventory_fields)
    write_csv(output_dir / "exclusions.csv", exclusion_rows, inventory_fields)
    write_csv(output_dir / "duplicate_groups.csv", duplicate_records, duplicate_fields)
    write_csv(output_dir / "split.csv", split_rows, split_fields)

    split_config = {
        "version": 1,
        "seed": SEED,
        "ratios": SPLIT_RATIOS,
        "algorithm": SPLIT_ALGORITHM,
        "ranking_definition": "sha256(f'{seed}|{relative_path}') per class",
        "allocation_definition": "floor(0.70*n), floor(0.15*n), remainder to test",
        "class_order": EXPECTED_CLASSES,
        "duplicate_policy": "keep lexicographically first decoded-pixel duplicate within same label",
    }
    (output_dir / "split_config.json").write_text(
        json.dumps(split_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "classes.json").write_text(
        json.dumps(CLASS_TO_INDEX, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    examples = create_examples_figure(
        dataset_dir, split_rows, output_dir / "train_examples.png"
    )
    (output_dir / "train_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    readable_rows = [row for row in rows if row["readable"]]
    control_names = [
        "inventory.csv",
        "exclusions.csv",
        "duplicate_groups.csv",
        "split.csv",
        "split_config.json",
        "classes.json",
        "train_examples.json",
    ]
    summary = {
        "stage": 3,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "name": zip_path.name,
            "size_bytes": zip_path.stat().st_size,
            "sha256": sha256_file(zip_path),
        },
        "dataset_root_name": dataset_dir.name,
        "runtime": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "pillow": Image.__version__,
            "numpy": np.__version__,
        },
        "inventory": {
            "total_files": len(rows),
            "readable_images": len(readable_rows),
            "eligible_images": sum(bool(row["eligible"]) for row in rows),
            "excluded_images": sum(not bool(row["eligible"]) for row in rows),
            "unexpected_paths": unexpected_paths,
            "unreadable_count": sum(not bool(row["readable"]) for row in rows),
            "class_counts_before_exclusions": dict(
                sorted(Counter(row["class_name"] for row in readable_rows).items())
            ),
            "class_counts_eligible": dict(
                sorted(
                    Counter(
                        row["class_name"] for row in rows if row["eligible"]
                    ).items()
                )
            ),
            "dimensions": dict(
                sorted(Counter(f"{row['width']}x{row['height']}" for row in readable_rows).items())
            ),
            "modes": dict(sorted(Counter(row["mode"] for row in readable_rows).items())),
            "formats": dict(
                sorted(Counter(row["format"] for row in readable_rows).items())
            ),
        },
        "duplicates": {
            "byte_duplicate_groups": len(byte_groups),
            "byte_duplicate_images": sum(len(group) for group in byte_groups),
            "pixel_duplicate_groups": len(pixel_groups),
            "pixel_duplicate_images": sum(len(group) for group in pixel_groups),
            "conflicting_label_groups": 0,
            "excluded_noncanonical_pixel_duplicates": sum(
                row["exclusion_reason"] == "duplicate_pixels_same_label" for row in rows
            ),
        },
        "geographic_metadata": {
            "dataset_described_as_georeferenced_by_authors": True,
            "usable_coordinates_or_scene_ids_in_rgb_archive": False,
            "evidence": "Archive contains class directories and JPEG files only; filenames expose class plus sequence number.",
            "split_implication": "Stratified image-level split; it does not demonstrate generalization to unseen regions.",
        },
        "split": {
            "config": split_config,
            "counts": dict(sorted(Counter(row["split"] for row in split_rows).items())),
            "counts_by_class": {
                class_name: dict(
                    sorted(
                        Counter(
                            row["split"]
                            for row in split_rows
                            if row["class_name"] == class_name
                        ).items()
                    )
                )
                for class_name in EXPECTED_CLASSES
            },
            "verification": verification,
        },
        "control_file_sha256": {
            name: sha256_file(output_dir / name) for name in control_names
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
