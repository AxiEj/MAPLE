import os
import re
from typing import Dict, List


class PartReader:
    """
    Read ML/MM partition definitions from a block-style file.

    Expected format:
        [CORE]
        1 2 3 4

        [ENV]
        5 6 7 8

    Rules:
      - file_path must be absolute.
      - natoms must be a positive integer.
      - [CORE] is required.
      - [ENV] is optional; if omitted, it is built as the complement of CORE.
      - Indices in the file are 1-based; returned indices are 0-based.
      - Empty lines and lines starting with '#' are ignored.

    Returns:
        Dict[str, List[int]] with keys 'core_indices' and 'env_indices'.
    """

    _valid_blocks = {"CORE", "ENV"}
    _header_pattern = re.compile(r"^\[(CORE|ENV)\]$", re.IGNORECASE)

    def __new__(cls, file_path: str, natoms: int) -> Dict[str, List[int]]:
        if not os.path.isabs(file_path):
            raise ValueError(f"PartReader requires an absolute path. Got: {file_path}")

        if not isinstance(natoms, int) or natoms <= 0:
            raise ValueError(f"PartReader requires natoms to be a positive integer. Got: {natoms}")

        resolved_path = cls._resolve_file_path(file_path)
        if not resolved_path:
            raise ValueError(f"PartReader: file not found: {file_path}")

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            raise ValueError(f"PartReader: failed to read {resolved_path}: {e}")

        blocks = {"CORE": [], "ENV": []}
        seen_blocks = set()
        current_block = None

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            header_match = cls._header_pattern.match(stripped)
            if header_match:
                current_block = header_match.group(1).upper()
                seen_blocks.add(current_block)
                continue

            if stripped.startswith("[") and stripped.endswith("]"):
                block_name = stripped[1:-1].strip()
                raise ValueError(
                    f"PartReader: unknown block '[{block_name}]' at line {line_num} in {resolved_path}"
                )

            if current_block is None:
                raise ValueError(
                    f"PartReader: found indices before any block header at line {line_num} in {resolved_path}"
                )

            indices = []
            for token in stripped.split():
                try:
                    indices.extend(cls._expand_index_token(token))
                except ValueError as e:
                    raise ValueError(f"PartReader: {e} at line {line_num}")

            for index in indices:
                if index < 1 or index > natoms:
                    raise ValueError(
                        f"PartReader: atom index {index} out of range 1..{natoms} at line {line_num}"
                    )

            blocks[current_block].extend(indices)

        if not blocks["CORE"]:
            raise ValueError(f"PartReader: [CORE] block is required and cannot be empty in {resolved_path}")

        if "ENV" in seen_blocks and not blocks["ENV"]:
            raise ValueError(f"PartReader: [ENV] block cannot be empty in {resolved_path}")

        core_indices = cls._normalize_indices(blocks["CORE"], "CORE")
        core_set = set(core_indices)

        if "ENV" in seen_blocks:
            env_indices = cls._normalize_indices(blocks["ENV"], "ENV")
            env_set = set(env_indices)
            overlap = sorted(core_set & env_set)
            if overlap:
                overlap_1based = ", ".join(str(idx + 1) for idx in overlap)
                raise ValueError(
                    f"PartReader: CORE and ENV overlap on atom indices: {overlap_1based}"
                )

            full_set = set(range(natoms))
            covered_set = core_set | env_set
            if covered_set != full_set:
                missing = sorted(full_set - covered_set)
                missing_1based = ", ".join(str(idx + 1) for idx in missing)
                raise ValueError(
                    f"PartReader: explicit [ENV] block leaves missing atom indices: {missing_1based}"
                )
        else:
            env_indices = [idx for idx in range(natoms) if idx not in core_set]

        return {
            "core_indices": core_indices,
            "env_indices": env_indices,
        }

    @staticmethod
    def _expand_index_token(token: str) -> List[int]:
        if "-" not in token:
            try:
                return [int(token)]
            except ValueError:
                raise ValueError(f"invalid atom index token '{token}'")

        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid atom index token '{token}'")

        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            raise ValueError(f"invalid atom index token '{token}'")

        if start > end:
            raise ValueError(f"invalid range '{token}'")

        return list(range(start, end + 1))

    @staticmethod
    def _normalize_indices(indices: List[int], block_name: str) -> List[int]:
        seen = set()
        normalized = []

        for index in indices:
            zero_based = index - 1
            if zero_based in seen:
                raise ValueError(f"PartReader: duplicate atom index {index} in [{block_name}]")
            seen.add(zero_based)
            normalized.append(zero_based)

        return normalized

    @staticmethod
    def _resolve_file_path(file_path: str) -> str:
        if os.path.isfile(file_path):
            return file_path

        directory = os.path.dirname(file_path)
        target_name = os.path.basename(file_path).lower()

        if not os.path.isdir(directory):
            return None

        for entry in os.listdir(directory):
            if entry.lower() == target_name:
                candidate = os.path.join(directory, entry)
                if os.path.isfile(candidate):
                    return candidate

        return None
