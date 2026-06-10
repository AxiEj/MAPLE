import ast
from pathlib import Path


MD_ROOT = Path(__file__).resolve().parents[3] / "maple" / "function" / "dispatcher" / "md"


def test_internal_md_modules_do_not_import_utils_shim():
    offenders = []
    for path in sorted(MD_ROOT.rglob("*.py")):
        if path.name in {"__init__.py", "utils.py"}:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _is_utils_shim_import(node):
                offenders.append(f"{path.relative_to(MD_ROOT)}:{node.lineno}")

    assert offenders == []


def _is_utils_shim_import(node: ast.ImportFrom) -> bool:
    if node.module == "maple.function.dispatcher.md.utils":
        return True
    return node.module == "utils" and node.level > 0
