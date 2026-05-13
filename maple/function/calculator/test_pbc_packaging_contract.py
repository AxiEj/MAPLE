from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"


def test_pbc_optional_extras_are_named():
    text = PYPROJECT.read_text()

    assert "pbc-aimnet =" in text
    assert '"aimnet[ase]>=0.2,<0.3"' in text
    assert "pbc-mace =" in text
    assert '"mace-torch>=0.3.14,<0.4"' in text
    assert "pbc =" in text


def test_setuptools_discovers_maple_subpackages():
    text = PYPROJECT.read_text()

    assert "[tool.setuptools.packages.find]" in text
    assert 'include = ["maple", "maple.*"]' in text


def test_readme_documents_explicit_pbc_backends():
    text = README.read_text()

    assert "PBC backend selection is explicit" in text
    assert "aimnet2-pbc" in text
    assert "mace-omat-pbc" in text
    assert 'pip install -e ".[pbc]"' in text
