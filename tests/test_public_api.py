"""Check the package's documented import surface."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import fieldline_tracing_proto as package

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _source_version():
    text = _PYPROJECT.read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)


def test_package_exports_public_workflow():
    assert package.__version__ == _source_version()
    try:
        installed = version("fieldline-tracing-proto")
    except PackageNotFoundError:
        installed = None  # running from a checkout without installation
    if installed is not None:
        assert package.__version__ == installed
    for name in package.__all__:
        assert callable(getattr(package, name))
