"""Check the package's documented import surface."""

from importlib.metadata import version

import fieldline_tracing_proto as package


def test_package_exports_public_workflow():
    assert package.__version__ == version("fieldline-tracing-proto")
    for name in package.__all__:
        assert callable(getattr(package, name))
