import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run statistical tests that require long stochastic trajectories",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="use --run-slow to run statistical tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
