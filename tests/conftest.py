# This file simply forces pytest_benchmark to use milliseconds as units
import pytest
@pytest.hookspec(firstresult=True)
def pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    return 'm', 1e3
