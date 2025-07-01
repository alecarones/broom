import pytest
import broom

@pytest.fixture(scope="session")
def broom_imported():
    """Fixture to ensure that the broom package is imported."""
    assert hasattr(broom, "__file__"), "broom package is not imported correctly"
    return broom

@pytest.fixture()
def config_all_path():
    """Fixture to provide the path to the configuration file with all component separations."""
    return "tests/data/config_all.yaml"

@pytest.fixture()
def config_simple_path():
    """Fixture to provide the path to the configuration file with just NILC."""
    return "tests/data/config_simple.yaml"
