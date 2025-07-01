import pytest
import broom

@pytest.fixture(scope="session")
def broom_imported():
    """Fixture to ensure that the broom package is imported."""
    assert hasattr(broom, "__file__"), "broom package is not imported correctly"
    return broom

@pytest.fixture()
def config_path():
    """Fixture to provide the path to the configuration file."""
    return "broom/configs/config_groundbased.yaml"
