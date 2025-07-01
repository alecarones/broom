import pytest
import broom.clusters

def test_import_clusters():
    assert hasattr(broom.clusters, "__file__")

if hasattr(broom.clusters, "get_scalar_tracer"):
    def test_get_scalar_tracer_runs():
        # Provide minimal dummy arguments
        broom.clusters.get_scalar_tracer(None)
