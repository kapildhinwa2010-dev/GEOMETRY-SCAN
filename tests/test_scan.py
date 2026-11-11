import numpy as np
import pytest

from geometry_scan.scan import anisotropy, geometry_report, intrinsic_dimension, layer_geometry


def _line_states():
    e0 = np.zeros(10)
    e0[0] = 1.0
    return np.outer(np.arange(1, 9) * 1.0, e0)


def _gaussian_states():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 10))


def _cone_states():
    return np.outer(np.arange(1, 6) * 1.0, np.ones(6))


def test_line_states():
    s = _line_states()
    assert intrinsic_dimension(s) == pytest.approx(1.0)
    assert anisotropy(s) > 0.99
    entry = layer_geometry(s, name="line")
    assert entry["pc1_variance_share"] == pytest.approx(1.0)
    assert entry["dim"] == 10
    assert entry["samples"] == 8
    assert entry["mean_norm"] == pytest.approx(4.5)


def test_gaussian_states():
    s = _gaussian_states()
    assert intrinsic_dimension(s) > 7.0
    assert abs(anisotropy(s)) < 0.3
    entry = layer_geometry(s)
    assert entry["pc1_variance_share"] < 0.5


def test_cone_states():
    s = _cone_states()
    assert anisotropy(s) == pytest.approx(1.0)
    assert intrinsic_dimension(s) == pytest.approx(1.0)


def test_geometry_report():
    names = ["line", "cloud", "cone"]
    r = geometry_report([_line_states(), _gaussian_states(), _cone_states()], names=names)
    assert r["num_layers"] == 3
    assert r["lowest_dimension_layer"] == 0
    assert r["highest_anisotropy_layer"] == 0
    assert [e["name"] for e in r["layers"]] == names


def test_degenerate_states():
    flat = np.ones((4, 5))
    assert intrinsic_dimension(flat) == 0.0
    entry = layer_geometry(flat)
    assert entry["pc1_variance_share"] == 0.0


def test_errors():
    with pytest.raises(ValueError):
        intrinsic_dimension(np.zeros(5))
    with pytest.raises(ValueError):
        anisotropy(np.ones((1, 4)))
    with pytest.raises(ValueError):
        geometry_report([])
    with pytest.raises(ValueError):
        geometry_report([np.ones((4, 4))], names=["a", "b"])