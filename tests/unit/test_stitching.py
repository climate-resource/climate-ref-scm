import numpy as np
import pandas as pd
import pytest

from climate_ref_scm.stitching import control_reference, normalise, stitch


def _series(years: range, value: float) -> pd.DataFrame:
    return pd.DataFrame({"World": np.full(len(years), value)}, index=pd.Index(list(years), name="year"))


def test_stitch_child_wins_overlap():
    child = _series(range(2010, 2020), 2.0)
    parent = _series(range(2000, 2015), 1.0)

    out = stitch([child, parent])

    assert out.index.tolist() == list(range(2000, 2020))
    assert (out.loc[2000:2009, "World"] == 1.0).all()
    assert (out.loc[2010:2019, "World"] == 2.0).all()


def test_stitch_rejects_gap():
    with pytest.raises(ValueError, match="missing years"):
        stitch([_series(range(2020, 2030), 2.0), _series(range(2000, 2015), 1.0)])


def test_control_reference_removes_linear_drift():
    years = np.arange(1, 201)
    control = pd.DataFrame({"World": 10.0 + 0.1 * years}, index=years)

    reference = control_reference(control, branch_year=50, n_years=5)

    np.testing.assert_allclose(reference["World"], 10.0 + 0.1 * np.arange(50, 55))


def test_control_reference_extends_at_edges():
    years = np.arange(1, 41)
    control = pd.DataFrame({"World": 10.0 + 0.1 * years}, index=years)

    # A centred window is incomplete for the first and last ten years of the control run.
    reference = control_reference(control, branch_year=1, n_years=40)

    np.testing.assert_allclose(reference["World"], 10.0 + 0.1 * years)


def test_control_reference_edges_resist_noise():
    years = np.arange(1, 101)
    # Alternating noise survives the running mean, so a two-point slope would swing widely.
    noise = np.where(years % 2 == 0, 1.0, -1.0)
    control = pd.DataFrame({"World": 10.0 + 0.01 * years + noise}, index=years)

    reference = control_reference(control, branch_year=1, n_years=100)

    np.testing.assert_allclose(reference["World"].iloc[:10], 10.0 + 0.01 * years[:10], atol=0.05)
    np.testing.assert_allclose(reference["World"].iloc[-10:], 10.0 + 0.01 * years[-10:], atol=0.05)


def test_control_reference_needs_years_after_branch():
    control = _series(range(1, 60), 0.0)

    with pytest.raises(ValueError, match="must cover years 55 to 64"):
        control_reference(control, branch_year=55, n_years=10)


def test_normalise_keeps_stitched_index():
    stitched = _series(range(1850, 1860), 5.0)
    control = _series(range(1, 200), 3.0)

    out = normalise(stitched, control, branch_year=100)

    assert out.index.equals(stitched.index)
    np.testing.assert_allclose(out["World"], 2.0)
