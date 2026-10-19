import numpy as np
import xarray as xr

from climate_ref_scm.regions import NH_LAND, NH_OCEAN, REGIONS, SH_LAND, SH_OCEAN, WORLD, region_weights


def test_boxes_sum_to_world():
    lat = xr.DataArray([-30.0, 0.0, 30.0], dims="lat")
    area = xr.DataArray(np.ones((3, 2)), dims=("lat", "lon"))
    sftlf = xr.DataArray([[100.0, 25.0], [50.0, 0.0], [0.0, 75.0]], dims=("lat", "lon"))

    weights = region_weights(lat, area, sftlf)

    assert set(weights) == set(REGIONS)
    boxes = weights[NH_LAND] + weights[NH_OCEAN] + weights[SH_LAND] + weights[SH_OCEAN]
    xr.testing.assert_allclose(boxes, weights[WORLD])


def test_equator_is_northern_hemisphere():
    lat = xr.DataArray([0.0], dims="lat")
    area = xr.DataArray([[2.0]], dims=("lat", "lon"))
    sftlf = xr.DataArray([[100.0]], dims=("lat", "lon"))

    weights = region_weights(lat, area, sftlf)

    assert float(weights[NH_LAND].sum()) == 2.0
    assert float(weights[SH_LAND].sum()) == 0.0
