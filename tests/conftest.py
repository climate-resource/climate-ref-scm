"""
Synthetic CMIP6 datasets with a drifting control run.

The control drifts linearly and the children add a known forcing signal on top,
so the expected anomaly is exactly the forcing signal.
"""

from collections.abc import Callable
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

SOURCE_ID = "TEST-ESM"
VARIANT = "r1i1p1f1"
CALENDAR = "noleap"
TIME_UNITS = "days since 0001-01-01"
LAT = np.array([-45.0, 45.0])
LON = np.array([90.0, 270.0])
AREA = np.array([[1.0, 1.0], [1.0, 1.0]])
# One land and one ocean cell in each hemisphere.
SFTLF = np.array([[100.0, 0.0], [0.0, 100.0]])
# Each cell keeps a distinct offset so regional means differ.
CELL_OFFSET = np.array([[0.0, 1.0], [2.0, 3.0]])
CONTROL_BASE = 287.0
CONTROL_DRIFT = 0.002
CONTROL_YEARS = (1, 300)
BRANCH_YEAR = 101
HISTORICAL_YEARS = (1850, 1879)
SSP_YEARS = (1880, 1899)


def control_value(control_year: np.ndarray) -> np.ndarray:
    return CONTROL_BASE + CONTROL_DRIFT * control_year


def forcing(year: np.ndarray) -> np.ndarray:
    return 0.05 * (year - HISTORICAL_YEARS[0])


def _write_tas(path: Path, years: tuple[int, int], values: Callable[[np.ndarray], np.ndarray]) -> None:
    times = [
        cftime.datetime(year, month, 15, calendar=CALENDAR)
        for year in range(years[0], years[1] + 1)
        for month in range(1, 13)
    ]
    year = np.array([t.year for t in times], dtype=float)
    data = values(year)[:, None, None] + CELL_OFFSET[None, :, :]
    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), data, {"units": "K"})},
        coords={"time": times, "lat": LAT, "lon": LON},
    )
    ds["time"].encoding.update(units=TIME_UNITS, calendar=CALENDAR)
    ds.to_netcdf(path)


def _write_fx(path: Path, name: str, values: np.ndarray) -> None:
    xr.Dataset({name: (("lat", "lon"), values)}, coords={"lat": LAT, "lon": LON}).to_netcdf(path)


def _row(
    path: Path, experiment_id: str, variable_id: str, table_id: str, **kwargs: object
) -> dict[str, object]:
    instance_id = (
        f"CMIP6.CMIP.TEST.{SOURCE_ID}.{experiment_id}.{VARIANT}.{table_id}.{variable_id}.gn.v20200101"
    )
    return {
        "instance_id": instance_id,
        "source_id": SOURCE_ID,
        "experiment_id": experiment_id,
        "variant_label": VARIANT,
        "member_id": VARIANT,
        "table_id": table_id,
        "variable_id": variable_id,
        "grid_label": "gn",
        "version": "v20200101",
        "calendar": CALENDAR,
        "time_units": TIME_UNITS,
        "path": str(path),
        "finalised": True,
        "start_time": None,
        "end_time": None,
        "parent_source_id": None,
        "parent_experiment_id": None,
        "parent_variant_label": None,
        "parent_time_units": None,
        "branch_time_in_parent": None,
        **kwargs,
    }


def _tas_row(
    path: Path, experiment_id: str, years: tuple[int, int], parent: str | None, branch: float | None
):
    return _row(
        path,
        experiment_id,
        "tas",
        "Amon",
        start_time=cftime.datetime(years[0], 1, 15, calendar=CALENDAR),
        end_time=cftime.datetime(years[1], 12, 15, calendar=CALENDAR),
        parent_source_id=SOURCE_ID if parent else None,
        parent_experiment_id=parent,
        parent_variant_label=VARIANT if parent else None,
        parent_time_units=TIME_UNITS if parent else None,
        branch_time_in_parent=branch,
    )


@pytest.fixture
def synthetic_catalog(tmp_path: Path) -> pd.DataFrame:
    """Catalog of piControl, historical and ssp245 tas plus the fixed fields."""
    offset = BRANCH_YEAR - HISTORICAL_YEARS[0]

    def child(year: np.ndarray) -> np.ndarray:
        return control_value(year + offset) + forcing(year)

    _write_tas(tmp_path / "piControl.nc", CONTROL_YEARS, control_value)
    _write_tas(tmp_path / "historical.nc", HISTORICAL_YEARS, child)
    _write_tas(tmp_path / "ssp245.nc", SSP_YEARS, child)
    _write_fx(tmp_path / "areacella.nc", "areacella", AREA)
    _write_fx(tmp_path / "sftlf.nc", "sftlf", SFTLF)

    branch_days = cftime.date2num(
        cftime.datetime(BRANCH_YEAR, 1, 1, calendar=CALENDAR), TIME_UNITS, calendar=CALENDAR
    )
    rows = [
        _tas_row(tmp_path / "piControl.nc", "piControl", CONTROL_YEARS, "piControl-spinup", 0.0),
        _tas_row(tmp_path / "historical.nc", "historical", HISTORICAL_YEARS, "piControl", branch_days),
        _tas_row(tmp_path / "ssp245.nc", "ssp245", SSP_YEARS, "historical", 0.0),
        _row(tmp_path / "areacella.nc", "piControl", "areacella", "fx"),
        _row(tmp_path / "sftlf.nc", "piControl", "sftlf", "fx"),
    ]
    return pd.DataFrame(rows)
