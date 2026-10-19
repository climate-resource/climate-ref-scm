"""
Regional means over the boxes used by MAGICC.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WORLD = "World"
NH_LAND = "World|Northern Hemisphere|Land"
NH_OCEAN = "World|Northern Hemisphere|Ocean"
SH_LAND = "World|Southern Hemisphere|Land"
SH_OCEAN = "World|Southern Hemisphere|Ocean"
REGIONS = (WORLD, NH_LAND, NH_OCEAN, SH_LAND, SH_OCEAN)


def region_weights(
    lat: xr.DataArray, areacella: xr.DataArray, sftlf: xr.DataArray
) -> dict[str, xr.DataArray]:
    """
    Area weights for each region.

    Cells centred on the equator count as northern hemisphere, as in netcdf-scm,
    so the four boxes always sum to the world.

    Parameters
    ----------
    lat
        Cell-centre latitude, broadcastable against the spatial grid
    areacella
        Cell area
    sftlf
        Land fraction in percent

    Returns
    -------
    :
        Weights keyed by region name
    """
    land = (sftlf / 100.0).clip(0.0, 1.0)
    ocean = 1.0 - land
    nh = (lat >= 0).astype("float64")
    sh = 1.0 - nh
    return {
        WORLD: areacella,
        NH_LAND: areacella * nh * land,
        NH_OCEAN: areacella * nh * ocean,
        SH_LAND: areacella * sh * land,
        SH_OCEAN: areacella * sh * ocean,
    }


def _open(paths: list[Path]) -> xr.Dataset:
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return xr.open_mfdataset(
        paths, combine="by_coords", decode_times=time_coder, data_vars="minimal", coords="minimal"
    )


def _on_grid(field: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    # Fixed fields can carry slightly different coordinate values from the same grid,
    # so they are matched by shape rather than aligned by label.
    if field.shape != template.shape:
        raise ValueError(f"Fixed field {field.name} has shape {field.shape}, expected {template.shape}")
    return template.copy(data=field.values.astype("float64"))


def annual_regional_means(
    variable_id: str, paths: list[Path], areacella_path: Path, sftlf_path: Path
) -> pd.DataFrame:
    """
    Annual means of a monthly variable over each region.

    Parameters
    ----------
    variable_id
        Variable to average
    paths
        Files holding the variable
    areacella_path
        File holding the cell areas
    sftlf_path
        File holding the land fraction

    Returns
    -------
    :
        Annual means indexed by integer year, with one column per region
    """
    with _open(paths) as ds, xr.open_dataset(areacella_path) as area_ds, xr.open_dataset(sftlf_path) as lf_ds:
        da = ds[variable_id].astype("float64")
        spatial_dims = [dim for dim in da.dims if dim != "time"]
        template = da.isel(time=0, drop=True)
        weights = region_weights(
            template["lat"],
            _on_grid(area_ds["areacella"], template),
            _on_grid(lf_ds["sftlf"], template),
        )

        months_per_year = da["time"].groupby("time.year").count().to_series()
        # One reduction over all regions, so the data is read once.
        stacked = xr.concat(list(weights.values()), dim=pd.Index(list(weights), name="region"))
        monthly = da.weighted(stacked.fillna(0.0)).mean(spatial_dims)
        annual = monthly.groupby("time.year").mean().load()

    out = pd.DataFrame(annual.transpose("year", "region").to_pandas())[months_per_year == 12]  # noqa: PLR2004
    out.columns.name = None
    out.index = out.index.astype(int)
    out.index.name = "year"
    return out.astype(np.float64)
