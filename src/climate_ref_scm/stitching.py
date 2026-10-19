"""
Stitch an experiment onto its parents and express it relative to the control run.
"""

import numpy as np
import pandas as pd
from loguru import logger

RUNNING_MEAN_YEARS = 21


def stitch(series: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Join annual series from a child and its ancestors into one continuous series.

    Where years overlap the child wins, so a scenario keeps its own values
    and takes only the earlier years from historical.

    Parameters
    ----------
    series
        Annual series indexed by year, ordered from the child up through its ancestors

    Returns
    -------
    :
        Continuous annual series

    Raises
    ------
    ValueError
        The joined series has a gap
    """
    out = series[0]
    for parent in series[1:]:
        out = pd.concat([parent[~parent.index.isin(out.index)], out]).sort_index()

    expected = np.arange(out.index.min(), out.index.max() + 1)
    missing = np.setdiff1d(expected, out.index)
    if missing.size:
        raise ValueError(f"Stitched series is missing years {missing.tolist()}")
    return out


def control_reference(
    control: pd.DataFrame, branch_year: int, n_years: int, window: int = RUNNING_MEAN_YEARS
) -> pd.DataFrame:
    """
    Compute the running mean of the control run, aligned to the years after the branch.

    Each value is the centred `window`-year mean of the control run
    at the matching point after the branch.
    Near the ends of the control run, where a full window is not available,
    the running mean is extended along a straight line fitted to its nearest `window` full-window values.

    Parameters
    ----------
    control
        Annual control run series indexed by year
    branch_year
        Year in the control run at which the first stitched year starts
    n_years
        Number of years to produce
    window
        Length of the running mean in years

    Returns
    -------
    :
        Reference values indexed by control run year

    Raises
    ------
    ValueError
        The control run does not cover the years after the branch,
        or is too short for a single full window
    """
    years = np.arange(branch_year, branch_year + n_years)
    missing = np.setdiff1d(years, control.index)
    if missing.size:
        raise ValueError(
            f"Control run must cover years {years[0]} to {years[-1]}, but is missing {missing.size} of them"
        )

    half = window // 2
    available = control.loc[branch_year - half : branch_year + n_years - 1 + half]
    if not np.array_equal(available.index, np.arange(available.index[0], available.index[-1] + 1)):
        raise ValueError("Control run has gaps around the branch")

    running = available.rolling(window, center=True).mean().dropna()
    min_points = 2
    if len(running) < min_points:
        raise ValueError(f"Control run is too short for a {window}-year running mean")

    return pd.DataFrame(
        {
            column: _extend_linearly(running.index.to_numpy(), running[column].to_numpy(), years, window)
            for column in running.columns
        },
        index=years,
    )


def _extend_linearly(x: np.ndarray, y: np.ndarray, target: np.ndarray, fit_points: int) -> np.ndarray:
    out: np.ndarray = np.interp(target, x, y)
    below, above = target < x[0], target > x[-1]
    if below.any():
        out[below] = np.polyval(np.polyfit(x[:fit_points], y[:fit_points], 1), target[below])
    if above.any():
        out[above] = np.polyval(np.polyfit(x[-fit_points:], y[-fit_points:], 1), target[above])
    return out


def normalise(
    stitched: pd.DataFrame, control: pd.DataFrame, branch_year: int, window: int = RUNNING_MEAN_YEARS
) -> pd.DataFrame:
    """
    Anomalies of the stitched series from the control run running mean.

    The first stitched year lines up with `branch_year` in the control run,
    so the reference also removes any drift in the control.

    Parameters
    ----------
    stitched
        Continuous annual series indexed by year
    control
        Annual control run series indexed by year
    branch_year
        Year in the control run that lines up with the first stitched year
    window
        Length of the running mean in years

    Returns
    -------
    :
        Anomalies with the same index as `stitched`
    """
    reference = control_reference(control, branch_year, len(stitched), window).set_axis(stitched.index)
    logger.debug(f"Normalised against control years {branch_year} to {branch_year + len(stitched) - 1}")
    return stitched - reference
