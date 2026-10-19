import numpy as np
import pandas as pd
from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.metric_values import SeriesMetricValue

from climate_ref_scm import provider
from climate_ref_scm.diagnostics import (
    OUTPUT_FILENAME,
    STITCHED_TAS,
    STITCHED_TOA_FLUXES,
    StitchedTimeseries,
)
from climate_ref_scm.regions import NH_LAND, NH_OCEAN, SH_LAND, SH_OCEAN, WORLD

from ..conftest import BRANCH_YEAR, HISTORICAL_YEARS, SSP_YEARS, control_value, forcing

# Cell offsets from the fixture: SH land 0, SH ocean 1, NH ocean 2, NH land 3.
REGION_OFFSET = {WORLD: 1.5, SH_LAND: 0.0, SH_OCEAN: 1.0, NH_OCEAN: 2.0, NH_LAND: 3.0}
YEARS = np.arange(HISTORICAL_YEARS[0], SSP_YEARS[1] + 1)
CONTROL_YEARS = YEARS + BRANCH_YEAR - HISTORICAL_YEARS[0]


def _solve(catalog: pd.DataFrame, diagnostic: StitchedTimeseries = STITCHED_TAS):
    executions = solve_executions({SourceDatasetType.CMIP6: catalog}, diagnostic, provider)
    return {e.datasets[SourceDatasetType.CMIP6].selector_dict()["experiment_id"]: e for e in executions}


def _run(catalog: pd.DataFrame, diagnostic: StitchedTimeseries, definition_factory):
    execution = _solve(catalog, diagnostic)["ssp245"]
    definition = definition_factory(diagnostic=diagnostic, cmip6=execution.datasets[SourceDatasetType.CMIP6])
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    diagnostic.execute(definition)
    result = diagnostic.build_execution_result(definition)

    assert result.successful
    out = pd.read_csv(definition.output_directory / OUTPUT_FILENAME)
    assert [int(c) for c in out.columns if c.isdigit()] == YEARS.tolist()
    series = SeriesMetricValue.load_from_json(definition.to_output_path(result.series_filename))
    return out, series


def _values(out: pd.DataFrame, variable: str) -> pd.DataFrame:
    return out[out["variable"] == variable].set_index("region")[YEARS.astype(str)]


def test_solve_adds_whole_chain(synthetic_catalog):
    executions = _solve(synthetic_catalog)

    assert set(executions) == {"historical", "ssp245"}
    ssp = executions["ssp245"].datasets[SourceDatasetType.CMIP6].datasets
    assert sorted(ssp["experiment_id"].unique()) == ["historical", "piControl", "ssp245"]
    assert sorted(ssp["variable_id"].unique()) == ["areacella", "sftlf", "tas"]


def test_solve_adds_chain_for_each_variable(synthetic_catalog):
    ssp = _solve(synthetic_catalog, STITCHED_TOA_FLUXES)["ssp245"].datasets[SourceDatasetType.CMIP6].datasets

    chains = ssp.groupby("variable_id")["experiment_id"].unique().map(sorted)
    for variable_id in ("rsdt", "rsut", "rlut"):
        assert chains[variable_id] == ["historical", "piControl", "ssp245"]


def test_solve_drops_group_without_control(synthetic_catalog):
    catalog = synthetic_catalog[
        ~((synthetic_catalog["experiment_id"] == "piControl") & (synthetic_catalog["variable_id"] == "rsut"))
    ]

    assert _solve(catalog, STITCHED_TOA_FLUXES) == {}
    assert set(_solve(catalog)) == {"historical", "ssp245"}


def test_execute_recovers_forcing(synthetic_catalog, definition_factory):
    out, series = _run(synthetic_catalog, STITCHED_TAS, definition_factory)

    absolute = _values(out, "Surface Air Temperature")
    anomaly = _values(out, "Surface Air Temperature Change")
    for region, offset in REGION_OFFSET.items():
        expected = control_value(CONTROL_YEARS) + forcing(YEARS) + offset
        np.testing.assert_allclose(absolute.loc[region], expected, atol=1e-9)
        # The constant cell offset cancels against the control run.
        np.testing.assert_allclose(anomaly.loc[region], forcing(YEARS), atol=1e-9)

    assert len(series) == 2 * len(REGION_OFFSET)
    assert {s.dimensions["statistic"] for s in series} == {"absolute", "anomaly"}


def test_execute_derives_net_flux(synthetic_catalog, definition_factory):
    out, series = _run(synthetic_catalog, STITCHED_TOA_FLUXES, definition_factory)

    # Every flux carries the same values, so rsdt - rsut - rlut is minus one of them.
    absolute = _values(out, "TOA Net Downward Flux")
    anomaly = _values(out, "TOA Net Downward Flux Change")
    for region, offset in REGION_OFFSET.items():
        expected = control_value(CONTROL_YEARS) + forcing(YEARS) + offset
        np.testing.assert_allclose(absolute.loc[region], -expected, atol=1e-9)
        np.testing.assert_allclose(anomaly.loc[region], -forcing(YEARS), atol=1e-9)

    assert set(out.loc[out["variable"].str.startswith("TOA"), "unit"]) == {"W m^-2"}
    assert {s.dimensions["metric"] for s in series} == {"rsdt", "rsut", "rlut", "rtnt"}
    assert len(series) == 4 * 2 * len(REGION_OFFSET)
