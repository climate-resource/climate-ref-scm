import numpy as np
import pandas as pd
from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.metric_values import SeriesMetricValue

from climate_ref_scm import provider
from climate_ref_scm.diagnostics import OUTPUT_FILENAME, StitchedTimeseries
from climate_ref_scm.regions import NH_LAND, NH_OCEAN, SH_LAND, SH_OCEAN, WORLD

from ..conftest import BRANCH_YEAR, HISTORICAL_YEARS, SSP_YEARS, control_value, forcing

# Cell offsets from the fixture: SH land 0, SH ocean 1, NH ocean 2, NH land 3.
REGION_OFFSET = {WORLD: 1.5, SH_LAND: 0.0, SH_OCEAN: 1.0, NH_OCEAN: 2.0, NH_LAND: 3.0}


def _solve(catalog: pd.DataFrame):
    executions = solve_executions({SourceDatasetType.CMIP6: catalog}, StitchedTimeseries(), provider)
    return {e.datasets[SourceDatasetType.CMIP6].selector_dict()["experiment_id"]: e for e in executions}


def test_solve_adds_whole_chain(synthetic_catalog):
    executions = _solve(synthetic_catalog)

    assert set(executions) == {"historical", "ssp245"}
    ssp = executions["ssp245"].datasets[SourceDatasetType.CMIP6].datasets
    assert sorted(ssp["experiment_id"].unique()) == ["historical", "piControl", "ssp245"]
    assert sorted(ssp["variable_id"].unique()) == ["areacella", "sftlf", "tas"]


def test_solve_drops_group_without_control(synthetic_catalog):
    catalog = synthetic_catalog[
        ~((synthetic_catalog["experiment_id"] == "piControl") & (synthetic_catalog["variable_id"] == "tas"))
    ]

    assert _solve(catalog) == {}


def test_execute_recovers_forcing(synthetic_catalog, definition_factory):
    execution = _solve(synthetic_catalog)["ssp245"]
    diagnostic = StitchedTimeseries()
    definition = definition_factory(diagnostic=diagnostic, cmip6=execution.datasets[SourceDatasetType.CMIP6])
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    diagnostic.execute(definition)
    result = diagnostic.build_execution_result(definition)

    assert result.successful
    out = pd.read_csv(definition.output_directory / OUTPUT_FILENAME)
    years = np.arange(HISTORICAL_YEARS[0], SSP_YEARS[1] + 1)
    assert [int(c) for c in out.columns if c.isdigit()] == years.tolist()

    absolute = out[out["variable"] == "Surface Air Temperature"].set_index("region")
    anomaly = out[out["variable"] == "Surface Air Temperature Change"].set_index("region")
    control_years = years + BRANCH_YEAR - HISTORICAL_YEARS[0]
    for region, offset in REGION_OFFSET.items():
        expected = control_value(control_years) + forcing(years) + offset
        np.testing.assert_allclose(absolute.loc[region, years.astype(str)], expected, atol=1e-9)
        # The constant cell offset cancels against the control run.
        np.testing.assert_allclose(anomaly.loc[region, years.astype(str)], forcing(years), atol=1e-9)

    series = SeriesMetricValue.load_from_json(definition.to_output_path(result.series_filename))
    assert len(series) == 2 * len(REGION_OFFSET)
    assert {s.dimensions["statistic"] for s in series} == {"absolute", "anomaly"}
