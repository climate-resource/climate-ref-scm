import numpy as np
import pandas as pd
import pytest
from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.exceptions import InvalidDiagnosticException
from climate_ref_core.metric_values import SeriesMetricValue

from climate_ref_scm import provider
from climate_ref_scm.constraints import nearest_member
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


def _solve(
    catalog: pd.DataFrame, source_type: SourceDatasetType, diagnostic: StitchedTimeseries = STITCHED_TAS
):
    executions = solve_executions({source_type: catalog}, diagnostic, provider)
    return {e.datasets[source_type].selector_dict()["experiment_id"]: e for e in executions}


def _datasets(catalog: pd.DataFrame, source_type: SourceDatasetType, diagnostic: StitchedTimeseries):
    return _solve(catalog, source_type, diagnostic)["ssp245"].datasets[source_type].datasets


def _run(
    catalog: pd.DataFrame, source_type: SourceDatasetType, diagnostic: StitchedTimeseries, definition_factory
):
    execution = _solve(catalog, source_type, diagnostic)["ssp245"]
    definition = definition_factory(diagnostic=diagnostic, execution_dataset_collection=execution.datasets)
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    diagnostic.execute(definition)
    result = diagnostic.build_execution_result(definition)

    assert result.successful
    out = pd.read_csv(definition.output_directory / OUTPUT_FILENAME)
    assert [int(c) for c in out.columns if c.isdigit()] == YEARS.tolist()
    series = SeriesMetricValue.load_from_json(definition.to_output_path(result.series_filename))
    return out, series


def _with_ssp_member(catalog: pd.DataFrame, member: str) -> pd.DataFrame:
    """Relabel ssp245 as `member`, which names a parent member that does not exist."""
    is_ssp = catalog["experiment_id"] == "ssp245"
    catalog = catalog.copy()
    catalog.loc[is_ssp, "instance_id"] = catalog.loc[is_ssp, "instance_id"].str.replace("r1i1p1f1", member)
    catalog.loc[is_ssp, ["variant_label", "member_id", "parent_variant_label"]] = member
    return catalog


def _values(out: pd.DataFrame, variable: str) -> pd.DataFrame:
    return out[out["variable"] == variable].set_index("region")[YEARS.astype(str)]


def test_solve_adds_whole_chain(synthetic_catalog, source_type):
    executions = _solve(synthetic_catalog, source_type)

    assert set(executions) == {"historical", "ssp245"}
    ssp = executions["ssp245"].datasets[source_type].datasets
    assert sorted(ssp["experiment_id"].unique()) == ["historical", "piControl", "ssp245"]
    assert sorted(ssp["variable_id"].unique()) == ["areacella", "sftlf", "tas"]


def test_solve_adds_chain_for_each_variable(synthetic_catalog, source_type):
    ssp = _datasets(synthetic_catalog, source_type, STITCHED_TOA_FLUXES)

    chains = ssp.groupby("variable_id")["experiment_id"].unique().map(sorted)
    for variable_id in ("rsdt", "rsut", "rlut"):
        assert chains[variable_id] == ["historical", "piControl", "ssp245"]


def test_solve_drops_group_without_control(synthetic_catalog, source_type):
    catalog = synthetic_catalog[
        ~((synthetic_catalog["experiment_id"] == "piControl") & (synthetic_catalog["variable_id"] == "rsut"))
    ]

    # The solver skips a diagnostic that raises this.
    with pytest.raises(InvalidDiagnosticException):
        _solve(catalog, source_type, STITCHED_TOA_FLUXES)
    assert set(_solve(catalog, source_type)) == {"historical", "ssp245"}


def test_solve_cmip7_parent_has_same_branding(synthetic_catalog):
    other_branding = synthetic_catalog[synthetic_catalog["variable_id"] == "tas"].assign(
        branding_suffix="tavg-u-hxy-u", branded_variable="tas_tavg-u-hxy-u"
    )
    catalog = pd.concat([synthetic_catalog, other_branding], ignore_index=True)

    ssp = _datasets(catalog, SourceDatasetType.CMIP7, STITCHED_TAS)
    assert set(ssp.loc[ssp["variable_id"] == "tas", "branding_suffix"]) == {"tavg-h2m-hxy-u"}


def test_execute_recovers_forcing(synthetic_catalog, source_type, definition_factory):
    out, series = _run(synthetic_catalog, source_type, STITCHED_TAS, definition_factory)

    absolute = _values(out, "Surface Air Temperature")
    anomaly = _values(out, "Surface Air Temperature Change")
    for region, offset in REGION_OFFSET.items():
        expected = control_value(CONTROL_YEARS) + forcing(YEARS) + offset
        np.testing.assert_allclose(absolute.loc[region], expected, atol=1e-9)
        # The constant cell offset cancels against the control run.
        np.testing.assert_allclose(anomaly.loc[region], forcing(YEARS), atol=1e-9)

    assert len(series) == 2 * len(REGION_OFFSET)
    assert {s.dimensions["statistic"] for s in series} == {"absolute", "anomaly"}


def test_execute_derives_net_flux(synthetic_catalog, source_type, definition_factory):
    out, series = _run(synthetic_catalog, source_type, STITCHED_TOA_FLUXES, definition_factory)

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


def test_nearest_member():
    members = ["r1i1p1f1", "r5i1p1f1", "r7i1p1f1", "r6i1p2f1"]

    assert nearest_member("r6i1p1f1", members) == "r5i1p1f1"
    assert nearest_member("r7i1p1f1", members) == "r7i1p1f1"
    assert nearest_member("r6i1p2f1", members) == "r6i1p2f1"


def test_solve_uses_nearest_parent_member(synthetic_catalog, source_type):
    catalog = _with_ssp_member(synthetic_catalog, "r6i1p1f1")

    ssp = _datasets(catalog, source_type, STITCHED_TAS)

    members = ssp.drop_duplicates("experiment_id").set_index("experiment_id")["variant_label"]
    assert members.to_dict() == {"ssp245": "r6i1p1f1", "historical": "r1i1p1f1", "piControl": "r1i1p1f1"}


def test_run_records_substituted_parent(synthetic_catalog, source_type, definition_factory):
    catalog = _with_ssp_member(synthetic_catalog, "r6i1p1f1")

    out, _ = _run(catalog, source_type, STITCHED_TOA_FLUXES, definition_factory)

    assert set(out["caveats"]) == {"historical r6i1p1f1 is not available, so r1i1p1f1 is used"}


def test_run_without_substitute_has_no_caveats(synthetic_catalog, source_type, definition_factory):
    out, _ = _run(synthetic_catalog, source_type, STITCHED_TAS, definition_factory)

    assert out["caveats"].isna().all()
