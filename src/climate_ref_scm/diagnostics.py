"""
Diagnostics for the SCM provider.
"""

import functools
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from climate_ref_core.constraints import (
    AddSupplementaryDataset,
    RequireContiguousTimerange,
    RequireFacets,
)
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.esgf import CMIP6Request
from climate_ref_core.metric_values import SeriesMetricValue
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification

from climate_ref_scm.constraints import AddAncestors, branch_year_in_control, experiment_chain
from climate_ref_scm.regions import annual_regional_means
from climate_ref_scm.stitching import RUNNING_MEAN_YEARS, normalise, stitch

# Experiments used to calibrate MAGICC, plus every parent they stitch onto.
EXPERIMENTS = (
    "historical",
    "hist-GHG",
    "hist-aer",
    "hist-nat",
    "esm-hist",
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
    "esm-ssp585",
    "1pctCO2",
    "abrupt-4xCO2",
    "abrupt-2xCO2",
    "abrupt-0p5xCO2",
)

FIXED_FIELDS = ("areacella", "sftlf")

TOA_FLUXES = ("rsdt", "rsut", "rlut")
NET_FLUX = "rtnt"

# Each derived variable is a signed sum of input variables,
# computed whenever a diagnostic processes all of its inputs.
DERIVED = {NET_FLUX: {"rsdt": 1, "rsut": -1, "rlut": -1}}

# Name used in the RCMIP layout and unit of each variable.
VARIABLES = {
    "tas": ("Surface Air Temperature", "K"),
    "rsdt": ("TOA Incident Shortwave Flux", "W m^-2"),
    "rsut": ("TOA Outgoing Shortwave Flux", "W m^-2"),
    "rlut": ("TOA Outgoing Longwave Flux", "W m^-2"),
    NET_FLUX: ("TOA Net Downward Flux", "W m^-2"),
}

OUTPUT_FILENAME = "timeseries.csv"
ABSOLUTE = "absolute"
ANOMALY = "anomaly"

# Suffix on the RCMIP variable name and normalisation label for each kind of series.
_KINDS = {
    ABSOLUTE: ("", "none"),
    ANOMALY: (" Change", f"{RUNNING_MEAN_YEARS}-yr-running-mean"),
}
_RCMIP_NAMES = {
    (variable_id, kind): f"{name}{suffix}"
    for variable_id, (name, _) in VARIABLES.items()
    for kind, (suffix, _) in _KINDS.items()
}


def _requests(  # noqa: PLR0913
    *,
    name: str,
    source_id: str,
    variable_ids: Sequence[str],
    experiment_id: str,
    experiment_span: tuple[str, str] | None,
    control_span: tuple[str, str],
) -> tuple[CMIP6Request, ...]:
    facets = {"source_id": source_id, "member_id": "r1i1p1f1"}
    return (
        *(
            CMIP6Request(
                slug=f"{name}-{variable_id}-{suffix}",
                facets={
                    **facets,
                    "experiment_id": experiment,
                    "variable_id": variable_id,
                    "table_id": "Amon",
                },
                time_span=span,
            )
            for variable_id in variable_ids
            for suffix, experiment, span in (
                ("experiment", experiment_id, experiment_span),
                ("control", "piControl", control_span),
            )
        ),
        *(
            CMIP6Request(
                slug=f"{name}-{fixed_field}",
                facets={
                    **facets,
                    "experiment_id": experiment_id,
                    "variable_id": fixed_field,
                    "table_id": "fx",
                },
            )
            for fixed_field in FIXED_FIELDS
        ),
    )


def _single_path(datasets: pd.DataFrame, variable_id: str) -> Path:
    paths = datasets.loc[datasets["variable_id"] == variable_id, "path"].unique()
    return Path(paths[0])


def stitched_series(datasets: pd.DataFrame, experiment_id: str, variable_id: str) -> dict[str, pd.DataFrame]:
    """
    Stitched absolute values and control-relative anomalies of one variable for each region.

    Parameters
    ----------
    datasets
        Catalog rows for the experiment, its ancestors and the fixed fields
    experiment_id
        Experiment to produce the series for
    variable_id
        Variable to average

    Returns
    -------
    :
        Absolute values and anomalies keyed by kind, each indexed by year with one column per region
    """
    areacella, sftlf = (_single_path(datasets, fixed_field) for fixed_field in FIXED_FIELDS)
    variable = datasets[datasets["variable_id"] == variable_id]

    chain = experiment_chain(variable, experiment_id)
    annual = [
        annual_regional_means(
            variable_id,
            sorted(Path(p) for p in variable.loc[variable["instance_id"] == row["instance_id"], "path"]),
            areacella,
            sftlf,
        )
        for row in chain
    ]

    stitched = stitch(annual[:-1])
    # The oldest ancestor starts at its branch point, so its first year lines up with the branch year.
    branch_year = branch_year_in_control(chain[-2], chain[-1])
    return {ABSOLUTE: stitched, ANOMALY: normalise(stitched, annual[-1], branch_year)}


def stitched_timeseries(
    datasets: pd.DataFrame, experiment_id: str, variable_ids: Sequence[str]
) -> pd.DataFrame:
    """
    Stitched series for several variables, and any variables derived from them, as one long table.

    Parameters
    ----------
    datasets
        Catalog rows for the experiment, its ancestors and the fixed fields
    experiment_id
        Experiment to produce the series for
    variable_ids
        Variables to average

    Returns
    -------
    :
        Long table with `variable_id`, `kind`, `region`, `year` and `value` columns
    """
    series = {
        variable_id: stitched_series(datasets, experiment_id, variable_id) for variable_id in variable_ids
    }
    for derived_id, terms in DERIVED.items():
        if set(terms) <= set(variable_ids):
            # Every step is linear, so the anomaly of a sum is the same sum of anomalies.
            series[derived_id] = {
                kind: functools.reduce(operator.add, (sign * series[v][kind] for v, sign in terms.items()))
                for kind in _KINDS
            }

    return pd.concat(
        [
            frame.reset_index()
            .melt(id_vars="year", var_name="region", value_name="value")
            .assign(variable_id=variable_id, kind=kind)
            for variable_id, frames in series.items()
            for kind, frame in frames.items()
        ],
        ignore_index=True,
    )


def to_rcmip_format(timeseries: pd.DataFrame, selectors: dict[str, str]) -> pd.DataFrame:
    """
    Wide table in the layout of the RCMIP submission files.
    """
    wide = timeseries.pivot_table(
        index=["variable_id", "kind", "region"], columns="year", values="value", sort=False
    ).reset_index()
    meta = pd.DataFrame(
        {
            "climate_model": selectors["source_id"],
            "model": "unspecified",
            "scenario": selectors["experiment_id"],
            "member_id": selectors["variant_label"],
            "region": wide["region"],
            "variable": [_RCMIP_NAMES[key] for key in zip(wide["variable_id"], wide["kind"])],
            "unit": [VARIABLES[variable_id][1] for variable_id in wide["variable_id"]],
            "normalisation": wide["kind"].map({kind: method for kind, (_, method) in _KINDS.items()}),
        }
    )
    return pd.concat([meta, wide.drop(columns=["variable_id", "kind", "region"])], axis="columns")


def from_rcmip_format(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Long table with `variable_id`, `kind`, `region`, `year` and `value` columns from the RCMIP layout.
    """
    timeseries = wide.melt(
        id_vars=["variable", "region"],
        value_vars=[c for c in wide.columns if c.isdigit()],
        var_name="year",
    ).dropna(subset=["value"])
    keys = {name: key for key, name in _RCMIP_NAMES.items()}
    timeseries["variable_id"], timeseries["kind"] = zip(*timeseries["variable"].map(keys))
    return timeseries


def build_series(timeseries: pd.DataFrame, dimensions: dict[str, str]) -> list[SeriesMetricValue]:
    """
    One REF series per variable, region and kind.
    """
    return [
        SeriesMetricValue(
            dimensions={
                **dimensions,
                "metric": str(variable_id),
                "region": str(region),
                "statistic": str(kind),
            },
            values=group["value"].tolist(),
            index=group["year"].astype(int).tolist(),
            index_name="year",
        )
        for (variable_id, kind, region), group in timeseries.groupby(
            ["variable_id", "kind", "region"], sort=True
        )
    ]


def _output_bundle() -> dict[str, Any]:
    bundle = CMECOutput.create_template()
    bundle["data"]["timeseries"] = {
        "filename": OUTPUT_FILENAME,
        "long_name": "Stitched regional annual means",
        "description": (
            "Annual means over the world and the four MAGICC boxes, stitched onto the parent experiments, "
            f"as absolute values and as anomalies from a {RUNNING_MEAN_YEARS}-year running mean "
            "of the control run."
        ),
    }
    return bundle


class StitchedTimeseries(Diagnostic):
    """
    Stitched regional annual means for calibrating simple climate models.

    Each experiment is joined onto its parents back to the start of `historical`
    and expressed relative to a running mean of the control run at the matching time.
    Results cover the world and the four MAGICC boxes (hemisphere by land or ocean).

    Parameters
    ----------
    name
        Human-readable name
    slug
        Unique identifier
    variable_ids
        Variables to process together in each execution, along with any variables derived from them
    test_data_spec
        Test cases for the diagnostic
    """

    facets = ("source_id", "experiment_id", "variant_label", "grid_label", "region", "metric", "statistic")

    def __init__(
        self,
        name: str,
        slug: str,
        variable_ids: tuple[str, ...],
        test_data_spec: TestDataSpecification | None = None,
    ):
        super().__init__()
        self.name = name
        self.slug = slug
        self.variable_ids = variable_ids
        self.test_data_spec = test_data_spec
        self.data_requirements = (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": variable_ids,
                            "table_id": ("Amon",),
                            "experiment_id": EXPERIMENTS,
                        }
                    ),
                ),
                group_by=("source_id", "experiment_id", "variant_label", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    AddAncestors(),
                    *(
                        AddSupplementaryDataset.from_defaults(fixed_field, SourceDatasetType.CMIP6)
                        for fixed_field in FIXED_FIELDS
                    ),
                    RequireFacets("variable_id", (*variable_ids, *FIXED_FIELDS)),
                ),
            ),
        )

    def execute(self, definition: ExecutionDefinition) -> None:
        """Stitch, normalise and write the regional series."""
        collection = definition.datasets[SourceDatasetType.CMIP6]
        selectors = collection.selector_dict()
        timeseries = stitched_timeseries(collection.datasets, selectors["experiment_id"], self.variable_ids)
        to_rcmip_format(timeseries, selectors).to_csv(
            definition.output_directory / OUTPUT_FILENAME, index=False
        )

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """Read the written series back into REF series values."""
        selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()
        timeseries = from_rcmip_format(pd.read_csv(definition.output_directory / OUTPUT_FILENAME))

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=_output_bundle(),
            cmec_metric_bundle=CMECMetric.create_template(),
            series=build_series(timeseries, selectors),
        )


# ACCESS-ESM1-5 historical branches at piControl year 161,
# and abrupt-4xCO2 at year 101, the first year of piControl.
STITCHED_TAS = StitchedTimeseries(
    name="Stitched timeseries",
    slug="stitched-timeseries",
    variable_ids=("tas",),
    test_data_spec=TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="ACCESS-ESM1-5 historical tas normalised against piControl",
                requests=_requests(
                    name="historical",
                    source_id="ACCESS-ESM1-5",
                    variable_ids=("tas",),
                    experiment_id="historical",
                    experiment_span=None,
                    control_span=("0151-01", "0335-12"),
                ),
            ),
            TestCase(
                name="abrupt-4xCO2",
                description="ACCESS-ESM1-5 abrupt-4xCO2, which branches at the start of piControl",
                requests=_requests(
                    name="abrupt-4xco2",
                    source_id="ACCESS-ESM1-5",
                    variable_ids=("tas",),
                    experiment_id="abrupt-4xCO2",
                    experiment_span=("0101-01", "0150-12"),
                    control_span=("0101-01", "0160-12"),
                ),
            ),
        ),
    ),
)

# MPI-ESM1-2-LR historical branches at piControl year 1850, the first year of piControl.
STITCHED_TOA_FLUXES = StitchedTimeseries(
    name="Stitched TOA fluxes",
    slug="stitched-toa-fluxes",
    variable_ids=TOA_FLUXES,
    test_data_spec=TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="MPI-ESM1-2-LR historical TOA fluxes normalised against piControl",
                requests=_requests(
                    name="toa-historical",
                    source_id="MPI-ESM1-2-LR",
                    variable_ids=TOA_FLUXES,
                    experiment_id="historical",
                    experiment_span=None,
                    control_span=("1850-01", "2029-12"),
                ),
            ),
        ),
    ),
)
