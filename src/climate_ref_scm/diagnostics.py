"""
Diagnostics for the SCM provider.
"""

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

OUTPUT_FILENAME = "timeseries.csv"
ABSOLUTE = "absolute"
ANOMALY = "anomaly"

# RCMIP variable name and normalisation label for each kind of series.
_KINDS = {
    ABSOLUTE: ("Surface Air Temperature", "none"),
    ANOMALY: ("Surface Air Temperature Change", f"{RUNNING_MEAN_YEARS}-yr-running-mean"),
}


def _requests(
    name: str,
    variable_id: str,
    experiment_id: str,
    experiment_span: tuple[str, str] | None,
    control_span: tuple[str, str],
) -> tuple[CMIP6Request, ...]:
    facets = {"source_id": "ACCESS-ESM1-5", "member_id": "r1i1p1f1"}
    return (
        *(
            CMIP6Request(
                slug=f"{name}-{suffix}",
                facets={
                    **facets,
                    "experiment_id": experiment,
                    "variable_id": variable_id,
                    "table_id": "Amon",
                },
                time_span=span,
            )
            for suffix, experiment, span in (
                (variable_id, experiment_id, experiment_span),
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


def stitched_timeseries(datasets: pd.DataFrame, experiment_id: str, variable_id: str) -> pd.DataFrame:
    """
    Stitched absolute values and control-relative anomalies for each region.

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
        Long table with `kind`, `region`, `year` and `value` columns
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
    anomaly = normalise(stitched, annual[-1], branch_year)

    return pd.concat(
        [
            frame.reset_index().melt(id_vars="year", var_name="region", value_name="value").assign(kind=kind)
            for kind, frame in ((ABSOLUTE, stitched), (ANOMALY, anomaly))
        ],
        ignore_index=True,
    )


def to_rcmip_format(timeseries: pd.DataFrame, selectors: dict[str, str], unit: str) -> pd.DataFrame:
    """
    Wide table in the layout of the RCMIP submission files.
    """
    wide = timeseries.pivot_table(index=["kind", "region"], columns="year", values="value").reset_index()
    meta = pd.DataFrame(
        {
            "climate_model": selectors["source_id"],
            "model": "unspecified",
            "scenario": selectors["experiment_id"],
            "member_id": selectors["variant_label"],
            "region": wide["region"],
            "variable": wide["kind"].map({kind: variable for kind, (variable, _) in _KINDS.items()}),
            "unit": unit,
            "normalisation": wide["kind"].map({kind: method for kind, (_, method) in _KINDS.items()}),
        }
    )
    return pd.concat([meta, wide.drop(columns=["kind", "region"])], axis="columns")


def from_rcmip_format(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Long table with `kind`, `region`, `year` and `value` columns from the RCMIP layout.
    """
    timeseries = wide.melt(
        id_vars=["variable", "region"],
        value_vars=[c for c in wide.columns if c.isdigit()],
        var_name="year",
    ).dropna(subset=["value"])
    timeseries["kind"] = timeseries["variable"].map(
        {variable: kind for kind, (variable, _) in _KINDS.items()}
    )
    return timeseries


def build_series(timeseries: pd.DataFrame, dimensions: dict[str, str]) -> list[SeriesMetricValue]:
    """
    One REF series per region and kind.
    """
    return [
        SeriesMetricValue(
            dimensions={**dimensions, "region": str(region), "statistic": str(kind)},
            values=group["value"].tolist(),
            index=group["year"].astype(int).tolist(),
            index_name="year",
        )
        for (kind, region), group in timeseries.groupby(["kind", "region"], sort=True)
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
    """

    name = "Stitched timeseries"
    slug = "stitched-timeseries"
    variable_id = "tas"
    unit = "K"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": (variable_id,),
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
                RequireFacets("variable_id", (variable_id, *FIXED_FIELDS)),
            ),
        ),
    )

    facets = ("source_id", "experiment_id", "variant_label", "grid_label", "region", "metric", "statistic")

    # ACCESS-ESM1-5 historical branches at piControl year 161,
    # and abrupt-4xCO2 at year 101, the first year of piControl.
    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="ACCESS-ESM1-5 historical tas normalised against piControl",
                requests=_requests("historical", variable_id, "historical", None, ("0151-01", "0335-12")),
            ),
            TestCase(
                name="abrupt-4xCO2",
                description="ACCESS-ESM1-5 abrupt-4xCO2, which branches at the start of piControl",
                requests=_requests(
                    "abrupt-4xco2",
                    variable_id,
                    "abrupt-4xCO2",
                    ("0101-01", "0150-12"),
                    ("0101-01", "0160-12"),
                ),
            ),
        ),
    )

    def execute(self, definition: ExecutionDefinition) -> None:
        """Stitch, normalise and write the regional series."""
        collection = definition.datasets[SourceDatasetType.CMIP6]
        selectors = collection.selector_dict()
        timeseries = stitched_timeseries(collection.datasets, selectors["experiment_id"], self.variable_id)
        to_rcmip_format(timeseries, selectors, self.unit).to_csv(
            definition.output_directory / OUTPUT_FILENAME, index=False
        )

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """Read the written series back into REF series values."""
        selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()
        timeseries = from_rcmip_format(pd.read_csv(definition.output_directory / OUTPUT_FILENAME))
        series = build_series(timeseries, {**selectors, "metric": self.variable_id})

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=_output_bundle(),
            cmec_metric_bundle=CMECMetric.create_template(),
            series=series,
        )
