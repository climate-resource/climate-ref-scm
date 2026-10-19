"""
Find the parents of a dataset in the catalog, up to its control run.
"""

from typing import Self

import cftime
import pandas as pd
from attrs import field, frozen
from climate_ref_core.constraints import AddParentDataset
from climate_ref_core.datasets import SourceDatasetType, select_latest_version
from loguru import logger

CONTROL_EXPERIMENTS = ("piControl", "esm-piControl")

# Guards against a cycle in malformed parent metadata.
_MAX_GENERATIONS = 10

# CMIP6 pins the parent to the same table_id, which CMIP7 splits into these facets.
_CMIP7_SHARED_FACETS = ("branding_suffix", "frequency", "region")


@frozen
class AddAncestors:
    """
    Add every ancestor of each variable in the group, up to and including its control run.

    An `ssp245` group gains its `historical` parent and the `piControl` behind that.
    The group is dropped if any link in the chain is missing or ambiguous.

    Every intermediate experiment must pass the requirement's filters,
    because only those datasets have their parent metadata read during solving.
    """

    parent_facet_map: dict[str, str] = field(
        factory=lambda: AddParentDataset.from_defaults(SourceDatasetType.CMIP6).parent_facet_map
    )

    @classmethod
    def from_defaults(cls, source_type: SourceDatasetType) -> Self:
        """
        Match parents on the facets that identify the same variable for `source_type`.
        """
        parent_facet_map = AddParentDataset.from_defaults(source_type).parent_facet_map
        if source_type == SourceDatasetType.CMIP7:
            parent_facet_map = {**parent_facet_map, **{facet: facet for facet in _CMIP7_SHARED_FACETS}}
        return cls(parent_facet_map)

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Add the ancestors of each variable in `group` from `data_catalog`.
        """
        selected = [group]
        for _, variable in group.groupby("variable_id"):
            ancestors = self._ancestors(variable, data_catalog)
            if ancestors is None:
                return group.loc[[]]
            selected.extend(ancestors)
        return pd.concat(selected)

    def _ancestors(self, child: pd.DataFrame, data_catalog: pd.DataFrame) -> list[pd.DataFrame] | None:
        ancestors: list[pd.DataFrame] = []
        for _ in range(_MAX_GENERATIONS):
            if child["experiment_id"].isin(CONTROL_EXPERIMENTS).all():
                return ancestors
            parent = self._find_parent(child, data_catalog)
            if parent is None:
                return None
            ancestors.append(parent)
            child = parent

        logger.debug(f"Constraint {self} not satisfied because the parent chain did not reach a control run")
        return None

    def _find_parent(self, child: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame | None:
        child_facets = child[list(self.parent_facet_map.values())].drop_duplicates()
        if len(child_facets) != 1 or child_facets.isna().any(axis=None):
            logger.debug(f"Constraint {self} not satisfied because the parent of {child_facets} is unclear")
            return None

        facets = child_facets.iloc[0]
        mask = pd.Series(True, index=data_catalog.index)
        for parent_facet, child_facet in self.parent_facet_map.items():
            mask &= data_catalog[parent_facet] == facets[child_facet]
        parent = data_catalog[mask]
        if parent.empty:
            logger.debug(
                f"Constraint {self} not satisfied because no parent was found for {facets.to_dict()}"
            )
            return None
        return select_latest_version(parent)


def experiment_chain(datasets: pd.DataFrame, experiment_id: str) -> list[pd.Series]:
    """
    Order the datasets from the target experiment up to its control run.

    Parameters
    ----------
    datasets
        Catalog rows for the target experiment and its ancestors, one or more rows per dataset
    experiment_id
        Experiment to start from

    Returns
    -------
    :
        One row per experiment, starting with the target and ending with the control run
    """
    by_experiment = {
        row["experiment_id"]: row for _, row in datasets.drop_duplicates("experiment_id").iterrows()
    }
    chain = [by_experiment[experiment_id]]
    while chain[-1]["experiment_id"] not in CONTROL_EXPERIMENTS:
        parent_id = chain[-1]["parent_experiment_id"]
        if parent_id not in by_experiment or len(chain) > len(by_experiment):
            raise ValueError(f"Parent {parent_id!r} of {chain[-1]['experiment_id']!r} is not in the datasets")
        chain.append(by_experiment[parent_id])
    return chain


def branch_year_in_control(child: pd.Series, control: pd.Series) -> int:
    """
    Year in the control run at which the child branched.

    Parameters
    ----------
    child
        Catalog row for the experiment whose parent is the control run
    control
        Catalog row for the control run

    Returns
    -------
    :
        Branch year in the control run's calendar
    """
    units = child.get("parent_time_units")
    if not isinstance(units, str) or not units:
        units = control["time_units"]
    branch = cftime.num2date(child["branch_time_in_parent"], units=units, calendar=control["calendar"])
    return int(branch.year)
