"""
Requests for CMIP7 data as published on ESGF.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Self

import intake_esgf
import pandas as pd
from climate_ref_core.esgf.base import IntakeESGFMixin

# CMIP7 is only indexed by the STAC catalogs.
_STAC_INDICES = ("discovery.east.esgf.io", "discovery.west.esgf.io")


class CMIP7ESGFRequest(IntakeESGFMixin):
    """
    Request for CMIP7 datasets published on ESGF.

    Unlike `climate_ref_core.esgf.CMIP7Request`, which converts CMIP6 data,
    this downloads the published CMIP7 files.
    Facets use the ESGF names, such as `variable_branding_suffix`.

    Parameters
    ----------
    slug
        Unique identifier for this request
    facets
        ESGF search facets
    remove_ensembles
        If True, keep only one ensemble member per model
    time_span
        Optional time range of files to fetch (start, end) in YYYY-MM format
    """

    source_type = "CMIP7"

    def __init__(
        self,
        slug: str,
        facets: dict[str, Any],
        remove_ensembles: bool = False,
        time_span: tuple[str, str] | None = None,
    ):
        self.slug = slug
        self.facets = {"project": "CMIP7", **facets}
        self.remove_ensembles = remove_ensembles
        self.time_span = time_span

    def fetch_datasets(self) -> pd.DataFrame:
        """Fetch dataset metadata from the ESGF STAC catalogs."""
        intake_esgf.conf["stac_indices"].update(dict.fromkeys(_STAC_INDICES, True))
        return super().fetch_datasets()

    def pin_to_datasets(self, datasets: Sequence[Mapping[str, Any]]) -> Self:
        """Return this request unchanged, because the STAC catalogs cannot search by `instance_id`."""
        return self

    def __repr__(self) -> str:
        return f"CMIP7ESGFRequest(slug={self.slug!r}, facets={self.facets!r})"
