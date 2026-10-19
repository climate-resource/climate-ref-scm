"""
Stitched, normalised timeseries from CMIP for calibrating simple climate models
"""

import importlib.metadata

from climate_ref_core.providers import DiagnosticProvider

from climate_ref_scm.diagnostics import STITCHED_TAS, STITCHED_TOA_FLUXES

__version__ = importlib.metadata.version("climate-ref-scm")

provider = DiagnosticProvider("SCM", __version__)

provider.register(STITCHED_TAS)
provider.register(STITCHED_TOA_FLUXES)
