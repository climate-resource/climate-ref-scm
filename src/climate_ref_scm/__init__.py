"""
Stitched, normalised timeseries from CMIP for calibrating simple climate models
"""

import importlib.metadata

from climate_ref_core.providers import DiagnosticProvider

from climate_ref_scm.diagnostics import StitchedTimeseries

__version__ = importlib.metadata.version("climate-ref-scm")

provider = DiagnosticProvider("SCM", __version__)

provider.register(StitchedTimeseries())
