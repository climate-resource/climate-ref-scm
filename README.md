# climate-ref-scm

Stitched, normalised timeseries from CMIP for calibrating simple climate models.
It replaces the CMIP6 processing previously done with
[netcdf-scm](https://gitlab.com/netcdf-scm/netcdf-scm) and
[calibration-data](https://gitlab.com/netcdf-scm/calibration-data).

## What it produces

Two diagnostics run once per model, experiment, member and grid:

- `stitched-timeseries` processes `tas`.
- `stitched-toa-fluxes` processes `rsdt`, `rsut` and `rlut`,
  and derives the net downward flux at the top of the atmosphere, `rtnt = rsdt - rsut - rlut`.

For each variable they:

- take annual means of the monthly values over the world and the four MAGICC boxes
  (northern and southern hemisphere, each split into land and ocean, using `areacella` and `sftlf`).
- stitch the experiment onto its parents, so `ssp245` carries `historical` back to 1850.
- subtract a 21-year running mean of the control run at the matching time after the branch,
  which also removes drift in the control run.

Results are written to `timeseries.csv` in the layout of the RCMIP submission files,
and stored in the REF as series with `metric` set to the variable
and `statistic` set to `absolute` or `anomaly`.

Cells centred on the equator count as northern hemisphere, as in netcdf-scm.

The control run must cover every year of the stitched series.
Where a full 21-year window does not fit at either end of the control run,
the running mean is extended along a straight line fitted to its nearest 21 full-window values.
netcdf-scm extended from the two nearest values, which is sensitive to noise,
so the first or last ten years of such runs can differ from its output by a few tenths of a kelvin.

## Installation

```bash
uv pip install "climate-ref-scm @ git+ssh://git@github.com/climate-resource/climate-ref-scm.git"
```

For development:

```bash
git clone git@github.com:climate-resource/climate-ref-scm.git
cd climate-ref-scm
uv sync --all-extras
```

## Usage

This package is a diagnostic provider for the
[Climate REF](https://github.com/Climate-REF/climate-ref).
Once installed, it is automatically discovered via the `climate-ref.providers`
entry point.

```bash
# Verify the provider is registered
ref providers list
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/
```

## License

Apache-2.0
