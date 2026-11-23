# GHOST (**G**aussian-process, **H**armonic-Oscillation, and **S**elf-correlation **T**oolkit)

GHOST is a desktop Python tool for inspecting stellar light curves and rotation periods. This first functional update is **GHOST_V0_2.py**, based on the existing Tkinter application.

- ACF, BGLS, classical Lomb–Scargle (LS), generalized Lomb–Scargle (GLS), and FFT searches, for full or half sectors.
- Ranked peaks and phase plots at P/2, P, and 2P, with a manually editable inspection period.
- Median aggregation by default, with mean available. Batch star periods combine estimates within each sector, then combine sector values with equal sector weight.
- Rossby numbers from Noyes1984 (B−V), Wright2011 (V−Ks), or Cranmer & Saar2011 (effective temperature), using manual or queried inputs.
- CSV, JSON, and figure exports; optional GP fitting retained from v0.1.

## Run

Use Python 3.11 or 3.12. From this repository's root:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python GHOST_V0_2.py
```

Tkinter must be included with your Python installation (on Ubuntu, install `python3-tk`). No server is required. The earlier `stellar-lc-platform/` web scaffold is retained for future work; it is separate from this desktop release.

Load a FITS, CSV, or text light curve, verify its member ID and sector, choose methods and bounds, and click **Calculate Selected Periods**. Open **Peaks & phase** to select a method/segment and inspect competing peaks. A fold-period edit changes the diagnostic only; it does not overwrite measured periods. Rossby controls let you select the calibration, period source, and manual inputs or query the required catalog inputs.

CSV example (time in days; optional positive flux uncertainty):

```csv
time,flux,flux_err
0.0,1.001,0.001
0.02,1.003,0.001
```

This illustrates the format; supply a real time series with enough samples and baseline for the requested period range. TESS-style FITS tables are supported through Astropy. Some additional mission formats need optional `lightkurve`.

## Command line

```sh
python GHOST_V0_2.py --file lightcurve.csv --period-method LS,GLS,FFT --segment all --pmin 0.2 --pmax 10 --aggregation median --output results/star
python GHOST_V0_2.py --file lightcurve.csv --period-method GLS --fold-period 4.2 --calculate-rossby --rossby-relation Wright2011 --v-ks 2.0 --output results/star
```

To query instead of entering a color, add `--calculate-rossby --query-photometry --member-id "HD 189733"` and select `--rossby-relation`. Cranmer2011 uses `--teff` in kelvin. Catalog failures are reported; manual inputs remain available.

Outputs include an overview PNG, measurements CSV, JSON with settings and provenance, ranked-peaks CSV, phase-comparison PNG, and folded-light-curve CSV. GP fitting is opt-in with `--run-gp`; optional backends/sampling require `celerite2`/`emcee`. See `--help` for all settings.

## Scientific notes and validation

See [SCIENCE.md](SCIENCE.md) for definitions, references, aggregation details, and limitations. A harmonic flag is a prompt for inspection, not a confirmed alias or a significance test. Rossby values depend on calibration and are not automatically corrected for extinction or young-star evolution.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Tests cover synthetic period recovery, harmonic diagnostics, FFT sampling restrictions, phase folding, reference turnover times, catalog-response fixtures, aggregation, and CLI exports. Plotting and CLI execution were checked headlessly. Interactive desktop behavior and live SIMBAD/IRSA access still need a local smoke test; network catalog checks did not succeed in the development environment. Optional GP backends were not revalidated in this update.
