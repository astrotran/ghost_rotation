# Architecture

## Layers

1. **Catalog layer**
   Resolves uploaded coordinates/names to stable astronomical identifiers.

2. **Observation layer**
   Discovers and retrieves TESS observations and caches local products.

3. **Scientific processing layer**
   Performs quality control, detrending, period inference, Rossby calculations,
   flare analysis, gyrochronology, spot modeling, and anomaly detection.

4. **Persistence/provenance layer**
   Stores stars, observations, runs, parameters, outputs, software versions,
   and quality/confidence metrics.

5. **API layer**
   Exposes scientific capabilities to the UI or external clients.

6. **Frontend**
   Provides catalog upload, star inspection, plots, analysis controls,
   and downloadable results.

## Important rule

Scientific calculations belong in `src/stellar_lc_platform/`.
The API should call those functions, not reimplement them.
