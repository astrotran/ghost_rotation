# Stellar LC Platform

A modular, web-based analysis platform for large-scale stellar light-curve workflows.

## Long-term goals

- Resolve stars from uploaded catalogs or astronomical databases
- Discover and retrieve TESS light curves
- Run automated light-curve quality control
- Support multiple detrending methods
- Estimate stellar rotation periods using multiple algorithms
- Combine period measurements into a consensus result
- Compute Rossby numbers
- Integrate stellar metadata, SED-derived parameters, and gyrochronology
- Model starspot-driven variability
- Detect statistical anomalies across stellar populations
- Provide interactive scientific visualizations
- Preserve provenance for every analysis run

## Design principle

The scientific Python package should remain independent of the web interface.
The API and frontend orchestrate validated scientific functions rather than
containing the scientific logic themselves.

## Current status

Repository scaffold only. Modules contain placeholders describing their
intended responsibilities.

## Suggested first milestone

1. Upload/read a small CSV containing RA/Dec.
2. Resolve each target to persistent identifiers such as Gaia source ID and TIC ID.
3. Discover available TESS light curves.
4. Download one selected light curve.
5. Produce a QC report.
6. Detrend it.
7. Feed it into the existing rotation-period workflow.
8. Save the result and provenance.
