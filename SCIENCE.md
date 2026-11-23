# GHOST v0.2 scientific definitions

## Period searches and inspection

All methods operate on the cleaned light curve; check clipping and period bounds before interpreting results. Full-sector and half-sector results are correlated measurements, not independent posterior samples.

- **LS** uses Astropy LombScargle with centered flux, a fixed zero mean (`fit_mean=False`), and equal weights.
- **GLS** uses the same sinusoid family with a floating mean (`fit_mean=True`) and flux uncertainties as weights. Missing uncertainties use the loader's fallback estimate. Both use Astropy's standard normalization and a frequency grid with eight samples per baseline resolution element, capped at 100,000 frequencies. A capped grid is flagged in metadata.
- **ACF/BGLS** retain the v0.1 implementations and preprocessing; BGLS has a separately bounded grid and configurable binning.
- **FFT** uses a Hann-windowed, linearly detrended periodogram of the longest contiguous run. Gaps larger than 1.5 median cadences split runs. Runs shorter than 50 points or with cadence deviations above 10% are rejected. Small timing deviations are interpolated onto a uniform grid; long gaps are not bridged. The Nyquist frequency and 80% of the run baseline constrain the search. Fourfold zero padding interpolates the spectrum without improving its physical resolution.

Ranked local maxima are separated by at least 1/T in frequency (T is the analyzed baseline; for FFT it is the retained run). The global maximum can be an edge peak and is flagged. Peaks within 5% of half or twice the strongest peak's period are marked for inspection. These are possible harmonics, not proof of an alias; a sampling-window alias can have another ratio. No peak false-alarm probability is claimed, and powers from different methods are not directly comparable.

Phase is `((time - epoch) / period) % 1`, with the first cleaned time as epoch. The diagnostic displays two cycles and compares P/2, P, and 2P. Display-bin medians are visual aids. Changing the inspection period does not change the period estimates or the Rossby period source. Folded CSVs retain the epoch and fold period.

Reference: [Astropy Lomb–Scargle documentation](https://docs.astropy.org/en/stable/timeseries/lombscargle.html).

## Aggregation

The default is median; mean is selectable. A single-LC candidate reduces its finite positive method/segment estimates with the selected statistic. The legacy field `final_averaged_prot` is retained for compatibility and now means this selected statistic.

For batch star summaries, raw method/segment period estimates are grouped by normalized member ID and sector. The median output is the median within each sector followed by the median across sectors; the mean output analogously uses means. Sectors receive equal weight in the final reduction. Repeated files/methods within a sector contribute estimates within that sector. Missing member IDs or sectors are excluded from star aggregation, but their per-LC results remain available. Star-level GP prior choices use these same summaries.

Averaging or taking the median of competing harmonics can produce a period with no physical peak. Inspect the method/segment results and phase plots before adopting a rotation period. These summary statistics do not supply period uncertainties.

## Rossby number

`Ro = P_rot / tau_c`, with both periods in days. Relations are selectable alternatives; GHOST does not average their Rossby estimates.

| Selection | Required input | Turnover-time prescription | Default input guardrail |
|---|---|---|---|
| Noyes1984 | Johnson B−V | x=1−(B−V); log10(tau)=1.362−0.166x+0.025x²−5.323x³ for x≥0; otherwise 1.362−0.14x | 0.4≤B−V≤1.6 |
| Wright2011 | Johnson V minus 2MASS Ks | X=V−Ks; log10(tau)=0.73+0.22X for X<3.5; otherwise −2.16+1.50X−0.13X² | 1.1<X<6.6 |
| Cranmer2011 | Effective temperature in kelvin | tau=314.24 exp[−Teff/1952.5−(Teff/6250)^18]+0.002 | 3300≤Teff≤7000 K |

Outside-range calculations require explicit extrapolation permission and are marked in exports. The guardrails do not establish applicability to an individual star. These main-sequence/ZAMS prescriptions require particular care for young, pre-main-sequence stars. No automatic extinction correction or uncertainty propagation is performed. Cranmer & Saar is temperature based; GHOST does not invent a color-to-temperature conversion.

Catalog queries first resolve an exact SIMBAD identifier. B and V come from SIMBAD `allfluxes`. For Wright2011, an unambiguous SIMBAD 2MASS identifier is matched exactly to the IRSA 2MASS Point Source Catalog, accepting Ks quality A/B/C only; generic K is never silently substituted for Ks. For Cranmer2011, finite positive literature temperatures in SIMBAD `mesFe_h` are summarized by their median; all retrieved values and bibliographic codes are retained for review. Catalog data may be heterogeneous, unresolved, reddened, or unsuitable; manually supplied inputs are supported. JSON preserves the resolved identity, retrieval time, input values, and catalog provenance when queried.

References:

- [Noyes et al. 1984](https://doi.org/10.1086/161801).
- [Wright et al. 2011, Equation 10](https://doi.org/10.1088/0004-637X/743/1/48); [author-hosted paper](https://hea-www.cfa.harvard.edu/~jdrake/papers/apj_743_1_48.pdf).
- [Cranmer & Saar 2011, Equation 36](https://doi.org/10.1088/0004-637X/741/1/54).
- [SIMBAD query documentation](https://docs.astropy.org/projects/astroquery/en/latest/simbad/simbad.html).
- [IRSA TAP documentation](https://irsa.ipac.caltech.edu/docs/program_interface/TAP.html).
