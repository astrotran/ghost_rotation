#!/usr/bin/env python3
"""
GHOST v0.2: desktop stellar rotation analysis with ACF, BGLS, LS, GLS and FFT.

Full/half-sector measurements include ranked diagnostic peaks. Phase folding
and P/2, P, 2P comparisons are available without a GP. Optional exact/celerite2
GP fitting is retained. Rossby inputs are queried or supplied manually for
Noyes1984 (B-V), Wright2011 (V-Ks), and Cranmer & Saar 2011 (Teff).

Median aggregation is the default; mean remains selectable. Batch star periods
reduce raw method/segment estimates within (member, sector), then reduce those
sector values with equal sector weight. The compatibility column name
final_averaged_prot now denotes the selected within-LC statistic. These summaries
are descriptive, not posterior consensus or harmonic resolution.

Run `python GHOST_V0_2.py` for Tkinter, or `--help` for CLI. GP is opt-in via
--run-gp on the CLI. Install requirements.txt; see README.md and SCIENCE.md.
"""

from __future__ import annotations

TOOL_VERSION = "GHOST-V0.2-2026-11-23"

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import re
import sys
import threading
import traceback
import ssl
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


def _install_numpy_compatibility_aliases() -> None:
    """Restore NumPy APIs still imported by older astronomy dependencies.

    NumPy 2.4 removed ``np.in1d`` and the private helper
    ``_check_interpolation_as_method``.  Older Astropy/Astroquery releases may
    still import those names.  These narrow compatibility shims preserve the
    previous behavior long enough for the catalog query to run; they do not
    change any rotation or Rossby-number calculation.
    """
    if not hasattr(np, "in1d"):
        def _in1d_compat(
            ar1, ar2, assume_unique: bool = False, invert: bool = False, *, kind=None
        ):
            a1 = np.asarray(ar1).ravel()
            a2 = np.asarray(ar2).ravel()
            kwargs = {
                "assume_unique": assume_unique,
                "invert": invert,
            }
            if kind is not None:
                kwargs["kind"] = kind
            try:
                return np.isin(a1, a2, **kwargs)
            except TypeError:
                kwargs.pop("kind", None)
                return np.isin(a1, a2, **kwargs)

        setattr(np, "in1d", _in1d_compat)

    # Astropy versions released before NumPy 2.4 may import this private
    # compatibility helper unconditionally during ``astroquery.simbad`` import.
    try:
        import numpy.lib._function_base_impl as _np_function_base
        if not hasattr(_np_function_base, "_check_interpolation_as_method"):
            def _check_interpolation_as_method(method, interpolation, fname):
                if interpolation is None:
                    return method
                if method != "linear":
                    raise TypeError(
                        "Do not pass both 'method' and the deprecated "
                        f"'interpolation' argument to {fname}."
                    )
                return interpolation

            setattr(
                _np_function_base,
                "_check_interpolation_as_method",
                _check_interpolation_as_method,
            )
    except Exception:
        # The direct TAP query below does not depend on Astropy, so failure to
        # install this optional import shim is non-fatal.
        pass


_install_numpy_compatibility_aliases()

import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit, minimize
from scipy.signal import butter, find_peaks, savgol_filter, sosfilt, sosfiltfilt
from scipy.special import logsumexp

import matplotlib
from matplotlib.figure import Figure

try:
    from numba import njit  # type: ignore
except ImportError:
    def njit(*args, **kwargs):
        """Small no-op fallback so BGLS still works without numba."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda func: func


# -----------------------------------------------------------------------------
# GP engine choices
# -----------------------------------------------------------------------------

GP_BACKENDS = (
    "GProtation (Angus+17)",
    "exoplanet / celerite2",
)

CELERITE_KERNELS = (
    "RotationTerm",
    "RotationTerm + Matern32",
    "Double RotationTerm (experimental)",
)

CELERITE_KERNEL_HELP = {
    "RotationTerm": (
        "Fast celerite2 stellar-rotation kernel. Two SHO modes represent P and P/2; "
        "finite Q lets coherence/amplitude evolve with time."
    ),
    "RotationTerm + Matern32": (
        "RotationTerm plus a smooth aperiodic component for baseline drift / slow "
        "magnetic variability. Usually the best alternative when the LC baseline moves."
    ),
    "Double RotationTerm (experimental)": (
        "Two rotation components sharing shape parameters. Useful for close competing "
        "periods / beat patterns, but more degenerate; the reported GP period is the "
        "prior-anchored primary period and the secondary period is saved in metadata."
    ),
}

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class LightCurveData:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    name: str = "lightcurve"
    source_path: str = ""
    normalization_note: str = ""


@dataclass
class FitSettings:
    pmin: float = 0.1
    pmax: float = 13.0
    clip_upper: float = 5.0
    clip_lower: float = 8.0
    acf_filter_order: int = 5
    acf_prior_width: float = 0.2
    acf_max_points: int = 30000
    gp_points: int = 500
    chunk_size: int = 250
    optimizer_starts: int = 10
    use_paper_hyperpriors: bool = False
    run_mcmc: bool = False
    mcmc_walkers: int = 40
    mcmc_steps: int = 600
    mcmc_burn: int = 250
    prediction_train_points: int = 500
    prediction_points: int = 1200
    posterior_scan_points: int = 120
    bgls_bin_minutes: float = 30.0
    bgls_max_steps: int = 3000
    rotation_smooth_hours: float = 6.0
    gp_backend: str = "GProtation (Angus+17)"
    celerite_kernel: str = "RotationTerm"
    celerite_max_points: int = 5000  # 0 = use every clean point
    random_seed: int = 42
    aggregation: str = "median"
    peak_count: int = 5
    ls_samples_per_peak: int = 8
    ls_max_steps: int = 100000

    def validate(self, baseline: Optional[float] = None) -> None:
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("Aggregation must be mean or median.")
        if self.peak_count < 1 or self.ls_samples_per_peak < 2 or self.ls_max_steps < 100:
            raise ValueError("Invalid peak/frequency-grid settings.")
        if not np.isfinite(self.pmin) or self.pmin <= 0:
            raise ValueError("Pmin must be positive.")
        if not np.isfinite(self.pmax) or self.pmax <= self.pmin:
            raise ValueError("Pmax must be larger than Pmin.")
        if baseline is not None and self.pmax > baseline:
            raise ValueError(
                f"Pmax={self.pmax:g} d exceeds the {baseline:g} d baseline. "
                "Use a smaller search range."
            )
        if self.gp_points < 30:
            raise ValueError("Use at least 30 GP points.")
        if self.chunk_size < 20:
            raise ValueError("Use a chunk size of at least 20.")
        if self.optimizer_starts < 1:
            raise ValueError("Use at least one optimizer start.")
        if self.bgls_bin_minutes <= 0:
            raise ValueError("BGLS bin size must be positive.")
        if self.bgls_max_steps < 20:
            raise ValueError("BGLS needs at least 20 frequency steps.")
        if self.rotation_smooth_hours <= 0:
            raise ValueError("Rotation smoothing window must be positive.")
        if self.gp_backend not in GP_BACKENDS:
            raise ValueError(f"Unknown GP backend: {self.gp_backend}")
        if self.celerite_kernel not in CELERITE_KERNELS:
            raise ValueError(f"Unknown celerite2 kernel: {self.celerite_kernel}")
        if self.celerite_max_points < 0:
            raise ValueError("celerite max points must be >= 0 (0 means all clean points).")
        if self.run_mcmc:
            if self.mcmc_walkers < 12:
                raise ValueError("Use at least 12 MCMC walkers.")
            if self.mcmc_steps <= self.mcmc_burn:
                raise ValueError("MCMC steps must exceed burn-in steps.")


@dataclass
class ACFCandidate:
    pmax_filter: float
    period: float
    peak_height: float
    tau: float
    quality: float
    residual_ss: float
    n_lags: int
    lags: np.ndarray = field(repr=False)
    acf: np.ndarray = field(repr=False)
    damped_model: np.ndarray = field(repr=False)


@dataclass
class PeriodMeasurement:
    segment: str
    method: str
    period: float
    score: float = np.nan
    note: str = ""
    periods: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    power: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    acf_candidate: Optional[ACFCandidate] = field(default=None, repr=False)
    peaks: list[dict] = field(default_factory=list)
    diagnostic_metadata: dict = field(default_factory=dict)


@dataclass
class PeriodAnalysis:
    measurements: dict[tuple[str, str], PeriodMeasurement]
    acf_combined: float
    bgls_combined: float
    final_averaged_prot: float
    acf_n_used: int
    bgls_n_used: int
    requested_segments: list[str]
    requested_methods: list[str]
    aggregation: str = "median"
    method_combined: dict[str, float] = field(default_factory=dict)

    def measurement(self, segment: str, method: str) -> Optional[PeriodMeasurement]:
        return self.measurements.get((segment, method.upper()))

    def prior_center(self, source: str) -> float:
        key = source.strip().lower()
        values = {
            "acf_combined": self.acf_combined,
            "bgls_combined": self.bgls_combined,
            "final_averaged_prot": self.final_averaged_prot,
            # For a single LC, across-LC mean/median collapse to the same
            # per-LC final estimate. The GUI overrides these with true star-level
            # references after a multi-LC batch has been measured.
            "final_mean_period": self.final_averaged_prot,
            "final_median_period": self.final_averaged_prot,
        }
        if key not in values:
            raise ValueError(
                "GP prior source must be ACF_combined, BGLS_combined, "
                "or final_averaged_prot."
            )
        value = float(values[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{source} is unavailable because no required period estimates "
                "were finite. Choose another prior source."
            )
        return value


@dataclass
class LCMetadata:
    member_id: str
    sector: str
    lc_source: str


@dataclass
class RossbyResult:
    """Rossby-number result for the currently loaded light curve."""

    b_v: float
    tau_c_days: float
    rossby_number: float
    period_source: str
    period_days: float
    queried_name: str = ""
    resolved_name: str = ""
    b_mag: float = np.nan
    v_mag: float = np.nan
    note: str = ""
    relation: str = "Noyes1984"
    input_name: str = "B-V"
    input_value: float = np.nan
    v_ks: float = np.nan
    ks_mag: float = np.nan
    teff: float = np.nan
    in_calibration: bool = True
    input_provenance: dict = field(default_factory=dict)


@dataclass
class FitResult:
    map_period: float
    map_theta: np.ndarray
    map_log_posterior: float
    period_median: float
    period_p16: float
    period_p84: float
    acf_candidates: list[ACFCandidate]
    period_mixture: np.ndarray
    posterior_period_samples: Optional[np.ndarray]
    scan_periods: np.ndarray
    scan_log_probability: np.ndarray
    gp_time: np.ndarray
    gp_mean: np.ndarray
    gp_std: np.ndarray
    phase: np.ndarray
    phase_flux: np.ndarray
    phase_bin_x: np.ndarray
    phase_bin_y: np.ndarray
    fit_time: np.ndarray
    fit_flux: np.ndarray
    fit_flux_err: np.ndarray
    n_input: int
    n_clean: int
    n_gp: int
    n_chunks: int
    period_analysis: PeriodAnalysis
    gp_prior_source: str
    gp_prior_center: float
    gp_backend: str = "GProtation (Angus+17)"
    gp_kernel: str = "Angus17 quasi-periodic"
    map_hyperparameters: dict[str, float] = field(default_factory=dict)
    acceptance_fraction: Optional[float] = None

    @property
    def period_plus(self) -> float:
        return self.period_p84 - self.period_median

    @property
    def period_minus(self) -> float:
        return self.period_median - self.period_p16


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.std(values)
    return float(sig)


def normal_logpdf(x: np.ndarray | float, mu, sigma) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    return -0.5 * ((x_arr - mu_arr) / sigma_arr) ** 2 - np.log(
        sigma_arr * math.sqrt(2.0 * math.pi)
    )


def weighted_quantile(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=float), quantiles)


def safe_float_array(values) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def choose_column(columns: Iterable[str], choices: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for choice in choices:
        if choice.lower() in lookup:
            return lookup[choice.lower()]
    return None


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "lightcurve"



SEGMENT_ORDER = ("full_sector", "sectorA", "sectorB")
METHOD_ORDER = ("ACF", "BGLS", "LS", "GLS", "FFT")


ALLOWED_LC_SOURCES = ("SPOC", "QLP", "T16")


def infer_lc_source_from_name(name: str) -> str:
    """Infer the research LC source using only the three canonical labels.

    We deliberately do not use arbitrary filename tails as the source. That
    was the reason cadence/flux tokens such as ``[1800.]_det_flux`` appeared in
    earlier tables. Unknown products remain ``unknown`` rather than being
    mislabeled as SPOC/QLP/T16.
    """
    upper = str(name).upper()
    if re.search(r"(?:^|[_\-])SPOC(?:[_\-]|$)", upper):
        return "SPOC"
    if re.search(r"(?:^|[_\-])QLP(?:[_\-]|$)", upper):
        return "QLP"
    if re.search(r"(?:^|[_\-])T16(?:[_\-]|$)", upper):
        return "T16"
    return "unknown"


def strip_lc_product_suffix(text: str) -> str:
    """Strip sector/product labels from an inferred stellar identifier."""
    value = str(text).strip().strip("_")
    value = re.sub(r"(?:^|_)S\d+(?:_|$).*$", "", value, flags=re.I).strip("_")
    value = re.split(
        r"_(?:SPOC|QLP|T16|TGLC|GSFC[-_]?ELEANOR(?:[-_]?LITE)?|ELEANOR(?:[-_]?LITE)?)(?:_|$)",
        value,
        maxsplit=1,
        flags=re.I,
    )[0].strip("_")
    value = re.sub(r"_(?:LC|LIGHTCURVE)$", "", value, flags=re.I).strip("_")
    return value or str(text).strip() or "unknown"


def infer_lc_metadata(path: str) -> LCMetadata:
    """Infer canonical member, sector, and LC source from an NYMG filename."""
    stem = Path(path).stem
    if stem.lower().endswith(".fits"):
        stem = Path(stem).stem
    source = infer_lc_source_from_name(stem)

    if "__" in stem:
        parts = stem.split("__")
        member = strip_lc_product_suffix(parts[0] or stem)
        sector = ""
        for part in parts:
            match = re.fullmatch(r"S(\d+)", part, flags=re.I)
            if match:
                sector = match.group(1)
                break
        if source == "unknown":
            for part in parts:
                candidate = infer_lc_source_from_name(part)
                if candidate != "unknown":
                    source = candidate
                    break
        return LCMetadata(member, sector, source)

    cleaned = re.sub(
        r"(?:_detrended_LC_Table|_detrended_LC|_LC_Table|_LC)$",
        "", stem, flags=re.I,
    )
    match = re.search(r"(?:^|_)S(\d+)(?:_|$)", cleaned, flags=re.I)
    sector = match.group(1) if match else ""
    prefix = cleaned[: match.start()].strip("_") if match else cleaned
    member = strip_lc_product_suffix(prefix)
    return LCMetadata(member, sector, source)


MEMBERSHIP_DESIGNATION_CANDIDATES = (
    "designation", "member_id", "name", "target_name", "source_name"
)


def load_membership_table(path: str) -> pd.DataFrame:
    """Load the optional membership reference table used for identity matching."""
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading the membership Parquet requires pyarrow. Install it with "
                "'python -m pip install pyarrow', or provide a CSV copy."
            ) from exc
    elif suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        frame = pd.read_csv(path, sep=sep)
    else:
        raise ValueError("Membership list must be Parquet, CSV, TSV, or TXT.")
    if frame.empty:
        raise ValueError("Membership list is empty.")
    return frame


def membership_designation_column(frame: pd.DataFrame) -> str:
    lookup = {str(c).strip().lower(): str(c) for c in frame.columns}
    for candidate in MEMBERSHIP_DESIGNATION_CANDIDATES:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        "Membership list needs a designation/name column. Tried: "
        + ", ".join(MEMBERSHIP_DESIGNATION_CANDIDATES)
    )


def match_membership_row(
    frame: Optional[pd.DataFrame], member_id: str
) -> tuple[str, Optional[pd.Series]]:
    """Return canonical designation and matching membership row when possible."""
    if frame is None or frame.empty:
        return strip_lc_product_suffix(member_id), None
    designation_col = membership_designation_column(frame)
    target = normalize_member_key(member_id)
    if not target or target == "unknown":
        return strip_lc_product_suffix(member_id), None
    normalized = frame[designation_col].astype(str).map(normalize_member_key)
    matches = frame.loc[normalized == target]
    if matches.empty:
        return strip_lc_product_suffix(member_id), None
    row = matches.iloc[0]
    canonical = str(row.get(designation_col, member_id)).strip() or str(member_id)
    return canonical, row


def membership_query_names(
    member_id: str, membership_row: Optional[pd.Series]
) -> list[str]:
    """Build SIMBAD identifiers from the canonical membership row.

    The supplied NYMG table's ``bmag``/``rmag`` are Gaia BP/RP, not Johnson
    B/V, so they are not used directly for Noyes B-V. The membership list is
    instead used to obtain a reliable canonical designation or Gaia DR3 ID.
    """
    names = [str(member_id).strip()]
    if membership_row is not None:
        for key in MEMBERSHIP_DESIGNATION_CANDIDATES:
            if key in membership_row.index:
                value = str(membership_row.get(key, "")).strip()
                if value and value.lower() != "nan":
                    names.append(value)
        for key in ("gdr3_id", "source_id", "gaia_dr3", "gaia_source_id"):
            if key in membership_row.index:
                value = str(membership_row.get(key, "")).strip()
                value = re.sub(r"\.0$", "", value)
                if value and value.lower() not in {"nan", "none", ""}:
                    names.append(f"Gaia DR3 {value}")
    return list(dict.fromkeys(name for name in names if name))


ROSSBY_PERIOD_SOURCES = (
    "final_mean_period", "final_median_period",
    "final_averaged_prot", "ACF_combined", "BGLS_combined",
    "GP_MAP", "GP_median",
    "ACF_full_sector", "ACF_sectorA", "ACF_sectorB",
    "BGLS_full_sector", "BGLS_sectorA", "BGLS_sectorB",
    *tuple(f"{method}_{segment}" for method in ("LS", "GLS", "FFT") for segment in SEGMENT_ORDER),
)


def noyes84_convective_turnover_time(b_v: float) -> tuple[float, str]:
    """Return Noyes et al. (1984) convective turnover time in days."""
    b_v = float(b_v)
    if not np.isfinite(b_v):
        raise ValueError("B-V must be finite.")
    x = 1.0 - b_v
    if x >= 0.0:
        log_tau = 1.362 - 0.166*x + 0.025*x**2 - 5.323*x**3
    else:
        log_tau = 1.362 - 0.14*x
    tau_c = float(10.0**log_tau)
    note = "Noyes84"
    if not (0.4 <= b_v <= 1.6):
        note += "; B-V outside usual calibration range (extrapolated)"
    return tau_c, note


def _query_name_candidates(member_id: str) -> list[str]:
    raw = str(member_id).strip()
    cleaned = re.sub(r"^\*+\s*", "", raw).replace("_", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    candidates = [raw, cleaned]
    if cleaned and not cleaned.lower().startswith("v*"):
        candidates.append(f"V* {cleaned}")
    return list(dict.fromkeys(name for name in candidates if name))


def _table_scalar(table, candidate_columns: Sequence[str]) -> float:
    if table is None:
        return np.nan
    names = {str(name).lower(): name for name in getattr(table, "colnames", [])}
    for candidate in candidate_columns:
        actual = names.get(candidate.lower())
        if actual is None:
            continue
        for value in table[actual]:
            if np.ma.is_masked(value):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return np.nan


def _table_text(table, candidate_columns: Sequence[str]) -> str:
    if table is None or len(table) == 0:
        return ""
    names = {str(name).lower(): name for name in getattr(table, "colnames", [])}
    for candidate in candidate_columns:
        actual = names.get(candidate.lower())
        if actual is None:
            continue
        value = table[actual][0]
        if np.ma.is_masked(value):
            continue
        if isinstance(value, bytes):
            return value.decode(errors="replace").strip()
        return str(value).strip()
    return ""


def _query_bv_with_astroquery(query_name: str) -> tuple[float, float, float, str]:
    """Query one identifier with Astroquery and return B-V, B, V, resolved name."""
    _install_numpy_compatibility_aliases()
    from astroquery.simbad import Simbad  # type: ignore

    errors = []

    # Current Astroquery/SIMBAD API: optical-filter names are case-sensitive.
    try:
        simbad = Simbad()
        simbad.add_votable_fields("B", "V")
        table = simbad.query_object(query_name)
        if table is not None and len(table):
            b_mag = _table_scalar(table, ("B", "FLUX_B", "flux_b"))
            v_mag = _table_scalar(table, ("V", "FLUX_V", "flux_v"))
            if np.isfinite(b_mag) and np.isfinite(v_mag):
                resolved = _table_text(table, ("main_id", "MAIN_ID")) or query_name
                return float(b_mag - v_mag), b_mag, v_mag, resolved
    except Exception as exc:
        errors.append(f"B/V fields: {type(exc).__name__}: {exc}")

    # Compatibility with older Astroquery releases.
    for fields in (("flux(B)", "flux(V)"), ("allfluxes",)):
        try:
            simbad = Simbad()
            simbad.add_votable_fields(*fields)
            table = simbad.query_object(query_name)
            if table is None or len(table) == 0:
                continue
            b_mag = _table_scalar(table, ("B", "FLUX_B", "flux_b"))
            v_mag = _table_scalar(table, ("V", "FLUX_V", "flux_v"))
            if np.isfinite(b_mag) and np.isfinite(v_mag):
                resolved = _table_text(table, ("main_id", "MAIN_ID")) or query_name
                return float(b_mag - v_mag), b_mag, v_mag, resolved
        except Exception as exc:
            errors.append(f"{fields}: {type(exc).__name__}: {exc}")

    raise ValueError("; ".join(errors) if errors else "no Johnson B and V values returned")


SIMBAD_TAP_ENDPOINTS = (
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "https://simbad.u-strasbg.fr/simbad/sim-tap/sync",
)


def _secure_tap_post(
    endpoint: str, payload: dict[str, str], timeout: float
) -> tuple[str, str]:
    """POST to a TAP service while retaining certificate verification.

    Windows.org Python installations sometimes lack a complete CA chain for
    ``urllib``.  We therefore try Requests with the native Windows trust store
    (when ``truststore`` is installed), then Requests/urllib with Certifi's CA
    bundle.  Verification is never disabled.
    """
    errors: list[str] = []

    # Preferred route: Requests plus the native OS certificate store.
    try:
        verify_options: list[object] = []
        try:
            import truststore  # type: ignore
            # This is an end-user application, and the injection happens only
            # when the user clicks the query button.  Import Requests afterward
            # so urllib3 sees the native Windows certificate store.
            truststore.inject_into_ssl()
            verify_options.append(True)
        except Exception as exc:
            errors.append(f"truststore unavailable: {type(exc).__name__}: {exc}")

        import requests  # type: ignore

        try:
            import certifi  # type: ignore
            verify_options.append(certifi.where())
        except Exception as exc:
            errors.append(f"certifi unavailable: {type(exc).__name__}: {exc}")

        verify_options.append(True)
        seen: set[str] = set()
        for verify in verify_options:
            key = str(verify)
            if key in seen:
                continue
            seen.add(key)
            try:
                response = requests.post(
                    endpoint,
                    data=payload,
                    headers={"User-Agent": "ACF-BGLS-GP-Rotation-GUI/2026-08-06"},
                    timeout=timeout,
                    verify=verify,
                )
                response.raise_for_status()
                return response.text, f"requests verify={verify}"
            except Exception as exc:
                errors.append(
                    f"requests verify={verify}: {type(exc).__name__}: {exc}"
                )
    except Exception as exc:
        errors.append(f"requests import/use failed: {type(exc).__name__}: {exc}")

    # Standard-library fallback, first with Certifi and then the system store.
    encoded = urlencode(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=encoded,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "ACF-BGLS-GP-Rotation-GUI/2026-08-06",
        },
        method="POST",
    )
    contexts: list[tuple[str, ssl.SSLContext]] = []
    try:
        import certifi  # type: ignore
        contexts.append(("urllib+certifi", ssl.create_default_context(cafile=certifi.where())))
    except Exception as exc:
        errors.append(f"urllib certifi context failed: {type(exc).__name__}: {exc}")

    try:
        context = ssl.create_default_context()
        # Python 3.13 enabled VERIFY_X509_STRICT by default.  Disabling only
        # this strictness flag can help with imperfect intermediate chains while
        # preserving CA and hostname verification.
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            context.verify_flags &= ~ssl.VERIFY_X509_STRICT
        contexts.append(("urllib+system-store", context))
    except Exception as exc:
        errors.append(f"urllib system context failed: {type(exc).__name__}: {exc}")

    for label, context in contexts:
        try:
            with urlopen(request, timeout=timeout, context=context) as response:
                return response.read().decode("utf-8", errors="replace"), label
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise RuntimeError("; ".join(errors[-8:]))


def _query_bv_with_direct_tap(query_name: str, timeout: float = 45.0) -> tuple[float, float, float, str]:
    """Query SIMBAD TAP without depending on Astroquery."""
    escaped_name = str(query_name).replace("'", "''")
    adql = f"""
        SELECT TOP 1 basic.main_id, allfluxes.B, allfluxes.V
        FROM basic
        JOIN ident ON basic.oid = ident.oidref
        LEFT JOIN allfluxes ON basic.oid = allfluxes.oidref
        WHERE ident.id = '{escaped_name}'
    """
    payload = {
        "request": "doQuery",
        "lang": "adql",
        "format": "csv",
        "query": adql,
    }

    endpoint_errors: list[str] = []
    for endpoint in SIMBAD_TAP_ENDPOINTS:
        try:
            body, transport = _secure_tap_post(endpoint, payload, timeout)
            reader = csv.DictReader(io.StringIO(body))
            row = next(reader, None)
            if row is None:
                raise ValueError("SIMBAD TAP returned no matching object")

            lower = {str(key).strip().lower(): value for key, value in row.items()}
            try:
                b_mag = float(lower.get("b"))
            except (TypeError, ValueError):
                b_mag = np.nan
            try:
                v_mag = float(lower.get("v"))
            except (TypeError, ValueError):
                v_mag = np.nan
            resolved = str(lower.get("main_id") or query_name).strip()
            if not (np.isfinite(b_mag) and np.isfinite(v_mag)):
                raise ValueError(
                    "SIMBAD resolved the object but did not return both Johnson B and V"
                )
            return float(b_mag - v_mag), float(b_mag), float(v_mag), resolved
        except Exception as exc:
            endpoint_errors.append(
                f"{endpoint}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(" | ".join(endpoint_errors))


def _astroquery_install_diagnostic(exc: BaseException) -> str:
    """Describe whether Astroquery is missing or merely failing to import."""
    try:
        spec = importlib.util.find_spec("astroquery")
    except Exception:
        spec = None
    if spec is None:
        location = "not found in this interpreter"
    else:
        location = str(spec.origin or spec.submodule_search_locations or "location unknown")
    return (
        f"Python interpreter: {sys.executable}; astroquery: {location}; "
        f"import error: {type(exc).__name__}: {exc}"
    )


def query_bv_from_simbad(
    member_id: str, progress=None, extra_candidates: Optional[Sequence[str]] = None
) -> tuple[float, float, float, str, str]:
    """Return (B-V, B, V, queried_name, resolved_name) from SIMBAD.

    The dependency-light direct TAP query is attempted first.  Astroquery is
    retained as a secondary fallback, since some user environments have older
    Astropy releases that are temporarily incompatible with newer NumPy.
    """
    attempts: list[str] = []
    tap_errors: list[str] = []
    astroquery_errors: list[str] = []

    candidates = _query_name_candidates(member_id)
    if extra_candidates:
        expanded = []
        for candidate in extra_candidates:
            expanded.extend(_query_name_candidates(str(candidate)))
        candidates = list(dict.fromkeys(candidates + expanded))

    # Direct TAP first: this avoids importing the user's full Astropy stack.
    for query_name in candidates:
        attempts.append(query_name)
        if progress:
            progress(f"Querying SIMBAD Johnson B and V: {query_name} ...")
        try:
            b_v, b_mag, v_mag, resolved = _query_bv_with_direct_tap(query_name)
            return b_v, b_mag, v_mag, query_name, resolved
        except Exception as exc:
            tap_errors.append(f"{query_name}: {type(exc).__name__}: {exc}")

    # Secondary Astroquery route.  The NumPy compatibility shim above handles
    # the known NumPy-2.4/private-helper import break in older Astropy releases.
    astroquery_available = True
    try:
        _install_numpy_compatibility_aliases()
        from astroquery.simbad import Simbad as _SimbadImportTest  # noqa: F401
    except Exception as exc:
        astroquery_available = False
        astroquery_errors.append(_astroquery_install_diagnostic(exc))

    if astroquery_available:
        for query_name in candidates:
            if progress:
                progress(f"Direct TAP failed; trying Astroquery: {query_name} ...")
            try:
                b_v, b_mag, v_mag, resolved = _query_bv_with_astroquery(query_name)
                return b_v, b_mag, v_mag, query_name, resolved
            except Exception as exc:
                astroquery_errors.append(
                    f"{query_name}: {type(exc).__name__}: {exc}"
                )

    details = []
    if tap_errors:
        details.append("Direct TAP: " + tap_errors[-1])
    if astroquery_errors:
        details.append("Astroquery: " + astroquery_errors[-1])
    detail_text = "\n\n".join(details)
    raise ValueError(
        "The SIMBAD query failed before Johnson B and V could be retrieved for "
        f"{', '.join(attempts) or member_id}. You can enter B-V manually."
        + (f"\n\nDiagnostic details:\n{detail_text}" if detail_text else "")
    )


def resolve_rossby_period(source: str, analysis: Optional[PeriodAnalysis], gp_result: Optional[FitResult]) -> float:
    """Resolve a selected period source to a finite period in days."""
    if analysis is None:
        raise ValueError("Calculate a period before calculating Rossby number.")
    direct = {
        "final_averaged_prot": analysis.final_averaged_prot,
        "final_mean_period": analysis.final_averaged_prot,
        "final_median_period": analysis.final_averaged_prot,
        "ACF_combined": analysis.acf_combined,
        "BGLS_combined": analysis.bgls_combined,
    }
    if source in direct:
        value = direct[source]
    elif source == "GP_MAP":
        if gp_result is None: raise ValueError("Run the GP fit before selecting GP_MAP.")
        value = gp_result.map_period
    elif source == "GP_median":
        if gp_result is None: raise ValueError("Run the GP fit before selecting GP_median.")
        value = gp_result.period_median
    else:
        match = re.fullmatch(r"(ACF|BGLS|LS|GLS|FFT)_(full_sector|sectorA|sectorB)", source)
        if not match: raise ValueError(f"Unknown Rossby period source: {source}")
        measurement = analysis.measurement(match.group(2), match.group(1))
        value = np.nan if measurement is None else measurement.period
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"The selected period source '{source}' is unavailable.")
    return value


def calculate_rossby_number(
    b_v: float, period_source: str, analysis: Optional[PeriodAnalysis],
    gp_result: Optional[FitResult] = None, *, queried_name: str = "",
    resolved_name: str = "", b_mag: float = np.nan, v_mag: float = np.nan,
    period_override: Optional[float] = None, relation: str = "Noyes1984",
    v_ks: float = np.nan, teff: float = np.nan, ks_mag: float = np.nan,
    allow_extrapolation: bool = False, input_provenance: Optional[dict] = None,
) -> RossbyResult:
    period_days = (
        resolve_rossby_period(period_source, analysis, gp_result)
        if period_override is None
        else float(period_override)
    )
    if not np.isfinite(period_days) or period_days <= 0:
        raise ValueError(f"The selected period source '{period_source}' is unavailable.")
    tau_c, input_name, value, valid, note = turnover_time(
        relation, b_v=b_v, v_ks=v_ks, teff=teff, allow_extrapolation=allow_extrapolation)
    return RossbyResult(
        b_v=float(b_v), tau_c_days=tau_c,
        rossby_number=float(period_days/tau_c), period_source=period_source,
        period_days=period_days, queried_name=queried_name,
        resolved_name=resolved_name, b_mag=float(b_mag), v_mag=float(v_mag), note=note,
        relation=relation, input_name=input_name, input_value=value, v_ks=v_ks, ks_mag=ks_mag,
        teff=teff, in_calibration=valid, input_provenance=input_provenance or {"source":"manual"},
    )


def split_lightcurve_halves(lc: LightCurveData) -> dict[str, LightCurveData]:
    """Return the full light curve and time-median sector A/B halves."""
    midpoint = float(np.nanmedian(lc.time))
    masks = {
        "full_sector": np.ones(lc.time.size, dtype=bool),
        "sectorA": lc.time <= midpoint,
        "sectorB": lc.time > midpoint,
    }
    output: dict[str, LightCurveData] = {}
    for name, mask in masks.items():
        output[name] = LightCurveData(
            time=lc.time[mask],
            flux=lc.flux[mask],
            flux_err=lc.flux_err[mask],
            name=f"{lc.name}:{name}",
            source_path=lc.source_path,
            normalization_note=lc.normalization_note,
        )
    return output


def estimate_cadence(time: np.ndarray) -> float:
    dt = np.diff(np.sort(np.asarray(time, dtype=float)))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.nanmedian(dt)) if dt.size else np.nan


def smooth_for_rotation(
    time: np.ndarray,
    flux: np.ndarray,
    window_hours: float = 6.0,
) -> np.ndarray:
    """Savitzky-Golay smoothing used by the supplied BGLS workflow."""
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    if time.size < 50:
        return flux.copy()
    dt = estimate_cadence(time)
    if not np.isfinite(dt) or dt <= 0:
        return flux.copy()
    window = max(int(round((window_hours / 24.0) / dt)), 7)
    if window % 2 == 0:
        window += 1
    if window >= flux.size:
        window = flux.size - 1 if flux.size % 2 == 0 else flux.size
    if window < 7:
        return flux.copy()
    return savgol_filter(flux, window_length=window, polyorder=2, mode="interp")


def bin_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    bin_minutes: float,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    if time.size < 10:
        return time, flux
    bin_days = float(bin_minutes) / (24.0 * 60.0)
    labels = np.floor((time - np.min(time)) / bin_days).astype(np.int64)
    frame = pd.DataFrame({"bin": labels, "time": time, "flux": flux})
    grouped = frame.groupby("bin", sort=True).agg(
        time=("time", "median"), flux=("flux", "median")
    )
    return grouped["time"].to_numpy(), grouped["flux"].to_numpy()


@njit(cache=False)
def bgls_numba(
    t: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    plow: float,
    phigh: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian generalized Lomb-Scargle implementation from the user script."""
    n = len(t)
    n_steps = int(n * (1.0 / plow - 1.0 / phigh))
    n_steps = min(n_steps, max_steps)
    if n_steps < 20:
        return np.empty(0), np.empty(0)

    frequency = np.linspace(1.0 / phigh, 1.0 / plow, n_steps)
    periods = 1.0 / frequency
    weights = 1.0 / (err * err)

    W = 0.0
    bigY = 0.0
    for j in range(n):
        W += weights[j]
        bigY += weights[j] * y[j]

    logp = np.empty(n_steps)
    for i in range(n_steps):
        omega = 2.0 * np.pi * frequency[i]
        s2 = 0.0
        c2 = 0.0
        for j in range(n):
            arg2 = 2.0 * omega * t[j]
            s2 += weights[j] * np.sin(arg2)
            c2 += weights[j] * np.cos(arg2)
        theta = 0.5 * np.arctan2(s2, c2)

        C = 0.0
        S = 0.0
        YCh = 0.0
        YSh = 0.0
        CCh = 0.0
        SSh = 0.0
        for j in range(n):
            x = omega * t[j] - theta
            cosx = np.cos(x)
            sinx = np.sin(x)
            wcosx = weights[j] * cosx
            wsinx = weights[j] * sinx
            C += wcosx
            S += wsinx
            YCh += y[j] * wcosx
            YSh += y[j] * wsinx
            CCh += wcosx * cosx
            SSh += wsinx * sinx

        eps = 1e-300
        log_const = -np.inf
        K = 0.0
        L = 0.0
        M = 0.0
        if abs(CCh) > eps and abs(SSh) > eps:
            K = (C * C * SSh + S * S * CCh - W * CCh * SSh) / (
                2.0 * CCh * SSh
            )
            L = (bigY * CCh * SSh - C * YCh * SSh - S * YSh * CCh) / (
                CCh * SSh
            )
            M = (YCh * YCh * SSh + YSh * YSh * CCh) / (2.0 * CCh * SSh)
            denom = CCh * SSh * abs(K)
            if denom > eps:
                log_const = -0.5 * np.log(denom)

        if K >= 0.0 or not np.isfinite(log_const):
            logp[i] = -np.inf
        else:
            logp[i] = log_const + M - L * L / (4.0 * K)

    max_logp = -np.inf
    for i in range(n_steps):
        if logp[i] > max_logp:
            max_logp = logp[i]
    powers = np.empty(n_steps)
    if not np.isfinite(max_logp):
        for i in range(n_steps):
            powers[i] = np.nan
        return periods, powers

    max_power = 0.0
    for i in range(n_steps):
        powers[i] = np.exp(logp[i] - max_logp)
        if powers[i] > max_power:
            max_power = powers[i]
    if max_power > 0.0:
        for i in range(n_steps):
            powers[i] /= max_power
    return periods, powers


def calculate_bgls_period(
    lc: LightCurveData,
    settings: FitSettings,
) -> PeriodMeasurement:
    if lc.time.size < 50:
        return PeriodMeasurement("", "BGLS", np.nan, note="too_few_points")
    smooth = smooth_for_rotation(lc.time, lc.flux, settings.rotation_smooth_hours)
    time, flux = bin_lightcurve(lc.time, smooth, settings.bgls_bin_minutes)
    if time.size < 20:
        return PeriodMeasurement("", "BGLS", np.nan, note="too_few_binned_points")
    baseline = float(np.max(time) - np.min(time))
    phigh = min(settings.pmax, 0.8 * baseline)
    if phigh <= settings.pmin:
        return PeriodMeasurement("", "BGLS", np.nan, note="baseline_too_short")
    y = flux - np.median(flux)
    sigma = robust_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        return PeriodMeasurement("", "BGLS", np.nan, note="bad_noise")
    err = np.full_like(y, sigma)
    periods, powers = bgls_numba(
        np.asarray(time - np.min(time), dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(err, dtype=np.float64),
        float(settings.pmin),
        float(phigh),
        int(settings.bgls_max_steps),
    )
    if periods.size == 0 or np.all(~np.isfinite(powers)):
        return PeriodMeasurement("", "BGLS", np.nan, note="empty_periodogram")
    index = int(np.nanargmax(powers))
    peak = float(powers[index])
    spread = float(np.nanstd(powers))
    snr = (peak - float(np.nanmedian(powers))) / spread if spread > 0 else np.nan
    return PeriodMeasurement(
        segment="",
        method="BGLS",
        period=float(periods[index]),
        score=float(snr),
        note="BGLS",
        periods=np.asarray(periods),
        power=np.asarray(powers),
    )


def build_centered_period_mixture(
    center_period: float,
    pmin: float,
    pmax: float,
    width: float = 0.2,
) -> np.ndarray:
    """Paper-style P/2, P, 2P mixture centered on a selected combined period."""
    if not np.isfinite(center_period) or center_period <= 0:
        raise ValueError("The selected GP prior center is not finite and positive.")
    components: list[list[float]] = []
    for factor, weight in ((1.0, 0.90), (0.5, 0.05), (2.0, 0.05)):
        period = center_period * factor
        if pmin <= period <= pmax:
            components.append([weight, math.log(period), width])
    if not components:
        raise ValueError("The selected prior and its harmonics lie outside Pmin/Pmax.")
    mixture = np.asarray(components, float)
    mixture[:, 0] /= mixture[:, 0].sum()
    return mixture


def _finite_mean(values: Sequence[float]) -> tuple[float, int]:
    array = np.asarray(values, float)
    array = array[np.isfinite(array) & (array > 0)]
    return (float(np.mean(array)), int(array.size)) if array.size else (np.nan, 0)


def calculate_period_analysis(
    raw_lc: LightCurveData,
    settings: FitSettings,
    segment_selection: str = "all",
    method_selection: str = "both",
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[LightCurveData, PeriodAnalysis]:
    """Calculate ACF/BGLS periods for full, A, and B selections."""
    if progress:
        progress("Cleaning and normalizing light curve")
    clean = preprocess_lightcurve(raw_lc, settings)
    baseline = float(clean.time[-1] - clean.time[0])
    settings.validate(baseline=baseline)

    segment_key = segment_selection.strip().lower()
    if segment_key in {"all", "all (full+a+b)", "full+a+b"}:
        requested_segments = list(SEGMENT_ORDER)
    else:
        lookup = {
            "full": "full_sector",
            "full_sector": "full_sector",
            "halfa": "sectorA",
            "half a": "sectorA",
            "sectora": "sectorA",
            "sector a": "sectorA",
            "halfb": "sectorB",
            "half b": "sectorB",
            "sectorb": "sectorB",
            "sector b": "sectorB",
        }
        if segment_key not in lookup:
            raise ValueError("Segment must be all, full_sector, sectorA, or sectorB.")
        requested_segments = [lookup[segment_key]]

    method_key = method_selection.strip().lower()
    if method_key in {"both", "acf+bgls", "bgls+acf"}:
        requested_methods = ["ACF", "BGLS"]
    elif method_key in {"all", "all methods"}:
        requested_methods = list(METHOD_ORDER)
    else:
        requested_methods = list(dict.fromkeys(x.strip().upper() for x in re.split(r"[,+]", method_key)))
        if not requested_methods or any(x not in METHOD_ORDER for x in requested_methods):
            raise ValueError("Choose ACF, BGLS, LS, GLS, FFT, all, or a comma-separated combination.")

    segments = split_lightcurve_halves(clean)
    measurements: dict[tuple[str, str], PeriodMeasurement] = {}
    total = len(requested_segments) * len(requested_methods)
    counter = 0
    for segment in requested_segments:
        segment_lc = segments[segment]
        for method in requested_methods:
            counter += 1
            if progress:
                progress(f"{method} {segment} ({counter}/{total})")
            if segment_lc.time.size < 50:
                measurement = PeriodMeasurement(
                    segment, method, np.nan, note="too_few_points"
                )
            elif method in {"LS", "GLS", "FFT"}:
                measurement = calculate_classical_period(segment_lc, settings, method)
                measurement.segment = segment
            elif method == "BGLS":
                measurement = calculate_bgls_period(segment_lc, settings)
                measurement.segment = segment
            else:
                segment_baseline = float(segment_lc.time[-1] - segment_lc.time[0])
                effective_pmax = min(settings.pmax, 0.8 * segment_baseline)
                if effective_pmax <= settings.pmin:
                    measurement = PeriodMeasurement(
                        segment, "ACF", np.nan, note="baseline_too_short"
                    )
                else:
                    local_settings = replace(settings, pmax=effective_pmax)
                    candidates = calculate_acf_candidates(
                        segment_lc, local_settings, progress=None
                    )
                    if candidates:
                        best = max(candidates, key=lambda candidate: candidate.quality)
                        measurement = PeriodMeasurement(
                            segment=segment,
                            method="ACF",
                            period=float(best.period),
                            score=float(best.quality),
                            note="ACF",
                            periods=np.asarray(best.lags),
                            power=np.asarray(best.acf),
                            acf_candidate=best,
                        )
                    else:
                        measurement = PeriodMeasurement(
                            segment, "ACF", np.nan, note="no_acf_candidate"
                        )
            if measurement.periods.size:
                measurement.peaks = strongest_peaks(measurement, measurement.diagnostic_metadata.get("baseline_days", float(np.ptp(segment_lc.time))), settings.peak_count)
            measurements[(segment, method)] = measurement

    acf_values = [
        measurements[(segment, "ACF")].period
        for segment in SEGMENT_ORDER
        if (segment, "ACF") in measurements
    ]
    bgls_values = [
        measurements[(segment, "BGLS")].period
        for segment in SEGMENT_ORDER
        if (segment, "BGLS") in measurements
    ]
    reduce = np.median if settings.aggregation == "median" else np.mean
    def summary(values):
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite) & (finite > 0)]
        return (float(reduce(finite)) if finite.size else np.nan), int(finite.size)
    acf_combined, acf_n = summary(acf_values)
    bgls_combined, bgls_n = summary(bgls_values)
    method_combined = {method: summary([m.period for m in measurements.values() if m.method == method])[0]
                       for method in requested_methods}
    final_period, _ = summary([m.period for m in measurements.values()])

    analysis = PeriodAnalysis(
        measurements=measurements,
        acf_combined=acf_combined,
        bgls_combined=bgls_combined,
        final_averaged_prot=final_period,
        acf_n_used=acf_n,
        bgls_n_used=bgls_n,
        requested_segments=requested_segments,
        requested_methods=requested_methods,
        aggregation=settings.aggregation, method_combined=method_combined,
    )
    if progress:
        progress(
            "Period analysis complete: "
            f"ACF_combined={acf_combined:.6g}, "
            f"BGLS_combined={bgls_combined:.6g}, final={final_period:.6g} d"
        )
    return clean, analysis


def period_analysis_dataframe(
    metadata: LCMetadata,
    analysis: PeriodAnalysis,
    gp_result: Optional[FitResult] = None,
    rossby_result: Optional[RossbyResult] = None,
) -> pd.DataFrame:
    """Build the one-light-curve result table, including optional Rossby data."""
    rows: list[dict] = []
    rossby = {
        "b_v": np.nan, "tau_c_days": np.nan, "rossby_number": np.nan,
        "rossby_period_source": "", "rossby_period_days": np.nan,
        "bv_query_name": "", "bv_resolved_name": "",
    }
    if rossby_result is not None:
        rossby.update({
            "b_v": rossby_result.b_v,
            "tau_c_days": rossby_result.tau_c_days,
            "rossby_number": rossby_result.rossby_number,
            "rossby_period_source": rossby_result.period_source,
            "rossby_period_days": rossby_result.period_days,
            "bv_query_name": rossby_result.queried_name,
            "bv_resolved_name": rossby_result.resolved_name,
        })

    def add_row(period, method, note):
        row = {
            "member_id": metadata.member_id, "sector": metadata.sector,
            "prot": period, "method": method, "note": note,
            "lc_source": metadata.lc_source,
            "gp_backend": gp_result.gp_backend if gp_result is not None else "",
            "gp_kernel": gp_result.gp_kernel if gp_result is not None else "",
        }
        row.update(rossby)
        rows.append(row)

    for method in METHOD_ORDER:
        for segment in SEGMENT_ORDER:
            m = analysis.measurement(segment, method)
            if m is None: continue
            note = method if np.isfinite(m.period) else f"{method}:{m.note}"
            add_row(m.period, segment, note)

    for period, method, note in [
        (analysis.acf_combined, "combined", f"ACF_combined ({analysis.acf_n_used}/3)"),
        (analysis.bgls_combined, "combined", f"BGLS_combined ({analysis.bgls_n_used}/3)"),
        (analysis.final_averaged_prot, "combined", "final_averaged_prot"),
    ]:
        if np.isfinite(period): add_row(period, method, note)
    if gp_result is not None:
        add_row(
            gp_result.map_period, "",
            f"GP[{gp_result.gp_backend} / {gp_result.gp_kernel}] prior={gp_result.gp_prior_source}",
        )

    columns = [
        "member_id", "sector", "prot", "method", "note", "lc_source",
        "gp_backend", "gp_kernel", "b_v", "tau_c_days", "rossby_number", "rossby_period_source",
        "rossby_period_days", "bv_query_name", "bv_resolved_name",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["aggregation"] = analysis.aggregation
    if rossby_result is not None:
        for key in ("relation", "input_name", "input_value", "v_ks", "ks_mag", "teff", "in_calibration", "note"):
            frame["rossby_" + key] = getattr(rossby_result, key)
        frame["rossby_input_provenance"] = json.dumps(rossby_result.input_provenance)
    return frame




# -----------------------------------------------------------------------------
# Batch aggregation helpers
# -----------------------------------------------------------------------------

BATCH_AGGREGATIONS = ("mean", "median")


def normalize_member_key(member_id: str) -> str:
    """Return a conservative grouping key for repeated light curves of one star."""
    text = strip_lc_product_suffix(str(member_id)).strip().lower()
    text = re.sub(r"^\*+\s*", "", text)
    text = re.sub(r"^v\*\s*", "", text)
    text = re.sub(r"[\s_]+", "", text)
    text = re.sub(r"[^a-z0-9+\-.]", "", text)
    return text or "unknown"


def batch_lc_period(
    analysis: PeriodAnalysis,
    gp_result: Optional[FitResult],
    source: str,
) -> tuple[float, str]:
    """Select the scalar period from one LC that will feed the star summary."""
    key = source.strip().lower()
    if key == "final_averaged_prot":
        value = analysis.final_averaged_prot
    elif key == "acf_combined":
        value = analysis.acf_combined
    elif key == "bgls_combined":
        value = analysis.bgls_combined
    elif key == "gp_map":
        value = gp_result.map_period if gp_result is not None else np.nan
    elif key == "gp_median":
        value = gp_result.period_median if gp_result is not None else np.nan
    else:
        raise ValueError(f"Unknown batch period source: {source}")
    if np.isfinite(value) and value > 0:
        return float(value), source
    return np.nan, f"{source}:unavailable"


def _measurement_period(
    analysis: PeriodAnalysis,
    segment: str,
    method: str,
) -> float:
    measurement = analysis.measurement(segment, method)
    if measurement is None or not np.isfinite(measurement.period):
        return np.nan
    return float(measurement.period)


def _period_consistency_flag(values: Sequence[float], tolerance: float = 0.20) -> str:
    """Describe direct or factor-of-two consistency among per-LC periods."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return "no_period"
    if array.size == 1:
        return "single_lc"

    median = float(np.median(array))
    direct_fractional = np.max(np.abs(array - median) / median)
    if direct_fractional <= tolerance:
        return "consistent"

    best_max_log_distance = np.inf
    factors = np.asarray([0.5, 1.0, 2.0])
    for reference in array:
        aligned = []
        for value in array:
            candidates = value * factors
            pick = candidates[np.argmin(np.abs(np.log(candidates / reference)))]
            aligned.append(pick)
        aligned = np.asarray(aligned)
        max_log_distance = float(np.max(np.abs(np.log(aligned / reference))))
        best_max_log_distance = min(best_max_log_distance, max_log_distance)
    if best_max_log_distance <= math.log1p(tolerance):
        return "factor_of_two_consistent"
    return "inconsistent"


def build_batch_tables(
    records: Sequence[dict],
    period_source: str,
    aggregation: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All method/segment rows, per-LC summaries and equal-sector star summaries.

    Final mean: mean of sector means. Final median: median of sector medians.
    Missing sector/identity values cannot contribute to the final star period.
    ``period_source`` remains a per-LC diagnostic selector; star summaries use
    measured non-GP periods so GP posteriors are not averaged with periodograms.
    """
    aggregation_key = aggregation.strip().lower()
    if aggregation_key not in BATCH_AGGREGATIONS:
        aggregation_key = "median"

    detailed_frames: list[pd.DataFrame] = []
    lc_rows: list[dict] = []

    for index, record in enumerate(records, start=1):
        path = str(record.get("path", ""))
        metadata = record.get("metadata") or infer_lc_metadata(path)
        member_key = normalize_member_key(metadata.member_id)
        error = str(record.get("error", "") or "")
        analysis = record.get("analysis")
        gp_result = record.get("gp_result")

        base_row = {
            "batch_index": index,
            "member_key": member_key,
            "member_id": metadata.member_id,
            "sector": metadata.sector,
            "lc_source": metadata.lc_source,
            "lc_file": Path(path).name,
            "lc_path": path,
        }

        if analysis is None:
            lc_rows.append({
                **base_row,
                "acf_full_sector": np.nan, "acf_sectorA": np.nan, "acf_sectorB": np.nan,
                "acf_combined": np.nan,
                "bgls_full_sector": np.nan, "bgls_sectorA": np.nan, "bgls_sectorB": np.nan,
                "bgls_combined": np.nan,
                "final_averaged_prot": np.nan,
                "gp_map": np.nan, "gp_median": np.nan,
                "gp_p16": np.nan, "gp_p84": np.nan,
                "gp_backend": "", "gp_kernel": "", "gp_prior_source": "",
                "selected_lc_prot": np.nan,
                "selected_period_source": period_source,
                "status": "failed",
                "error": error or "analysis_failed",
            })
            continue

        selected_period, selected_note = batch_lc_period(analysis, gp_result, period_source)

        detail = period_analysis_dataframe(metadata, analysis, gp_result, None)
        if not detail.empty:
            # Only raw full/A/B method measurements. No combined/final/GP rows.
            detail = detail.loc[
                detail["method"].isin(SEGMENT_ORDER)
                & detail["note"].astype(str).str.startswith(METHOD_ORDER)
            ].copy()
            detail.insert(0, "batch_index", index)
            detail.insert(1, "member_key", member_key)
            detail.insert(2, "lc_file", Path(path).name)
            detail.insert(3, "lc_path", path)
            detail["batch_status"] = "ok"
            detail = detail[[
                "batch_index", "member_key", "member_id", "sector", "lc_source",
                "lc_file", "lc_path", "prot", "method", "note", "batch_status",
            ]]
            detailed_frames.append(detail)

        lc_rows.append({
            **base_row,
            **{f"{method.lower()}_{segment}": _measurement_period(analysis, segment, method)
               for method in METHOD_ORDER for segment in SEGMENT_ORDER},
            "acf_full_sector": _measurement_period(analysis, "full_sector", "ACF"),
            "acf_sectorA": _measurement_period(analysis, "sectorA", "ACF"),
            "acf_sectorB": _measurement_period(analysis, "sectorB", "ACF"),
            "acf_combined": analysis.acf_combined,
            "bgls_full_sector": _measurement_period(analysis, "full_sector", "BGLS"),
            "bgls_sectorA": _measurement_period(analysis, "sectorA", "BGLS"),
            "bgls_sectorB": _measurement_period(analysis, "sectorB", "BGLS"),
            "bgls_combined": analysis.bgls_combined,
            "final_averaged_prot": analysis.final_averaged_prot,
            "gp_map": gp_result.map_period if gp_result is not None else np.nan,
            "gp_median": gp_result.period_median if gp_result is not None else np.nan,
            "gp_p16": gp_result.period_p16 if gp_result is not None else np.nan,
            "gp_p84": gp_result.period_p84 if gp_result is not None else np.nan,
            "gp_backend": gp_result.gp_backend if gp_result is not None else "",
            "gp_kernel": gp_result.gp_kernel if gp_result is not None else "",
            "gp_prior_source": gp_result.gp_prior_source if gp_result is not None else "",
            "selected_lc_prot": selected_period,
            "selected_period_source": selected_note,
            "status": "ok" if np.isfinite(selected_period) else "period_unavailable",
            "error": error,
        })

    detailed = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()
    lc_summary = pd.DataFrame(lc_rows)

    star_rows: list[dict] = []
    if not lc_summary.empty:
        for member_key, group in lc_summary.groupby("member_key", sort=False, dropna=False):
            # Reduce raw method/segment estimates within sectors, then across
            # sectors. The optional per-LC diagnostic selector does not alter this.
            detail_group = detailed.loc[detailed["member_key"] == member_key].copy() if not detailed.empty else pd.DataFrame()
            if member_key in {"", "unknown", "nan", "none"}:
                detail_group = pd.DataFrame()
            if not detail_group.empty:
                detail_group["prot"] = pd.to_numeric(detail_group["prot"], errors="coerce")
                good_sector = detail_group["sector"].notna() & ~detail_group["sector"].astype(str).str.strip().str.lower().isin(["", "nan", "unknown", "none"])
                detail_group = detail_group.loc[good_sector & np.isfinite(detail_group["prot"]) & (detail_group["prot"] > 0)]
            if not detail_group.empty:
                sector_group = detail_group.groupby("sector")["prot"]
                sector_means, sector_medians = sector_group.mean(), sector_group.median()
                finite = sector_means if aggregation_key == "mean" else sector_medians
            else:
                sector_means = sector_medians = finite = pd.Series(dtype=float)
            selected_finite = finite
            if finite.size:
                mean_period = float(sector_means.mean())
                median_period = float(sector_medians.median())
                std_period = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
                min_period = float(np.min(finite))
                max_period = float(np.max(finite))
            else:
                mean_period = median_period = np.nan
                std_period = min_period = max_period = np.nan
            if selected_finite.size:
                selected_mean = float(np.mean(selected_finite))
                selected_median = float(np.median(selected_finite))
                final_period = mean_period if aggregation_key == "mean" else median_period
            else:
                final_period = np.nan

            sectors = sorted({
                str(value) for value in group["sector"]
                if str(value).strip() and str(value).lower() != "nan"
            })
            sources = sorted({
                str(value) for value in group["lc_source"]
                if str(value).strip() and str(value).lower() != "nan"
            })
            filenames = [str(value) for value in group["lc_file"]]
            failures = group.loc[group["status"] == "failed", "lc_file"].astype(str).tolist()
            gp_backends = sorted({str(v) for v in group.get("gp_backend", pd.Series(dtype=str)) if str(v).strip()})
            gp_kernels = sorted({str(v) for v in group.get("gp_kernel", pd.Series(dtype=str)) if str(v).strip()})
            star_rows.append({
                "member_key": member_key,
                "member_id": str(group["member_id"].iloc[0]),
                "final_mean_period": mean_period,
                "final_median_period": median_period,
                "final_prot": final_period,
                "period_source_per_lc": "finite method/segment estimates -> sector -> star",
                "aggregation": aggregation_key,
                "mean_prot": mean_period,
                "median_prot": median_period,
                "std_prot": std_period,
                "min_prot": min_period,
                "max_prot": max_period,
                "n_lcs_selected": int(len(group)),
                "n_lcs_used": int(np.isfinite(pd.to_numeric(group["final_averaged_prot"], errors="coerce")).sum()),
                "n_sectors_used": int(finite.size),
                "consistency_flag": _period_consistency_flag(finite.to_numpy()),
                "sectors": ",".join(sectors),
                "lc_sources": ",".join(sources),
                "gp_backends": ",".join(gp_backends),
                "gp_kernels": ",".join(gp_kernels),
                "lc_files": " | ".join(filenames),
                "n_failed_lcs": int(len(failures)),
                "failed_lcs": " | ".join(failures),
                "b_v": np.nan, "b_mag": np.nan, "v_mag": np.nan,
                "tau_c_days": np.nan, "rossby_number": np.nan,
                "rossby_period_source": "", "rossby_period_days": np.nan,
                "bv_query_name": "", "bv_resolved_name": "", "rossby_note": "",
            })
    star_summary = pd.DataFrame(star_rows)

    # Repeat the star-level mean/median on each LC row so the per-LC summary is
    # self-contained when exported. With a one-LC star both values equal that
    # LC's selected period by construction.
    if not lc_summary.empty and not star_summary.empty:
        references = star_summary[["member_key", "final_mean_period", "final_median_period"]]
        lc_summary = lc_summary.merge(references, on="member_key", how="left")

    return detailed, lc_summary, star_summary


def propagate_star_rossby_to_batch_tables(
    detailed: pd.DataFrame,
    lc_summary: pd.DataFrame,
    star_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy one catalog/Rossby result per star into the two LC-level tables."""
    if star_summary.empty:
        return detailed, lc_summary
    lookup_columns = [
        "member_key", "b_v", "tau_c_days", "rossby_number",
        "rossby_period_source", "rossby_period_days",
        "bv_query_name", "bv_resolved_name", "rossby_note",
        "rossby_relation", "rossby_input_name", "rossby_input_value", "v_ks", "ks_mag", "teff",
        "in_calibration", "input_provenance",
    ]
    lookup = star_summary.reindex(columns=lookup_columns).drop_duplicates("member_key")

    if not lc_summary.empty:
        lc_summary = lc_summary.drop(
            columns=[column for column in lookup_columns[1:] if column in lc_summary.columns],
            errors="ignore",
        ).merge(lookup, on="member_key", how="left")

    if not detailed.empty:
        replacement = {
            row["member_key"]: row for _, row in lookup.iterrows()
        }
        for column in lookup_columns[1:]:
            detailed[column] = detailed["member_key"].map(
                lambda key, c=column: replacement.get(key, {}).get(c, np.nan)
            )
        # rossby_period_source/days already come from the star lookup above.
    return detailed, lc_summary


# -----------------------------------------------------------------------------
# Light-curve loading and preprocessing
# -----------------------------------------------------------------------------


def _load_table_lightcurve(path: str) -> LightCurveData:
    # sep=None lets pandas infer comma/whitespace/tab separators.
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, comment="#")

    if df.empty:
        raise ValueError("The table is empty.")

    time_col = choose_column(
        df.columns,
        ["time", "t", "btjd", "bkjd", "jd", "mjd", "timestamp"],
    )
    flux_col = choose_column(
        df.columns,
        [
            "detrended_flux",
            "normalized_flux",
            "norm_flux",
            "flux",
            "pdcsap_flux",
            "sap_flux",
            "relative_flux",
            "residual_flux",
        ],
    )
    err_col = choose_column(
        df.columns,
        [
            "flux_err",
            "detrended_flux_err",
            "normalized_flux_err",
            "error",
            "err",
            "sigma",
            "uncertainty",
        ],
    )

    if time_col is None or flux_col is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) < 2:
            raise ValueError(
                "Could not identify time and flux columns. Expected names such as "
                "time/flux or at least two numeric columns."
            )
        time_col = time_col or numeric[0]
        flux_col = flux_col or numeric[1]
        if err_col is None and len(numeric) >= 3:
            err_col = numeric[2]

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[flux_col], errors="coerce").to_numpy(float)
    if err_col is not None:
        yerr = pd.to_numeric(df[err_col], errors="coerce").to_numpy(float)
    else:
        yerr = np.full_like(y, np.nan)

    return LightCurveData(
        t,
        y,
        yerr,
        name=Path(path).stem,
        source_path=str(path),
        normalization_note=f"table columns: {time_col}, {flux_col}, {err_col}",
    )


def _load_fits_lightcurve(path: str) -> LightCurveData:
    errors: list[str] = []

    # First choice: Lightkurve understands SPOC, QLP, and many common products.
    try:
        import lightkurve as lk  # type: ignore

        obj = lk.read(path, quality_bitmask="hard")
        if not isinstance(obj, lk.LightCurve) and hasattr(obj, "to_lightcurve"):
            obj = obj.to_lightcurve()
        if hasattr(obj, "remove_nans"):
            obj = obj.remove_nans()

        t = safe_float_array(obj.time)
        y = safe_float_array(obj.flux)
        if getattr(obj, "flux_err", None) is not None:
            yerr = safe_float_array(obj.flux_err)
        else:
            yerr = np.full_like(y, np.nan)
        return LightCurveData(
            t,
            y,
            yerr,
            name=Path(path).stem,
            source_path=str(path),
            normalization_note="loaded with lightkurve",
        )
    except Exception as exc:
        errors.append(f"lightkurve: {exc}")

    # Fallback: inspect binary FITS tables directly.
    try:
        from astropy.io import fits  # type: ignore

        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                names = getattr(data, "names", None)
                if data is None or names is None:
                    continue
                time_col = choose_column(names, ["TIME", "BTJD", "BKJD", "BJD"])
                flux_col = choose_column(
                    names,
                    [
                        "PDCSAP_FLUX",
                        "KSPSAP_FLUX",
                        "SAP_FLUX",
                        "FLUX",
                        "DETRENDED_FLUX",
                    ],
                )
                err_col = choose_column(
                    names,
                    [
                        "PDCSAP_FLUX_ERR",
                        "KSPSAP_FLUX_ERR",
                        "SAP_FLUX_ERR",
                        "FLUX_ERR",
                    ],
                )
                quality_col = choose_column(names, ["QUALITY", "SAP_QUALITY"])
                if time_col is None or flux_col is None:
                    continue
                t = np.asarray(data[time_col], dtype=float)
                y = np.asarray(data[flux_col], dtype=float)
                yerr = (
                    np.asarray(data[err_col], dtype=float)
                    if err_col is not None
                    else np.full_like(y, np.nan)
                )
                if quality_col is not None:
                    quality = np.asarray(data[quality_col])
                    good = quality == 0
                    t, y, yerr = t[good], y[good], yerr[good]
                return LightCurveData(
                    t,
                    y,
                    yerr,
                    name=Path(path).stem,
                    source_path=str(path),
                    normalization_note="loaded from FITS table",
                )
    except Exception as exc:
        errors.append(f"astropy: {exc}")

    raise RuntimeError(
        "Could not read this FITS light curve. Install lightkurve and astropy.\n"
        + "\n".join(errors)
    )


def load_lightcurve(path: str) -> LightCurveData:
    suffix = Path(path).suffix.lower()
    if suffix in {".fits", ".fit", ".fts", ".fz"}:
        return _load_fits_lightcurve(path)
    if suffix in {".csv", ".txt", ".dat", ".tsv"}:
        return _load_table_lightcurve(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def preprocess_lightcurve(
    lc: LightCurveData,
    settings: FitSettings,
) -> LightCurveData:
    t = np.asarray(lc.time, dtype=float)
    y = np.asarray(lc.flux, dtype=float)
    yerr = np.asarray(lc.flux_err, dtype=float)

    finite = np.isfinite(t) & np.isfinite(y)
    t, y, yerr = t[finite], y[finite], yerr[finite]
    if t.size < 30:
        raise ValueError("Fewer than 30 finite light-curve points remain.")

    order = np.argsort(t)
    t, y, yerr = t[order], y[order], yerr[order]

    # Remove duplicate times, retaining the first occurrence.
    _, unique_idx = np.unique(t, return_index=True)
    unique_idx = np.sort(unique_idx)
    t, y, yerr = t[unique_idx], y[unique_idx], yerr[unique_idx]

    med = float(np.median(y))
    sig = robust_sigma(y)
    if not np.isfinite(sig) or sig <= 0:
        raise ValueError("The flux has zero or invalid scatter.")

    # If the baseline flux dominates its variability, convert to relative flux.
    if abs(med) > 5.0 * sig and abs(med) > 1e-12:
        y = y / med - 1.0
        yerr = yerr / abs(med)
        norm_note = "median-divided and unit-subtracted"
    else:
        y = y - med
        norm_note = "median-subtracted"

    # Iterative asymmetric robust clipping. This is useful for young-star flares.
    keep = np.ones(y.size, dtype=bool)
    for _ in range(3):
        center = np.median(y[keep])
        scatter = robust_sigma(y[keep])
        if not np.isfinite(scatter) or scatter <= 0:
            break
        new_keep = (y >= center - settings.clip_lower * scatter) & (
            y <= center + settings.clip_upper * scatter
        )
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    t, y, yerr = t[keep], y[keep], yerr[keep]
    if t.size < 30:
        raise ValueError("Sigma clipping left fewer than 30 points.")

    # Estimate missing/non-positive uncertainties from first differences.
    noise = robust_sigma(np.diff(y)) / math.sqrt(2.0)
    if not np.isfinite(noise) or noise <= 0:
        noise = max(robust_sigma(y) * 0.1, 1e-8)
    bad_err = ~np.isfinite(yerr) | (yerr <= 0)
    yerr[bad_err] = noise
    # Avoid unrealistically tiny formal uncertainties causing numerical trouble.
    floor = max(noise * 0.03, 1e-10)
    yerr = np.maximum(yerr, floor)

    # Recenter once after clipping.
    y = y - np.median(y)

    return LightCurveData(
        t,
        y,
        yerr,
        name=lc.name,
        source_path=lc.source_path,
        normalization_note=f"{lc.normalization_note}; {norm_note}",
    )


# -----------------------------------------------------------------------------
# ACF-informed period prior
# -----------------------------------------------------------------------------


def regularize_for_acf(
    t: np.ndarray,
    y: np.ndarray,
    pmin: float,
    max_points: int = 30000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Interpolate onto a regular grid and return a real-data mask.

    The target cadence is never coarser than Pmin/20, preserving at least
    ~20 samples per shortest searched period while keeping 20-second TESS
    light curves computationally manageable.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    dt_native = float(np.median(np.diff(t)))
    baseline = float(t[-1] - t[0])
    dt_target = max(dt_native, pmin / 20.0, baseline / max(max_points - 1, 1))
    grid = np.arange(t[0], t[-1] + 0.5 * dt_target, dt_target)
    interp = np.interp(grid, t, y)

    # Mark intervals that cross substantial original gaps. The paper's code
    # fills gaps before filtering and zeros the inserted samples afterward.
    real_mask = np.ones(grid.size, dtype=float)
    gap_threshold = max(5.0 * dt_native, 2.5 * dt_target)
    gap_inds = np.where(np.diff(t) > gap_threshold)[0]
    for idx in gap_inds:
        real_mask[(grid > t[idx]) & (grid < t[idx + 1])] = 0.0

    return grid, interp, real_mask, dt_target


def weighted_autocorrelation(
    values: np.ndarray,
    weights: np.ndarray,
    max_lag_samples: int,
) -> np.ndarray:
    """FFT ACF with pair-count normalization for zero-filled gaps."""
    z = np.asarray(values, float).copy()
    w = np.asarray(weights, float).copy()
    good = w > 0
    if good.sum() < 10:
        raise ValueError("Too few real samples for ACF calculation.")
    z[good] -= np.average(z[good], weights=w[good])
    z[~good] = 0.0

    n = z.size
    nfft = 1 << int(np.ceil(np.log2(max(2 * n - 1, 2))))
    fz = np.fft.rfft(z * w, n=nfft)
    fw = np.fft.rfft(w, n=nfft)
    numerator = np.fft.irfft(fz * np.conj(fz), n=nfft)[:n]
    denominator = np.fft.irfft(fw * np.conj(fw), n=nfft)[:n]
    denominator = np.maximum(denominator, 1.0)
    acf = numerator / denominator
    if not np.isfinite(acf[0]) or acf[0] <= 0:
        raise ValueError("Invalid zero-lag ACF normalization.")
    acf /= acf[0]
    return acf[: max_lag_samples + 1]


def paper_pmax_ladder(pmin: float, pmax: float, baseline: float) -> list[float]:
    standard = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=float)
    upper = min(pmax, max(baseline * 0.75, pmin * 1.5))
    vals = [float(v) for v in standard if v > pmin and v <= upper]
    if pmax > pmin and pmax <= baseline and not any(np.isclose(pmax, vals)):
        vals.append(float(pmax))
    if not vals:
        vals = [float(min(pmax, baseline * 0.75))]
    return sorted(set(round(v, 10) for v in vals if v > pmin))


def _damped_oscillator(lag: np.ndarray, amplitude: float, tau: float, period: float):
    return amplitude * np.exp(-lag / tau) * np.cos(2.0 * np.pi * lag / period)


def calculate_acf_candidates(
    lc: LightCurveData,
    settings: FitSettings,
    progress: Optional[Callable[[str], None]] = None,
) -> list[ACFCandidate]:
    t, y = lc.time, lc.flux
    baseline = float(t[-1] - t[0])
    grid, interp, real_mask, dt = regularize_for_acf(
        t, y, settings.pmin, settings.acf_max_points
    )
    fs = 1.0 / dt
    nyquist = 0.5 * fs
    highcut = min(1.0 / settings.pmin, nyquist * 0.95)
    ladder = paper_pmax_ladder(settings.pmin, settings.pmax, baseline)
    candidates: list[ACFCandidate] = []

    for index, pmax_filter in enumerate(ladder, start=1):
        if progress:
            progress(f"ACF candidate {index}/{len(ladder)}: Pmax={pmax_filter:g} d")
        lowcut = 1.0 / pmax_filter
        if not (0 < lowcut < highcut < nyquist):
            continue

        sos = butter(
            settings.acf_filter_order,
            [lowcut, highcut],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        try:
            filtered = sosfiltfilt(sos, interp)
        except ValueError:
            filtered = sosfilt(sos, interp)
        filtered = filtered - np.median(filtered[real_mask > 0])
        filtered[real_mask == 0] = 0.0

        max_lag = min(2.0 * pmax_filter, baseline * 0.95)
        max_lag_samples = min(int(max_lag / dt), grid.size - 2)
        if max_lag_samples < 10:
            continue
        ac = weighted_autocorrelation(filtered, real_mask, max_lag_samples)
        lags = np.arange(ac.size) * dt

        # Paper: boxcar smooth with width Pmax/10.
        smooth_samples = max(1, int(round((pmax_filter / 10.0) / dt)))
        if smooth_samples >= 3:
            ac_smooth = uniform_filter1d(ac, size=smooth_samples, mode="nearest")
        else:
            ac_smooth = ac

        # Select the first meaningful peak after the first trough.
        min_distance = max(1, int(0.25 * settings.pmin / dt))
        peaks, _ = find_peaks(ac_smooth, distance=min_distance)
        troughs, _ = find_peaks(-ac_smooth, distance=min_distance)
        peaks = peaks[lags[peaks] >= settings.pmin]
        troughs = troughs[lags[troughs] > 0]
        if peaks.size == 0 or troughs.size == 0:
            continue
        first_trough = troughs[0]
        peaks_after = peaks[peaks > first_trough]
        if peaks_after.size == 0:
            continue
        peak_idx = int(peaks_after[0])
        period = float(lags[peak_idx])
        if not (settings.pmin <= period <= settings.pmax):
            continue

        prev_troughs = troughs[troughs < peak_idx]
        next_troughs = troughs[troughs > peak_idx]
        prev_value = ac_smooth[prev_troughs[-1]] if prev_troughs.size else 0.0
        h1 = float(ac_smooth[peak_idx] - prev_value)
        if next_troughs.size:
            h2 = float(ac_smooth[peak_idx] - ac_smooth[next_troughs[0]])
            height = 0.5 * (h1 + h2)
        else:
            height = h1
        if not np.isfinite(height) or height <= 0:
            continue

        tau_max = max(20.0 * pmax_filter / period, period * 1.01)
        try:
            popt, _ = curve_fit(
                lambda x, amp, tau: _damped_oscillator(x, amp, tau, period),
                lags,
                ac_smooth,
                p0=[1.0, max(2.0 * period, dt)],
                bounds=([-3.0, dt], [3.0, tau_max]),
                maxfev=10000,
            )
            amplitude, tau = float(popt[0]), float(popt[1])
        except Exception:
            amplitude, tau = 1.0, min(2.0 * period, tau_max)

        model = _damped_oscillator(lags, amplitude, tau, period)
        residual_ss = float(np.sum((model - ac_smooth) ** 2))
        n_lags = int(lags.size)
        quality = (tau / period) * (n_lags * height / max(residual_ss, 1e-12))
        if not np.isfinite(quality) or quality <= 0:
            continue

        candidates.append(
            ACFCandidate(
                pmax_filter=pmax_filter,
                period=period,
                peak_height=height,
                tau=tau,
                quality=quality,
                residual_ss=residual_ss,
                n_lags=n_lags,
                lags=lags,
                acf=ac_smooth,
                damped_model=model,
            )
        )

    return candidates


def build_period_mixture(
    candidates: Sequence[ACFCandidate],
    pmin: float,
    pmax: float,
    width: float = 0.2,
) -> np.ndarray:
    """Return rows [weight, mean_lnP, sigma_lnP]."""
    valid = [c for c in candidates if np.isfinite(c.quality) and c.quality > 0]
    if not valid:
        return np.empty((0, 3), dtype=float)
    qsum = sum(c.quality for c in valid)
    components: list[list[float]] = []
    for candidate in valid:
        base_weight = candidate.quality / qsum
        for factor, relative_weight in [(1.0, 0.90), (0.5, 0.05), (2.0, 0.05)]:
            period = candidate.period * factor
            if pmin <= period <= pmax:
                components.append(
                    [base_weight * relative_weight, math.log(period), width]
                )
    if not components:
        return np.empty((0, 3), dtype=float)
    mixture = np.asarray(components, float)
    mixture[:, 0] /= mixture[:, 0].sum()
    return mixture


def period_prior_logpdf(ln_period: float, mixture: np.ndarray) -> float:
    if mixture.size == 0:
        return 0.0  # log-uniform in P means uniform in ln(P).
    terms = np.log(mixture[:, 0]) + normal_logpdf(
        ln_period, mixture[:, 1], mixture[:, 2]
    )
    return float(logsumexp(terms))


def sample_period_prior(
    rng: np.random.Generator,
    mixture: np.ndarray,
    pmin: float,
    pmax: float,
) -> float:
    lo, hi = math.log(pmin), math.log(pmax)
    if mixture.size == 0:
        return float(rng.uniform(lo, hi))
    for _ in range(1000):
        idx = int(rng.choice(mixture.shape[0], p=mixture[:, 0]))
        value = float(rng.normal(mixture[idx, 1], mixture[idx, 2]))
        if lo <= value <= hi:
            return value
    return float(np.clip(mixture[np.argmax(mixture[:, 0]), 1], lo, hi))


# -----------------------------------------------------------------------------
# Quasi-periodic Gaussian-process model
# -----------------------------------------------------------------------------


def qp_kernel_matrix(
    t1: np.ndarray,
    t2: np.ndarray,
    amplitude: float,
    length_scale: float,
    gamma: float,
    period: float,
) -> np.ndarray:
    dt = np.subtract.outer(np.asarray(t1, float), np.asarray(t2, float))
    exponent = -0.5 * (dt / length_scale) ** 2
    exponent -= 0.5 * gamma * np.sin(np.pi * dt / period) ** 2
    return amplitude * np.exp(exponent)


class RotationGPModel:
    """Exact QP-GP likelihood evaluated as a sum over independent chunks.

    Optimization coordinates are:
        [ln A, ln(l/P), ln Gamma, ln jitter, ln P]
    with ln(l/P) >= 0. This enforces the paper's l >= P constraint without a
    discontinuous optimizer boundary.
    """

    def __init__(
        self,
        lc: LightCurveData,
        settings: FitSettings,
        mixture: np.ndarray,
    ):
        self.lc = lc
        self.settings = settings
        self.mixture = mixture
        self.rng = np.random.default_rng(settings.random_seed)
        self.baseline = float(lc.time[-1] - lc.time[0])
        self.y_scale = max(robust_sigma(lc.flux), 1e-8)
        self.noise_scale = max(
            float(np.median(lc.flux_err)),
            robust_sigma(np.diff(lc.flux)) / math.sqrt(2.0),
            self.y_scale * 1e-3,
        )

        n_select = min(settings.gp_points, lc.time.size)
        if n_select < lc.time.size:
            inds = np.sort(self.rng.choice(lc.time.size, n_select, replace=False))
        else:
            inds = np.arange(lc.time.size)
        self.fit_time = lc.time[inds]
        self.fit_flux = lc.flux[inds]
        self.fit_flux_err = lc.flux_err[inds]

        chunk_size = min(settings.chunk_size, n_select)
        self.chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for start in range(0, n_select, chunk_size):
            stop = min(start + chunk_size, n_select)
            if stop - start >= 10:
                self.chunks.append(
                    (
                        self.fit_time[start:stop],
                        self.fit_flux[start:stop],
                        self.fit_flux_err[start:stop],
                    )
                )
        if not self.chunks:
            raise ValueError("No valid GP chunks were created.")

        self._set_priors_and_bounds()

    def _set_priors_and_bounds(self) -> None:
        s = self.settings
        p_center = (
            float(np.exp(self.mixture[np.argmax(self.mixture[:, 0]), 1]))
            if self.mixture.size
            else math.sqrt(s.pmin * s.pmax)
        )

        if s.use_paper_hyperpriors:
            # Table 2 of Angus et al. (2017), in natural-log parameters.
            self.prior_mu = np.array([-13.0, 7.2, -2.3, -17.0])
            self.prior_sigma = np.array([5.7, 1.2, 1.4, 5.0])
            lnA_bounds = (-20.0, 0.0)
            l_bounds = (math.exp(2.0), math.exp(20.0))
            lnGamma_bounds = (-10.0, 3.0)
            lnjitter_bounds = (-20.0, 0.0)
        else:
            variance = self.y_scale**2
            self.prior_mu = np.array(
                [
                    math.log(max(variance, 1e-16)),
                    math.log(max(3.0 * p_center, self.baseline * 0.5, s.pmin)),
                    -2.3,
                    math.log(max(self.noise_scale, 1e-12)),
                ]
            )
            self.prior_sigma = np.array([2.5, 1.5, 1.4, 2.5])
            lnA_bounds = (
                math.log(max(variance * 1e-4, 1e-20)),
                math.log(max(variance * 1e4, 1e-10)),
            )
            l_bounds = (
                max(s.pmin, 2.0 * np.median(np.diff(self.lc.time))),
                max(10.0 * self.baseline, 5.0 * s.pmax),
            )
            lnGamma_bounds = (-10.0, 3.0)
            lnjitter_bounds = (
                math.log(max(self.noise_scale * 1e-3, 1e-14)),
                math.log(max(self.y_scale * 10.0, self.noise_scale * 100.0)),
            )

        self.ln_l_min = math.log(l_bounds[0])
        self.ln_l_max = math.log(l_bounds[1])
        max_eta = max(0.1, self.ln_l_max - math.log(s.pmin))
        self.bounds = [
            lnA_bounds,
            (0.0, max_eta),
            lnGamma_bounds,
            lnjitter_bounds,
            (math.log(s.pmin), math.log(s.pmax)),
        ]

    @staticmethod
    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
        lnA, ln_l_over_p, lnGamma, lnjitter, lnP = np.asarray(theta, float)
        period = math.exp(lnP)
        length_scale = period * math.exp(ln_l_over_p)
        return (
            math.exp(lnA),
            length_scale,
            math.exp(lnGamma),
            math.exp(lnjitter),
            period,
        )

    def actual_log_parameters(self, theta: np.ndarray) -> np.ndarray:
        lnA, eta, lnGamma, lnjitter, lnP = np.asarray(theta, float)
        return np.array([lnA, lnP + eta, lnGamma, lnjitter, lnP])

    def log_prior(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, float)
        if theta.shape != (5,) or not np.all(np.isfinite(theta)):
            return -np.inf
        for value, (lo, hi) in zip(theta, self.bounds):
            if value < lo or value > hi:
                return -np.inf

        actual = self.actual_log_parameters(theta)
        lnA, lnl, lnGamma, lnjitter, lnP = actual
        if not (self.ln_l_min <= lnl <= self.ln_l_max):
            return -np.inf
        if lnl < lnP:  # paper constraint: spot-evolution scale >= period
            return -np.inf

        nuisance = np.array([lnA, lnl, lnGamma, lnjitter])
        lp = float(np.sum(normal_logpdf(nuisance, self.prior_mu, self.prior_sigma)))
        lp += period_prior_logpdf(lnP, self.mixture)
        return lp

    def _chunk_log_likelihood(
        self,
        theta: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
    ) -> float:
        amplitude, length_scale, gamma, jitter, period = self.unpack(theta)
        kernel = qp_kernel_matrix(t, t, amplitude, length_scale, gamma, period)
        diag = yerr**2 + jitter**2 + max(amplitude, 1e-20) * 1e-10
        kernel[np.diag_indices_from(kernel)] += diag
        try:
            factor, lower = cho_factor(kernel, lower=True, check_finite=False)
            alpha = cho_solve((factor, lower), y, check_finite=False)
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
        logdet = 2.0 * np.sum(np.log(np.diag(factor)))
        value = -0.5 * (
            float(y @ alpha) + logdet + y.size * math.log(2.0 * math.pi)
        )
        return value if np.isfinite(value) else -np.inf

    def log_likelihood(self, theta: np.ndarray) -> float:
        total = 0.0
        for t, y, yerr in self.chunks:
            value = self._chunk_log_likelihood(theta, t, y, yerr)
            if not np.isfinite(value):
                return -np.inf
            total += value
        return float(total)

    def log_posterior(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    def sample_prior(self) -> np.ndarray:
        for _ in range(10000):
            lnP = sample_period_prior(
                self.rng, self.mixture, self.settings.pmin, self.settings.pmax
            )
            lnA = float(self.rng.normal(self.prior_mu[0], self.prior_sigma[0]))
            lnl = float(self.rng.normal(self.prior_mu[1], self.prior_sigma[1]))
            lnGamma = float(self.rng.normal(self.prior_mu[2], self.prior_sigma[2]))
            lnjitter = float(self.rng.normal(self.prior_mu[3], self.prior_sigma[3]))
            lnl = max(lnl, lnP + 1e-4)
            theta = np.array([lnA, lnl - lnP, lnGamma, lnjitter, lnP])
            if np.isfinite(self.log_prior(theta)):
                return theta
        # Deterministic fallback at clipped prior centers.
        lnP = math.log(math.sqrt(self.settings.pmin * self.settings.pmax))
        lnl = max(self.prior_mu[1], lnP + 0.1)
        theta = np.array(
            [self.prior_mu[0], lnl - lnP, self.prior_mu[2], self.prior_mu[3], lnP]
        )
        return np.array(
            [np.clip(v, lo + 1e-8, hi - 1e-8) for v, (lo, hi) in zip(theta, self.bounds)]
        )

    def deterministic_starts(self) -> list[np.ndarray]:
        starts: list[np.ndarray] = []
        if self.mixture.size:
            order = np.argsort(self.mixture[:, 0])[::-1]
            unique_periods: list[float] = []
            for idx in order:
                lnP = float(self.mixture[idx, 1])
                if any(abs(lnP - prev) < 0.03 for prev in unique_periods):
                    continue
                unique_periods.append(lnP)
                lnl = max(self.prior_mu[1], lnP + 0.1)
                starts.append(
                    np.array(
                        [
                            self.prior_mu[0],
                            lnl - lnP,
                            self.prior_mu[2],
                            self.prior_mu[3],
                            lnP,
                        ]
                    )
                )
                if len(starts) >= 6:
                    break
        return starts

    def fit_map(
        self,
        progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[np.ndarray, float]:
        starts = self.deterministic_starts()
        while len(starts) < self.settings.optimizer_starts:
            starts.append(self.sample_prior())
        starts = starts[: self.settings.optimizer_starts]

        best_theta: Optional[np.ndarray] = None
        best_logpost = -np.inf

        def objective(theta):
            value = self.log_posterior(theta)
            return 1e100 if not np.isfinite(value) else -value

        for index, start in enumerate(starts, start=1):
            if progress:
                progress(f"GP optimizer start {index}/{len(starts)}")
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=self.bounds,
                options={"maxiter": 400, "ftol": 1e-9, "maxls": 30},
            )
            theta = np.asarray(result.x, float)
            logpost = self.log_posterior(theta)
            if np.isfinite(logpost) and logpost > best_logpost:
                best_logpost = float(logpost)
                best_theta = theta

        if best_theta is None:
            raise RuntimeError("Every GP optimizer start failed.")
        return best_theta, best_logpost

    def run_mcmc(
        self,
        map_theta: np.ndarray,
        progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[np.ndarray, float]:
        try:
            import emcee  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "MCMC refinement requires emcee: pip install emcee"
            ) from exc

        ndim = 5
        nwalkers = max(self.settings.mcmc_walkers, 2 * ndim + 2)
        positions = np.empty((nwalkers, ndim), dtype=float)
        local_count = nwalkers // 2
        scales = np.array([0.08, 0.08, 0.08, 0.08, 0.03])
        for i in range(local_count):
            for _ in range(1000):
                trial = map_theta + self.rng.normal(size=ndim) * scales
                if np.isfinite(self.log_prior(trial)):
                    positions[i] = trial
                    break
            else:
                positions[i] = self.sample_prior()
        for i in range(local_count, nwalkers):
            positions[i] = self.sample_prior()

        try:
            moves = [
                (emcee.moves.KDEMove(), 0.4),
                (emcee.moves.DEMove(), 0.4),
                (emcee.moves.DESnookerMove(), 0.2),
            ]
        except Exception:
            moves = [
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ]

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.log_posterior,
            moves=moves,
        )
        if progress:
            progress(
                f"MCMC: {nwalkers} walkers × {self.settings.mcmc_steps} steps"
            )
        sampler.run_mcmc(positions, self.settings.mcmc_steps, progress=False)
        chain = sampler.get_chain(discard=self.settings.mcmc_burn, flat=True)
        logprob = sampler.get_log_prob(discard=self.settings.mcmc_burn, flat=True)
        good = np.all(np.isfinite(chain), axis=1) & np.isfinite(logprob)
        chain = chain[good]
        if chain.shape[0] < 50:
            raise RuntimeError("Too few finite MCMC samples survived burn-in.")
        return chain, float(np.mean(sampler.acceptance_fraction))

    def conditional_period_scan(
        self,
        map_theta: np.ndarray,
        n_points: Optional[int] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_points = n_points or self.settings.posterior_scan_points
        ln_periods = np.linspace(
            math.log(self.settings.pmin), math.log(self.settings.pmax), n_points
        )
        values = np.full(n_points, -np.inf)
        map_lnl = map_theta[4] + map_theta[1]
        for i, lnP in enumerate(ln_periods):
            theta = map_theta.copy()
            theta[4] = lnP
            theta[1] = max(map_lnl - lnP, 1e-6)
            values[i] = self.log_posterior(theta)
        finite = np.isfinite(values)
        if finite.any():
            values[finite] -= np.max(values[finite])
        if progress:
            progress("Computed conditional period scan")
        return np.exp(ln_periods), values

    def predict(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_train = min(self.settings.prediction_train_points, self.lc.time.size)
        if n_train < self.lc.time.size:
            # Evenly spaced indices are visually more stable than a random draw.
            train_idx = np.linspace(0, self.lc.time.size - 1, n_train).astype(int)
        else:
            train_idx = np.arange(self.lc.time.size)
        t_train = self.lc.time[train_idx]
        y_train = self.lc.flux[train_idx]
        e_train = self.lc.flux_err[train_idx]
        t_pred = np.linspace(
            self.lc.time[0], self.lc.time[-1], self.settings.prediction_points
        )

        amplitude, length_scale, gamma, jitter, period = self.unpack(theta)
        K = qp_kernel_matrix(
            t_train, t_train, amplitude, length_scale, gamma, period
        )
        K[np.diag_indices_from(K)] += e_train**2 + jitter**2 + amplitude * 1e-10
        factor, lower = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve((factor, lower), y_train, check_finite=False)
        Kstar = qp_kernel_matrix(
            t_pred, t_train, amplitude, length_scale, gamma, period
        )
        mean = Kstar @ alpha
        # Latent predictive variance; diag(K**)=A for this stationary kernel.
        v = solve_triangular(factor, Kstar.T, lower=lower, check_finite=False)
        variance = np.maximum(amplitude - np.sum(v**2, axis=0), 0.0)
        return t_pred, mean, np.sqrt(variance)




def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-8, 1.0 - 1e-8))
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _load_celerite2():
    """Import celerite2 lazily so the Angus backend works without it installed."""
    try:
        import celerite2  # type: ignore
        from celerite2 import terms  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The exoplanet/celerite2 GP backend requires celerite2. Install it with "
            "`python -m pip install -U celerite2` (or `exoplanet[extras]`)."
        ) from exc
    return celerite2, terms


class CeleriteRotationGPModel:
    """Modern scalable stellar-rotation GP using celerite2.

    The exoplanet project now delegates its scalable GP functionality to
    celerite2, so GHOST uses the celerite2 Python interface directly.  The
    period prior is deliberately the same harmonic-aware prior used by the
    Angus/GProtation branch so backend comparisons are meaningful.

    Kernel choices:
      * RotationTerm: two damped SHO modes at approximately P and P/2.
      * RotationTerm + Matern32: adds slow aperiodic baseline variability.
      * Double RotationTerm: experimental second independent rotation period.
    """

    def __init__(self, lc: LightCurveData, settings: FitSettings, mixture: np.ndarray):
        self.lc = lc
        self.settings = settings
        self.mixture = mixture
        self.kernel_name = settings.celerite_kernel
        self.rng = np.random.default_rng(settings.random_seed)
        self.baseline = float(lc.time[-1] - lc.time[0])
        self.y_scale = max(robust_sigma(lc.flux), 1e-8)
        self.noise_scale = max(
            float(np.nanmedian(lc.flux_err)),
            robust_sigma(np.diff(lc.flux)) / math.sqrt(2.0),
            self.y_scale * 1e-3,
        )
        p_center = (
            float(np.exp(mixture[np.argmax(mixture[:, 0]), 1]))
            if mixture.size else math.sqrt(settings.pmin * settings.pmax)
        )
        self.p_center = float(np.clip(p_center, settings.pmin, settings.pmax))

        # celerite scales linearly, so unlike the Angus exact covariance branch
        # we can retain far more points and do not need independent chunks.
        nmax = int(settings.celerite_max_points)
        if nmax > 0 and lc.time.size > nmax:
            inds = np.linspace(0, lc.time.size - 1, nmax).astype(int)
        else:
            inds = np.arange(lc.time.size)
        self.fit_time = np.asarray(lc.time[inds], float)
        self.fit_flux = np.asarray(lc.flux[inds], float)
        self.fit_flux_err = np.asarray(lc.flux_err[inds], float)
        self._set_parameterization()

    def _set_parameterization(self) -> None:
        s = self.settings
        y = self.y_scale
        n = self.noise_scale
        # Common parameter centers in transformed coordinates.
        self.period_index: int
        if self.kernel_name == "RotationTerm":
            self.parameter_names = [
                "ln_sigma", "ln_Q0", "ln_dQ", "logit_f", "ln_jitter", "ln_period"
            ]
            self.period_index = 5
            self.prior_mu = np.array([math.log(y), 0.0, 0.0, 0.0, math.log(n), math.log(self.p_center)])
            self.prior_sigma = np.array([2.0, 1.5, 1.5, 1.5, 2.5, 1.0])
            self.bounds = [
                (math.log(max(y * 1e-3, 1e-12)), math.log(max(y * 1e3, 1e-8))),
                (math.log(1e-2), math.log(1e2)),
                (math.log(1e-2), math.log(1e2)),
                (_logit(0.01), _logit(0.99)),
                (math.log(max(n * 1e-3, 1e-14)), math.log(max(y * 10.0, n * 100.0))),
                (math.log(s.pmin), math.log(s.pmax)),
            ]
        elif self.kernel_name == "RotationTerm + Matern32":
            base_rho = max(2.0 * self.p_center, 0.5 * self.baseline, s.pmin)
            self.parameter_names = [
                "ln_sigma", "ln_Q0", "ln_dQ", "logit_f", "ln_jitter",
                "ln_baseline_sigma", "ln_baseline_rho", "ln_period"
            ]
            self.period_index = 7
            self.prior_mu = np.array([
                math.log(y), 0.0, 0.0, 0.0, math.log(n),
                math.log(max(y * 0.5, 1e-12)), math.log(base_rho), math.log(self.p_center),
            ])
            self.prior_sigma = np.array([2.0, 1.5, 1.5, 1.5, 2.5, 2.5, 1.5, 1.0])
            self.bounds = [
                (math.log(max(y * 1e-3, 1e-12)), math.log(max(y * 1e3, 1e-8))),
                (math.log(1e-2), math.log(1e2)),
                (math.log(1e-2), math.log(1e2)),
                (_logit(0.01), _logit(0.99)),
                (math.log(max(n * 1e-3, 1e-14)), math.log(max(y * 10.0, n * 100.0))),
                (math.log(max(y * 1e-4, 1e-13)), math.log(max(y * 1e2, 1e-8))),
                (math.log(max(2.0 * np.median(np.diff(self.fit_time)), s.pmin * 0.25)),
                 math.log(max(20.0 * self.baseline, 5.0 * s.pmax))),
                (math.log(s.pmin), math.log(s.pmax)),
            ]
        elif self.kernel_name == "Double RotationTerm (experimental)":
            self.parameter_names = [
                "ln_sigma_primary", "ln_sigma_secondary", "ln_Q0", "ln_dQ",
                "logit_f", "ln_jitter", "ln_period", "ln_period_ratio_secondary"
            ]
            self.period_index = 6
            self.prior_mu = np.array([
                math.log(y), math.log(max(y * 0.5, 1e-12)), 0.0, 0.0, 0.0,
                math.log(n), math.log(self.p_center), 0.0,
            ])
            self.prior_sigma = np.array([2.0, 2.5, 1.5, 1.5, 1.5, 2.5, 1.0, 0.35])
            self.bounds = [
                (math.log(max(y * 1e-3, 1e-12)), math.log(max(y * 1e3, 1e-8))),
                (math.log(max(y * 1e-4, 1e-13)), math.log(max(y * 1e2, 1e-8))),
                (math.log(1e-2), math.log(1e2)),
                (math.log(1e-2), math.log(1e2)),
                (_logit(0.01), _logit(0.99)),
                (math.log(max(n * 1e-3, 1e-14)), math.log(max(y * 10.0, n * 100.0))),
                (math.log(s.pmin), math.log(s.pmax)),
                (math.log(0.5), math.log(2.0)),
            ]
        else:
            raise ValueError(f"Unsupported celerite2 kernel: {self.kernel_name}")

    def unpack(self, theta: np.ndarray) -> dict[str, float]:
        theta = np.asarray(theta, float)
        if self.kernel_name == "RotationTerm":
            ln_sigma, lnQ0, lndQ, logit_f, lnj, lnP = theta
            return {
                "sigma_rotation": math.exp(ln_sigma), "Q0": math.exp(lnQ0),
                "dQ": math.exp(lndQ), "f": _sigmoid(logit_f),
                "jitter_relative_flux": math.exp(lnj), "period_days": math.exp(lnP),
            }
        if self.kernel_name == "RotationTerm + Matern32":
            ln_sigma, lnQ0, lndQ, logit_f, lnj, lnbs, lnbr, lnP = theta
            return {
                "sigma_rotation": math.exp(ln_sigma), "Q0": math.exp(lnQ0),
                "dQ": math.exp(lndQ), "f": _sigmoid(logit_f),
                "jitter_relative_flux": math.exp(lnj),
                "baseline_sigma": math.exp(lnbs), "baseline_rho_days": math.exp(lnbr),
                "period_days": math.exp(lnP),
            }
        lns1, lns2, lnQ0, lndQ, logit_f, lnj, lnP, lnr = theta
        period = math.exp(lnP)
        return {
            "sigma_rotation_primary": math.exp(lns1),
            "sigma_rotation_secondary": math.exp(lns2),
            "Q0": math.exp(lnQ0), "dQ": math.exp(lndQ), "f": _sigmoid(logit_f),
            "jitter_relative_flux": math.exp(lnj), "period_days": period,
            "secondary_period_days": period * math.exp(lnr),
            "secondary_period_ratio": math.exp(lnr),
        }

    def _build_kernel(self, theta: np.ndarray):
        _, terms = _load_celerite2()
        p = self.unpack(theta)
        if self.kernel_name == "RotationTerm":
            return terms.RotationTerm(
                sigma=p["sigma_rotation"], period=p["period_days"],
                Q0=p["Q0"], dQ=p["dQ"], f=p["f"],
            )
        if self.kernel_name == "RotationTerm + Matern32":
            rotation = terms.RotationTerm(
                sigma=p["sigma_rotation"], period=p["period_days"],
                Q0=p["Q0"], dQ=p["dQ"], f=p["f"],
            )
            baseline = terms.Matern32Term(
                sigma=p["baseline_sigma"], rho=p["baseline_rho_days"]
            )
            return rotation + baseline
        rotation1 = terms.RotationTerm(
            sigma=p["sigma_rotation_primary"], period=p["period_days"],
            Q0=p["Q0"], dQ=p["dQ"], f=p["f"],
        )
        rotation2 = terms.RotationTerm(
            sigma=p["sigma_rotation_secondary"], period=p["secondary_period_days"],
            Q0=p["Q0"], dQ=p["dQ"], f=p["f"],
        )
        return rotation1 + rotation2

    def log_prior(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, float)
        if theta.shape != (len(self.bounds),) or not np.all(np.isfinite(theta)):
            return -np.inf
        for value, (lo, hi) in zip(theta, self.bounds):
            if value < lo or value > hi:
                return -np.inf
        params = self.unpack(theta)
        if self.kernel_name == "Double RotationTerm (experimental)":
            p2 = params["secondary_period_days"]
            if p2 < self.settings.pmin or p2 > self.settings.pmax:
                return -np.inf
        # Gaussian nuisance priors in transformed coordinates, but replace the
        # period coordinate with the same ACF/BGLS-informed mixture as Angus.
        mask = np.ones(theta.size, dtype=bool)
        mask[self.period_index] = False
        lp = float(np.sum(normal_logpdf(theta[mask], self.prior_mu[mask], self.prior_sigma[mask])))
        lp += period_prior_logpdf(theta[self.period_index], self.mixture)
        return lp

    def log_likelihood(self, theta: np.ndarray) -> float:
        celerite2, _ = _load_celerite2()
        try:
            params = self.unpack(theta)
            kernel = self._build_kernel(theta)
            gp = celerite2.GaussianProcess(kernel, mean=0.0)
            diag = self.fit_flux_err**2 + params["jitter_relative_flux"]**2
            gp.compute(self.fit_time, diag=diag, quiet=True)
            value = float(gp.log_likelihood(self.fit_flux))
            return value if np.isfinite(value) else -np.inf
        except Exception:
            return -np.inf

    def log_posterior(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    def sample_prior(self) -> np.ndarray:
        for _ in range(10000):
            theta = self.rng.normal(self.prior_mu, self.prior_sigma)
            theta[self.period_index] = sample_period_prior(
                self.rng, self.mixture, self.settings.pmin, self.settings.pmax
            )
            theta = np.array([
                np.clip(value, lo + 1e-8, hi - 1e-8)
                for value, (lo, hi) in zip(theta, self.bounds)
            ])
            if np.isfinite(self.log_prior(theta)):
                return theta
        theta = np.array([
            np.clip(value, lo + 1e-8, hi - 1e-8)
            for value, (lo, hi) in zip(self.prior_mu, self.bounds)
        ])
        theta[self.period_index] = float(np.clip(
            math.log(self.p_center), self.bounds[self.period_index][0], self.bounds[self.period_index][1]
        ))
        return theta

    def deterministic_starts(self) -> list[np.ndarray]:
        starts: list[np.ndarray] = []
        if self.mixture.size:
            order = np.argsort(self.mixture[:, 0])[::-1]
            seen: list[float] = []
            for idx in order:
                lnP = float(self.mixture[idx, 1])
                if any(abs(lnP - prev) < 0.03 for prev in seen):
                    continue
                seen.append(lnP)
                theta = self.prior_mu.copy()
                theta[self.period_index] = lnP
                theta = np.array([
                    np.clip(value, lo + 1e-8, hi - 1e-8)
                    for value, (lo, hi) in zip(theta, self.bounds)
                ])
                if np.isfinite(self.log_prior(theta)):
                    starts.append(theta)
                if len(starts) >= 6:
                    break
        return starts

    def fit_map(self, progress: Optional[Callable[[str], None]] = None) -> tuple[np.ndarray, float]:
        starts = self.deterministic_starts()
        while len(starts) < self.settings.optimizer_starts:
            starts.append(self.sample_prior())
        starts = starts[: self.settings.optimizer_starts]
        best_theta = None
        best_logpost = -np.inf

        def objective(theta):
            value = self.log_posterior(theta)
            return 1e100 if not np.isfinite(value) else -value

        for index, start in enumerate(starts, start=1):
            if progress:
                progress(
                    f"celerite2 {self.kernel_name}: optimizer start {index}/{len(starts)}"
                )
            soln = minimize(
                objective, start, method="L-BFGS-B", bounds=self.bounds,
                options={"maxiter": 500, "ftol": 1e-9, "maxls": 40},
            )
            theta = np.asarray(soln.x, float)
            logpost = self.log_posterior(theta)
            if np.isfinite(logpost) and logpost > best_logpost:
                best_theta, best_logpost = theta, float(logpost)
        if best_theta is None:
            raise RuntimeError(
                "Every celerite2 optimizer start failed. Try another kernel, a broader "
                "period prior, or inspect the light curve for preprocessing problems."
            )
        return best_theta, best_logpost

    def run_mcmc(self, map_theta: np.ndarray, progress=None) -> tuple[np.ndarray, float]:
        try:
            import emcee  # type: ignore
        except ImportError as exc:
            raise ImportError("MCMC refinement requires emcee: pip install emcee") from exc
        ndim = map_theta.size
        nwalkers = max(self.settings.mcmc_walkers, 2 * ndim + 2)
        positions = np.empty((nwalkers, ndim), float)
        scales = np.full(ndim, 0.08)
        scales[self.period_index] = 0.03
        for i in range(nwalkers):
            if i < nwalkers // 2:
                for _ in range(1000):
                    trial = map_theta + self.rng.normal(size=ndim) * scales
                    if np.isfinite(self.log_prior(trial)):
                        positions[i] = trial
                        break
                else:
                    positions[i] = self.sample_prior()
            else:
                positions[i] = self.sample_prior()
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, moves=moves)
        if progress:
            progress(f"celerite2 MCMC: {nwalkers} walkers × {self.settings.mcmc_steps} steps")
        sampler.run_mcmc(positions, self.settings.mcmc_steps, progress=False)
        chain = sampler.get_chain(discard=self.settings.mcmc_burn, flat=True)
        logprob = sampler.get_log_prob(discard=self.settings.mcmc_burn, flat=True)
        good = np.all(np.isfinite(chain), axis=1) & np.isfinite(logprob)
        chain = chain[good]
        if chain.shape[0] < 50:
            raise RuntimeError("Too few finite celerite2 MCMC samples survived burn-in.")
        return chain, float(np.mean(sampler.acceptance_fraction))

    def conditional_period_scan(self, map_theta: np.ndarray, n_points=None, progress=None):
        n_points = n_points or self.settings.posterior_scan_points
        ln_periods = np.linspace(math.log(self.settings.pmin), math.log(self.settings.pmax), n_points)
        values = np.full(n_points, -np.inf)
        for i, lnP in enumerate(ln_periods):
            theta = map_theta.copy()
            theta[self.period_index] = lnP
            values[i] = self.log_posterior(theta)
        finite = np.isfinite(values)
        if finite.any():
            values[finite] -= np.max(values[finite])
        if progress:
            progress("Computed celerite2 conditional period scan")
        return np.exp(ln_periods), values

    def predict(self, theta: np.ndarray):
        celerite2, _ = _load_celerite2()
        params = self.unpack(theta)
        kernel = self._build_kernel(theta)
        gp = celerite2.GaussianProcess(kernel, mean=0.0)
        diag = self.fit_flux_err**2 + params["jitter_relative_flux"]**2
        gp.compute(self.fit_time, diag=diag, quiet=False)
        t_pred = np.linspace(self.lc.time[0], self.lc.time[-1], self.settings.prediction_points)
        mean, variance = gp.predict(self.fit_flux, t=t_pred, return_var=True)
        return np.asarray(t_pred), np.asarray(mean), np.sqrt(np.maximum(np.asarray(variance), 0.0))

    def map_hyperparameters(self, theta: np.ndarray) -> dict[str, float]:
        return {key: float(value) for key, value in self.unpack(theta).items()}


# -----------------------------------------------------------------------------
# End-to-end fitting and plotting
# -----------------------------------------------------------------------------


def phase_bin(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    nbins: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase = ((time - time[0]) / period) % 1.0
    order = np.argsort(phase)
    phase, flux = phase[order], flux[order]
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if mask.any():
            binned[i] = np.median(flux[mask])
    good = np.isfinite(binned)
    return phase, flux, centers[good], binned[good]


def fit_rotation_period(
    raw_lc: LightCurveData,
    settings: FitSettings,
    progress: Optional[Callable[[str], None]] = None,
    period_analysis: Optional[PeriodAnalysis] = None,
    gp_prior_source: str = "final_averaged_prot",
    gp_prior_override: Optional[float] = None,
) -> tuple[LightCurveData, FitResult]:
    """Fit the selected GP backend using a shared harmonic-aware period prior.

    GProtation reproduces the Angus et al. (2017) quasi-periodic covariance
    logic.  The exoplanet/celerite2 branch uses scalable celerite2 kernels but
    keeps the same ACF/BGLS-derived P, P/2, 2P prior so backend comparisons are
    not confounded by a different period prior.
    """
    if period_analysis is None:
        lc, period_analysis = calculate_period_analysis(
            raw_lc, settings, segment_selection="all", method_selection="both", progress=progress
        )
    else:
        if progress:
            progress("Cleaning and normalizing light curve for GP")
        lc = preprocess_lightcurve(raw_lc, settings)
        settings.validate(float(lc.time[-1] - lc.time[0]))

    if gp_prior_override is None:
        prior_center = period_analysis.prior_center(gp_prior_source)
    else:
        prior_center = float(gp_prior_override)
        if not np.isfinite(prior_center) or prior_center <= 0:
            raise ValueError(f"GP prior override for {gp_prior_source} is unavailable.")
    mixture = build_centered_period_mixture(
        prior_center, settings.pmin, settings.pmax, settings.acf_prior_width
    )
    acf_candidates = [
        measurement.acf_candidate
        for measurement in period_analysis.measurements.values()
        if measurement.acf_candidate is not None
    ]

    if settings.gp_backend == "exoplanet / celerite2":
        # Fail early with a useful dependency message before optimization starts.
        _load_celerite2()
        model = CeleriteRotationGPModel(lc, settings, mixture)
        backend_label = "exoplanet / celerite2"
        kernel_label = settings.celerite_kernel
        if progress:
            progress(
                f"Fitting {backend_label} GP ({kernel_label}) with "
                f"{gp_prior_source}={prior_center:.6g} d prior"
            )
    else:
        model = RotationGPModel(lc, settings, mixture)
        backend_label = "GProtation (Angus+17)"
        kernel_label = "Angus17 quasi-periodic"
        if progress:
            progress(
                f"Fitting Angus+17 QP-GP with {gp_prior_source}={prior_center:.6g} d prior"
            )

    map_theta, map_logpost = model.fit_map(progress=progress)
    if isinstance(model, CeleriteRotationGPModel):
        map_hyper = model.map_hyperparameters(map_theta)
        map_period = float(map_hyper["period_days"])
        period_index = model.period_index
    else:
        amplitude, length_scale, gamma, jitter, map_period = model.unpack(map_theta)
        map_hyper = {
            "A_covariance": float(amplitude),
            "length_scale_days": float(length_scale),
            "Gamma": float(gamma),
            "jitter_relative_flux": float(jitter),
            "period_days": float(map_period),
        }
        period_index = 4

    posterior_samples = None
    acceptance = None
    if settings.run_mcmc:
        chain, acceptance = model.run_mcmc(map_theta, progress=progress)
        posterior_samples = np.exp(chain[:, period_index])
        p16, p50, p84 = weighted_quantile(posterior_samples, [0.16, 0.50, 0.84])
    else:
        p16 = p50 = p84 = map_period

    scan_periods, scan_logp = model.conditional_period_scan(map_theta, progress=progress)
    if not settings.run_mcmc:
        finite = np.isfinite(scan_logp)
        if finite.any():
            weights = np.zeros_like(scan_logp)
            weights[finite] = np.exp(scan_logp[finite] - np.max(scan_logp[finite]))
            if weights.sum() > 0:
                weights /= weights.sum()
                cdf = np.cumsum(weights)
                p16 = np.interp(0.16, cdf, scan_periods)
                p50 = np.interp(0.50, cdf, scan_periods)
                p84 = np.interp(0.84, cdf, scan_periods)

    if progress:
        progress("Calculating GP predictive curve")
    gp_time, gp_mean, gp_std = model.predict(map_theta)
    phase, phase_flux, phase_bin_x, phase_bin_y = phase_bin(lc.time, lc.flux, map_period)

    n_chunks = len(model.chunks) if hasattr(model, "chunks") else 1
    result = FitResult(
        map_period=float(map_period),
        map_theta=np.asarray(map_theta),
        map_log_posterior=float(map_logpost),
        period_median=float(p50),
        period_p16=float(p16),
        period_p84=float(p84),
        acf_candidates=list(acf_candidates),
        period_mixture=mixture,
        posterior_period_samples=posterior_samples,
        scan_periods=scan_periods,
        scan_log_probability=scan_logp,
        gp_time=gp_time,
        gp_mean=gp_mean,
        gp_std=gp_std,
        phase=phase,
        phase_flux=phase_flux,
        phase_bin_x=phase_bin_x,
        phase_bin_y=phase_bin_y,
        fit_time=model.fit_time,
        fit_flux=model.fit_flux,
        fit_flux_err=model.fit_flux_err,
        n_input=int(raw_lc.time.size),
        n_clean=int(lc.time.size),
        n_gp=int(model.fit_time.size),
        n_chunks=int(n_chunks),
        period_analysis=period_analysis,
        gp_prior_source=gp_prior_source,
        gp_prior_center=float(prior_center),
        gp_backend=backend_label,
        gp_kernel=kernel_label,
        map_hyperparameters=map_hyper,
        acceptance_fraction=acceptance,
    )
    if progress:
        secondary = map_hyper.get("secondary_period_days")
        suffix = f"; secondary={secondary:.6g} d" if secondary is not None else ""
        progress(
            f"Done: {backend_label} GP MAP period = {map_period:.6g} d{suffix}"
        )
    return lc, result


def period_prior_curve(
    periods: np.ndarray,
    mixture: np.ndarray,
    pmin: float,
    pmax: float,
) -> np.ndarray:
    lnP = np.log(periods)
    if mixture.size == 0:
        curve = np.ones_like(periods)
    else:
        curve = np.exp(
            np.array([period_prior_logpdf(v, mixture) for v in lnP])
            - max(period_prior_logpdf(v, mixture) for v in lnP)
        )
    curve[(periods < pmin) | (periods > pmax)] = 0.0
    return curve


def _period_text(value: float) -> str:
    return f"{value:.6g} d" if np.isfinite(value) else "—"


def draw_period_analysis_figure(fig, lc, analysis):
    """Overview; detailed peaks and alternative folds have a dedicated GUI tab."""
    fig.clear();axes=fig.subplots(2,2);ax_lc,ax_phase,ax_spectrum,ax_compare=axes.ravel()
    ax_lc.plot(lc.time,lc.flux,".",ms=2,alpha=.5)
    ax_lc.axvline(np.median(lc.time),ls="--",lw=1,color="gray")
    ax_lc.set(xlabel="Time [days]",ylabel="Cleaned flux",title=lc.name)
    period=analysis.final_averaged_prot
    if np.isfinite(period) and period>0:
        phase,flux,px,py=phase_bin(lc.time,lc.flux,period)
        for off in (0,1):
            ax_phase.plot(phase+off,flux,".",ms=2,alpha=.3)
            ax_phase.plot(px+off,py,"o-",ms=3,color="C3")
    ax_phase.set(xlabel="Phase (two cycles)",ylabel="Cleaned flux",
                 title=f"Candidate {analysis.aggregation}: {_period_text(period)}")
    for method in analysis.requested_methods:
        measurements=[analysis.measurement(seg,method) for seg in SEGMENT_ORDER]
        values=[m.period if m else np.nan for m in measurements]
        ax_compare.plot(np.arange(3),values,"o-",label=method)
        m=next((m for m in measurements if m is not None and m.periods.size),None)
        if m is not None:
            order=np.argsort(m.periods);power=m.power.copy();finite=np.isfinite(power)
            if np.any(finite):
                low,high=np.min(power[finite]),np.max(power[finite])
                power=(power-low)/(high-low) if high>low else power*0
            ax_spectrum.plot(m.periods[order],power[order],lw=1,label=f"{method}: {m.segment}")
    ax_spectrum.set(xlabel="Period / ACF lag [days]",ylabel="Scaled to 0–1 for display only",
                    title="First available segment per method; inspect Peaks & phase")
    if ax_spectrum.lines:ax_spectrum.legend(fontsize=7)
    ax_compare.set_xticks([0,1,2],["Full","Half A","Half B"])
    ax_compare.set(ylabel="Period [days]",title="Method / segment comparison")
    ax_compare.legend(fontsize=8)
    fig.tight_layout()


def draw_fit_figure(
    fig: Figure,
    lc: LightCurveData,
    result: FitResult,
    settings: FitSettings,
) -> None:
    fig.clear()
    axes = fig.subplots(2, 2)
    ax_lc, ax_phase = axes[0]
    ax_compare, ax_post = axes[1]
    analysis = result.period_analysis

    ax_lc.plot(lc.time, lc.flux, ".", ms=2, alpha=0.45, label="cleaned LC")
    ax_lc.plot(result.gp_time, result.gp_mean, lw=1.5, label="QP-GP mean")
    ax_lc.fill_between(
        result.gp_time,
        result.gp_mean - result.gp_std,
        result.gp_mean + result.gp_std,
        alpha=0.18,
        linewidth=0,
        label="latent 1σ",
    )
    n_markers = int((lc.time[-1] - lc.time[0]) / result.map_period) + 1
    if n_markers <= 80:
        for tmark in lc.time[0] + np.arange(n_markers + 1) * result.map_period:
            ax_lc.axvline(tmark, lw=0.6, ls=":", alpha=0.25)
    ax_lc.set_xlabel("Time [days]")
    ax_lc.set_ylabel("Relative flux")
    ax_lc.set_title(
        f"{lc.name}\nGP $P_{{rot}}$={result.map_period:.5g} d; "
        f"{result.gp_backend} / {result.gp_kernel}; "
        f"prior={result.gp_prior_source} ({result.gp_prior_center:.5g} d)"
    )
    ax_lc.legend(loc="best", fontsize=8)

    ax_phase.plot(result.phase, result.phase_flux, ".", ms=2, alpha=0.25)
    ax_phase.plot(result.phase_bin_x, result.phase_bin_y, "o-", ms=3, lw=1.5)
    ax_phase.set_xlabel("Phase")
    ax_phase.set_ylabel("Relative flux")
    ax_phase.set_title(
        f"Phase folded at {result.map_period:.5g} d\n"
        f"median={result.period_median:.5g} "
        f"(-{result.period_minus:.2g}/+{result.period_plus:.2g}) d"
    )

    x = np.arange(len(SEGMENT_ORDER))
    for method, marker in (("ACF", "o"), ("BGLS", "s")):
        values = []
        for segment in SEGMENT_ORDER:
            measurement = analysis.measurement(segment, method)
            values.append(measurement.period if measurement is not None else np.nan)
        ax_compare.plot(x, values, marker + "-", label=method)
    if np.isfinite(analysis.acf_combined):
        ax_compare.axhline(analysis.acf_combined, ls="--", lw=1.0, label="ACF combined")
    if np.isfinite(analysis.bgls_combined):
        ax_compare.axhline(analysis.bgls_combined, ls="-.", lw=1.0, label="BGLS combined")
    if np.isfinite(analysis.final_averaged_prot):
        ax_compare.axhline(
            analysis.final_averaged_prot, ls=":", lw=1.3, label="final averaged"
        )
    ax_compare.axhline(result.map_period, lw=1.5, label="GP MAP")
    ax_compare.set_xticks(x, ["Full", "Half A", "Half B"])
    ax_compare.set_ylabel("Period [days]")
    ax_compare.set_title("Period-method consistency")
    ax_compare.legend(loc="best", fontsize=7, ncol=2)

    period_grid = np.geomspace(settings.pmin, settings.pmax, 1000)
    prior = period_prior_curve(
        period_grid, result.period_mixture, settings.pmin, settings.pmax
    )
    ax_post.plot(period_grid, prior, lw=1.3, label=f"{result.gp_prior_source} prior")
    if result.posterior_period_samples is not None:
        counts, edges = np.histogram(
            result.posterior_period_samples,
            bins=60,
            range=(settings.pmin, settings.pmax),
            density=True,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        if np.nanmax(counts) > 0:
            counts = counts / np.nanmax(counts)
        ax_post.plot(centers, counts, lw=1.5, label="MCMC posterior")
    else:
        conditional = np.zeros_like(result.scan_log_probability)
        finite = np.isfinite(result.scan_log_probability)
        conditional[finite] = np.exp(result.scan_log_probability[finite])
        if conditional.max() > 0:
            conditional /= conditional.max()
        ax_post.plot(
            result.scan_periods,
            conditional,
            lw=1.5,
            label="conditional posterior scan",
        )
    ax_post.axvline(result.map_period, ls=":", lw=1.3, label="GP MAP")
    ax_post.set_xscale("log")
    ax_post.set_xlim(settings.pmin, settings.pmax)
    ax_post.set_ylim(bottom=0)
    ax_post.set_xlabel("Period [days]")
    ax_post.set_ylabel("Relative density")
    ax_post.set_title("Selected prior and GP posterior")
    ax_post.legend(loc="best", fontsize=8)

    footer = (
        f"ACF combined={_period_text(analysis.acf_combined)}; "
        f"BGLS combined={_period_text(analysis.bgls_combined)}; "
        f"final average={_period_text(analysis.final_averaged_prot)} | "
        f"N input/clean/GP={result.n_input}/{result.n_clean}/{result.n_gp}; "
        f"GP={result.gp_backend} / {result.gp_kernel}; chunks={result.n_chunks}"
    )
    if result.acceptance_fraction is not None:
        footer += f"; MCMC acceptance={result.acceptance_fraction:.3f}"
    fig.text(0.5, 0.008, footer, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))


def result_to_json_dict(
    lc: LightCurveData,
    result: FitResult,
    settings: FitSettings,
    metadata: Optional[LCMetadata] = None,
    rossby_result: Optional[RossbyResult] = None,
) -> dict:
    analysis = result.period_analysis
    measurement_rows = []
    for (segment, method), measurement in analysis.measurements.items():
        measurement_rows.append(
            {
                "segment": segment,
                "method": method,
                "period_days": measurement.period,
                "peaks": measurement.peaks,
                "diagnostic_metadata": measurement.diagnostic_metadata,
                "score": measurement.score,
                "note": measurement.note,
            }
        )
    payload = {
        "source_path": lc.source_path,
        "name": lc.name,
        "normalization": lc.normalization_note,
        "gp_backend": result.gp_backend,
        "gp_kernel": result.gp_kernel,
        "gp_prior_source": result.gp_prior_source,
        "gp_prior_center_days": result.gp_prior_center,
        "period_map_days": result.map_period,
        "period_median_days": result.period_median,
        "period_p16_days": result.period_p16,
        "period_p84_days": result.period_p84,
        "map_log_posterior": result.map_log_posterior,
        "period_analysis": {
            "measurements": measurement_rows,
            "acf_combined_days": analysis.acf_combined,
            "bgls_combined_days": analysis.bgls_combined,
            "final_averaged_prot_days": analysis.final_averaged_prot,
            "acf_n_used": analysis.acf_n_used,
            "bgls_n_used": analysis.bgls_n_used,
        },
        "map_hyperparameters": result.map_hyperparameters,
        "counts": {
            "n_input": result.n_input,
            "n_clean": result.n_clean,
            "n_gp": result.n_gp,
            "n_chunks": result.n_chunks,
        },
        "mcmc_acceptance_fraction": result.acceptance_fraction,
        "period_mixture": [
            {
                "weight": row[0],
                "mean_ln_period": row[1],
                "sigma_ln_period": row[2],
                "mean_period_days": math.exp(row[1]),
            }
            for row in result.period_mixture
        ],
        "settings": asdict(settings),
    }
    if rossby_result is not None:
        payload["rossby"] = asdict(rossby_result)
    if metadata is not None:
        payload["metadata"] = asdict(metadata)
        payload["result_table"] = period_analysis_dataframe(
            metadata, analysis, result, rossby_result
        ).to_dict(orient="records")
    return payload


# -----------------------------------------------------------------------------
# Classical period searches, diagnostic peaks, folding, and Rossby calibrations
# -----------------------------------------------------------------------------

def strongest_peaks(measurement: PeriodMeasurement, baseline: float, count: int = 5) -> list[dict]:
    """Rank local maxima separated by >= 1/T in frequency (not adjacent grid bins).

    Ratios to the strongest peak identify harmonic candidates, not proven aliases.
    The search-boundary global maximum is retained and explicitly flagged.
    """
    p, y = np.asarray(measurement.periods), np.asarray(measurement.power)
    good = np.isfinite(p) & (p > 0) & np.isfinite(y)
    p, y = p[good], y[good]
    if not p.size: return []
    order = np.argsort(1 / p); p, y = p[order], y[order]
    candidates = list(find_peaks(y)[0])
    best = int(np.argmax(y))
    if best not in candidates: candidates.append(best)
    chosen = []
    for idx in sorted(candidates, key=lambda i: y[i], reverse=True):
        if any(abs(1/p[idx]-1/p[j]) < 1/max(baseline, 1e-12) for j in chosen): continue
        chosen.append(idx)
        if len(chosen) >= count: break
    rows = []
    for rank, idx in enumerate(chosen, 1):
        ratio = float(p[idx]/p[best])
        relation = "strongest" if idx == best else ""
        for factor, name in ((.5,"near P/2"),(2.,"near 2P")):
            if abs(ratio/factor-1) <= .05: relation = name
        boundary = idx in (0,len(p)-1)
        rows.append({"rank":rank,"period_days":float(p[idx]),"power":float(y[idx]),
                     "ratio_to_strongest":ratio,"harmonic_candidate":relation,
                     "at_search_boundary":boundary})
    return rows


def calculate_classical_period(lc: LightCurveData, settings: FitSettings, method: str) -> PeriodMeasurement:
    t, y, err = lc.time, lc.flux, lc.flux_err
    baseline = float(np.ptp(t)); phigh = min(settings.pmax, .8*baseline)
    if phigh <= settings.pmin:
        return PeriodMeasurement("",method,np.nan,note="baseline_too_short")
    if method in {"LS","GLS"}:
        from astropy.timeseries import LombScargle
        nfreq = max(100, int(np.ceil((1/settings.pmin-1/phigh)*baseline*settings.ls_samples_per_peak))+1)
        capped = nfreq > settings.ls_max_steps
        freq = np.linspace(1/phigh, 1/settings.pmin, min(nfreq,settings.ls_max_steps))
        model = LombScargle(t-t[0], y, dy=err if method=="GLS" else None,
                           fit_mean=method=="GLS", center_data=True, nterms=1)
        power = model.power(freq, method="auto", normalization="standard")
        note = "unweighted, fixed centered mean" if method=="LS" else "uncertainty-weighted, floating mean"
        if capped: note += "; frequency grid capped: narrow the range for better resolution"
        metadata = {"normalization":"standard","frequency_steps":len(freq),"grid_capped":capped,
                    "fit_mean":method=="GLS","uses_flux_errors":method=="GLS","baseline_days":baseline}
    elif method == "FFT":
        from scipy.signal import periodogram
        from scipy.fft import next_fast_len
        dt = float(np.median(np.diff(t)))
        # No interpolation across large gaps. Use the longest well-sampled run.
        pieces = np.split(np.arange(t.size), np.flatnonzero(np.diff(t)>1.5*dt)+1)
        ix = max(pieces, key=len)
        if len(ix)<50: return PeriodMeasurement("","FFT",np.nan,note="no_contiguous_run_with_50_points")
        tt, yy = t[ix], y[ix]; spacing = np.diff(tt)
        jitter = float(np.max(np.abs(spacing/dt-1)))
        if jitter>.1:
            return PeriodMeasurement("","FFT",np.nan,note="irregular_sampling_use_LS_or_GLS",
                                     diagnostic_metadata={"max_relative_jitter":jitter})
        grid = np.arange(tt[0],tt[-1]+dt*.1,dt)
        uniform = np.interp(grid,tt,yy)
        freq,power = periodogram(uniform,fs=1/dt,window="hann",detrend="linear",
                                nfft=next_fast_len(4*len(grid)),scaling="spectrum")
        effective_baseline = float(np.ptp(grid))
        phigh = min(phigh,.8*effective_baseline)
        keep = (freq>0)&(freq>=1/phigh)&(freq<=min(1/settings.pmin,.5/dt))
        freq,power=freq[keep],power[keep]
        if not freq.size: return PeriodMeasurement("","FFT",np.nan,note="no_FFT_bins_in_search_range")
        note = f"Hann-window FFT; longest contiguous run {len(ix)}/{len(t)} points; no long-gap interpolation"
        metadata={"normalization":"power spectrum","baseline_days":effective_baseline,
                  "native_frequency_resolution_per_day":1/effective_baseline,"cadence_days":dt,
                  "run_start":float(tt[0]),"run_stop":float(tt[-1]),"run_points":len(ix),
                  "max_relative_jitter":jitter,"zero_padding_factor":4}
    else: raise ValueError(f"Unknown method: {method}")
    if np.all(~np.isfinite(power)):
        return PeriodMeasurement("",method,np.nan,note="no_finite_power")
    best=int(np.nanargmax(power))
    return PeriodMeasurement("",method,float(1/freq[best]),float(power[best]),note,
                             periods=1/freq,power=power,diagnostic_metadata=metadata)


ROSSBY_RELATIONS = ("Noyes1984", "Wright2011", "Cranmer2011")


def turnover_time(relation: str, *, b_v=np.nan, v_ks=np.nan, teff=np.nan,
                  allow_extrapolation=False) -> tuple[float,str,float,bool,str]:
    """Return tau [days] and calibration metadata. No implicit color conversion.

    Wright+2011 Eq.10: https://doi.org/10.1088/0004-637X/743/1/48
    Cranmer & Saar 2011 Eq.36: https://doi.org/10.1088/0004-637X/741/1/54
    Noyes+1984: https://doi.org/10.1086/161801
    """
    if relation=="Noyes1984":
        name,value,low,high="B-V",float(b_v),.4,1.6
        valid=np.isfinite(value) and low<=value<=high
    elif relation=="Wright2011":
        name,value,low,high="V-Ks",float(v_ks),1.1,6.6
        valid=np.isfinite(value) and low<value<high
    elif relation=="Cranmer2011":
        name,value,low,high="Teff [K]",float(teff),3300.,7000.
        valid=np.isfinite(value) and low<=value<=high
    else: raise ValueError(f"Unknown turnover-time relation: {relation}")
    if not np.isfinite(value) or (relation=="Cranmer2011" and value<=0):
        raise ValueError(f"{relation} requires a finite {name} input (positive for temperature).")
    if not valid and not allow_extrapolation:
        raise ValueError(f"{name}={value:g} is outside the {low:g}–{high:g} calibration guardrail. "
                         "Choose another relation or explicitly allow extrapolation.")
    if relation=="Noyes1984": tau,_=noyes84_convective_turnover_time(value)
    elif relation=="Wright2011": tau=10**(.73+.22*value if value<3.5 else -2.16+1.50*value-.13*value**2)
    else: tau=314.24*np.exp(-value/1952.5-(value/6250.)**18)+.002
    if not np.isfinite(tau) or tau<=0: raise ValueError("Invalid turnover time from extrapolation.")
    note=f"{relation}; "+("within input range" if valid else "EXTRAPOLATED")
    note += "; no extinction correction applied; main-sequence/ZAMS calibration may not apply to young stars"
    if relation=="Cranmer2011" and tau<.5: note+="; hot-star turnover time is very small, so Rossby may be large"
    return float(tau),name,value,bool(valid),note


def _tap_rows(endpoint: str, query: str, timeout=25.) -> list[dict]:
    body,_=_secure_tap_post(endpoint,{"request":"doQuery","lang":"ADQL","format":"csv","query":query},timeout)
    if body.lstrip().startswith("<"): raise RuntimeError("Catalog returned an XML/HTML error instead of a CSV table")
    return [{str(k).strip().lower():v for k,v in row.items()} for row in csv.DictReader(io.StringIO(body))]


def query_rossby_inputs(member_id: str, relation: str, progress=None, extra_candidates=None) -> dict:
    """Exact identifier resolution; never silently use generic K as 2MASS Ks.

    B,V: SIMBAD allfluxes; Ks: IRSA 2MASS PSC by the SIMBAD 2MASS identifier;
    Teff: median of finite positive SIMBAD mesFe_h literature measurements.
    Individual temperatures and bibliographic references are retained.
    """
    if relation not in ROSSBY_RELATIONS: raise ValueError("Unknown Rossby relation")
    errors=[];resolved_rows=None;selected_endpoint=None;queried=""
    names=list(dict.fromkeys([*(extra_candidates or []),*_query_name_candidates(member_id)]))
    for name in names:
        escaped=str(name).replace("'","''")
        for endpoint in SIMBAD_TAP_ENDPOINTS:
            try:
                if progress: progress(f"Resolving {name} in SIMBAD")
                rows=_tap_rows(endpoint,f"SELECT DISTINCT basic.oid, basic.main_id FROM basic JOIN ident ON basic.oid=ident.oidref WHERE ident.id='{escaped}'")
                if len(rows)>1: raise ValueError("Ambiguous identifier: multiple objects resolved")
                if not rows: continue
                resolved_rows=rows;selected_endpoint=endpoint;queried=name;break
            except Exception as exc: errors.append(str(exc))
        if resolved_rows: break
    if not resolved_rows: raise RuntimeError("Could not resolve catalog identity: "+"; ".join(errors[-2:]))
    oid=int(resolved_rows[0]["oid"]);resolved=resolved_rows[0]["main_id"]
    result={"b_v":np.nan,"b_mag":np.nan,"v_mag":np.nan,"ks_mag":np.nan,"v_ks":np.nan,"teff":np.nan,
            "queried_name":queried,"resolved_name":resolved,
            "input_provenance":{"resolved_name":resolved,"simbad_oid":oid,"query_name":queried,
                                "retrieved_utc":datetime.now(__import__('datetime').timezone.utc).isoformat()}}
    def number(value):
        try: return float(value)
        except (ValueError,TypeError): return np.nan
    if relation in {"Noyes1984","Wright2011"}:
        rows=_tap_rows(selected_endpoint,f"SELECT B,V FROM allfluxes WHERE oidref={oid}")
        if not rows: raise ValueError("Resolved object has no SIMBAD B/V photometry; enter values manually")
        result["b_mag"]=number(rows[0].get("b"));result["v_mag"]=number(rows[0].get("v"))
        result["b_v"]=result["b_mag"]-result["v_mag"]
        result["input_provenance"]["photometry_source"]="SIMBAD allfluxes Johnson B,V"
        result["input_provenance"]["photometry_rows"]=rows
        if not np.isfinite(result["v_mag"]): raise ValueError("SIMBAD V is missing; enter photometry manually")
        if relation=="Noyes1984" and not np.isfinite(result["b_v"]): raise ValueError("SIMBAD B is missing; enter B-V manually")
    if relation=="Wright2011":
        ids=_tap_rows(selected_endpoint,f"SELECT id FROM ident WHERE oidref={oid} AND id LIKE '2MASS J%'")
        ids=list(dict.fromkeys(r["id"].strip().removeprefix("2MASS J") for r in ids))
        if len(ids)!=1: raise ValueError("No unique 2MASS identifier; enter a verified V-Ks color manually")
        designation=ids[0].replace("'","''")
        rows=_tap_rows("https://irsa.ipac.caltech.edu/TAP/sync",f"SELECT designation,ks_m,ks_cmsig,ph_qual FROM fp_psc WHERE designation='{designation}'")
        if len(rows)!=1: raise ValueError("No unique 2MASS PSC match; enter V-Ks manually")
        quality=rows[0].get("ph_qual","").strip()
        if len(quality)<3 or quality[2] not in "ABC": raise ValueError(f"2MASS Ks quality {quality!r} is not A/B/C; inspect before manual use")
        result["ks_mag"]=number(rows[0].get("ks_m"));result["v_ks"]=result["v_mag"]-result["ks_mag"]
        if not np.isfinite(result["v_ks"]): raise ValueError("2MASS Ks is unavailable")
        result["input_provenance"]["ks_source"]="IRSA 2MASS PSC, exact designation"
        result["input_provenance"]["ks_rows"]=rows
    if relation=="Cranmer2011":
        rows=_tap_rows(selected_endpoint,f"SELECT teff,bibcode FROM mesFe_h WHERE oidref={oid}")
        values=[number(r.get("teff")) for r in rows];values=[v for v in values if np.isfinite(v) and v>0]
        if not values: raise ValueError("SIMBAD has no literature Teff; enter a measured/SED-fit Teff manually")
        result["teff"]=float(np.median(values))
        result["input_provenance"].update(teff_source="median of SIMBAD mesFe_h literature values",
            teff_measurements=rows,teff_min=min(values),teff_max=max(values),n_teff=len(values))
    return result


def analysis_json(lc, analysis, settings, rossby=None, metadata=None) -> dict:
    return {"tool_version":TOOL_VERSION,"source_path":lc.source_path,"name":lc.name,
            "settings":asdict(settings),"aggregation":analysis.aggregation,
            "candidate_period_days":analysis.final_averaged_prot,"method_combined":analysis.method_combined,
            "phase_epoch_days":float(lc.time[0]),"normalization":lc.normalization_note,
            "measurements":[{"segment":m.segment,"method":m.method,"period_days":m.period,
                             "score":m.score,"note":m.note,"peaks":m.peaks,
                             "diagnostic_metadata":m.diagnostic_metadata,
                             "period_grid":m.periods.tolist(),"power":m.power.tolist()}
                            for m in analysis.measurements.values()],
            "metadata":asdict(metadata) if metadata else None,"rossby":asdict(rossby) if rossby else None}


def strict_json(value):
    if isinstance(value, dict): return {str(k):strict_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [strict_json(v) for v in value]
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,np.bool_): return bool(value)
    return value


def peak_dataframe(analysis):
    return pd.DataFrame([{"method":m.method,"segment":m.segment,**peak}
                         for m in analysis.measurements.values() for peak in m.peaks])


def draw_peak_phase_figure(fig, lc, measurement, period=None):
    fig.clear();axes=fig.subplots(2,2);ax=axes[0,0]
    p=measurement.period if period is None else float(period)
    if not np.isfinite(p) or p<=0: raise ValueError("Choose a finite positive fold period")
    order=np.argsort(measurement.periods)
    ax.plot(measurement.periods[order],measurement.power[order],lw=1)
    for peak in measurement.peaks:
        ax.plot(peak["period_days"],peak["power"],"o",ms=4)
        ax.annotate(str(peak["rank"]),(peak["period_days"],peak["power"]),xytext=(3,5),textcoords="offset points",fontsize=8)
    for factor,color in ((.5,"C1"),(1.,"C3"),(2.,"C2")):
        if measurement.periods.size and np.min(measurement.periods)<=p*factor<=np.max(measurement.periods):
            ax.axvline(p*factor,ls="--",color=color,alpha=.7,label=f"{factor:g} P")
    ax.set(xlabel="Lag [d]" if measurement.method=="ACF" else "Period [d]",
           ylabel="ACF" if measurement.method=="ACF" else "Power / relative probability",
           title=f"{measurement.method}: {measurement.segment}\nranked peaks (not confidence levels)")
    if ax.lines: ax.legend(fontsize=7)
    segment=split_lightcurve_halves(lc)[measurement.segment]
    for ax,factor in zip([axes[0,1],axes[1,0],axes[1,1]],[.5,1.,2.]):
        ph=((segment.time-lc.time[0])/(p*factor))%1
        # Same epoch for every panel; bins are display-only.
        for offset in (0,1): ax.plot(ph+offset,segment.flux,".",ms=2,alpha=.35)
        edges=np.linspace(0,1,41);centers=(edges[1:]+edges[:-1])/2
        binned=np.array([np.median(segment.flux[(ph>=a)&(ph<b)]) if np.any((ph>=a)&(ph<b)) else np.nan for a,b in zip(edges[:-1],edges[1:])])
        for offset in (0,1):ax.plot(centers+offset,binned,"o-",ms=3,color="C3",lw=1)
        ax.set(xlabel="Phase (two cycles)",ylabel="Cleaned flux",title=f"{factor:g} P = {p*factor:.6g} d",xlim=(0,2))
    fig.suptitle(f"{lc.name} | epoch {lc.time[0]:.8f} d | inspect harmonics; no automatic adoption",fontsize=10)
    fig.tight_layout()



# -----------------------------------------------------------------------------
# Tkinter GUI
# -----------------------------------------------------------------------------


class RotationGUI:
    CONFIG_FILE = "GHOST_V0_config.json"

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("GHOST v0.2 — Stellar Rotation Workbench")
        self.root.geometry("1500x920")

        self.directory = self._load_last_directory()
        self.membership_path = self._load_last_membership_path()
        self.membership_df: Optional[pd.DataFrame] = None
        self.current_path: Optional[str] = None
        self.raw_lc: Optional[LightCurveData] = None
        self.clean_lc: Optional[LightCurveData] = None
        self.period_analysis: Optional[PeriodAnalysis] = None
        self.result: Optional[FitResult] = None
        self.rossby_result: Optional[RossbyResult] = None
        self.result_table_df = pd.DataFrame()
        self.batch_detailed_df = pd.DataFrame()
        self.batch_lc_summary_df = pd.DataFrame()
        self.batch_star_summary_df = pd.DataFrame()
        self.batch_records: list[dict] = []
        self.batch_cancel_event = threading.Event()
        self.last_settings: Optional[FitSettings] = None
        self.worker_running = False

        self._build_ui(FigureCanvasTkAgg, NavigationToolbar2Tk)
        if self.membership_path and os.path.isfile(self.membership_path):
            try:
                self._load_membership_path(self.membership_path, quiet=True)
            except Exception:
                self.membership_path = ""
        if self.directory and os.path.isdir(self.directory):
            self.load_directory()

    def _load_last_directory(self) -> str:
        try:
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as handle:
                return str(json.load(handle).get("directory", ""))
        except Exception:
            return ""

    def _load_last_membership_path(self) -> str:
        try:
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as handle:
                return str(json.load(handle).get("membership_path", ""))
        except Exception:
            return ""

    def _save_last_directory(self) -> None:
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as handle:
                json.dump({
                    "directory": self.directory,
                    "membership_path": self.membership_path,
                }, handle)
        except Exception:
            pass

    def _build_ui(self, FigureCanvasTkAgg, NavigationToolbar2Tk) -> None:
        tk, ttk = self.tk, self.ttk

        top = ttk.Frame(self.root, padding=5)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Light-curve directory:").pack(side=tk.LEFT)
        self.dir_label = ttk.Label(top, text=self.directory or "None selected")
        self.dir_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Choose Directory", command=self.choose_directory).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(top, text="Open One File", command=self.open_one_file).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(top, text="Load Membership List", command=self.choose_membership_list).pack(
            side=tk.LEFT, padx=3
        )
        self.membership_status_var = tk.StringVar(value="Membership: not loaded")
        ttk.Label(top, textvariable=self.membership_status_var).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Exit", command=self.root.destroy).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body, padding=5)
        right = ttk.Frame(body, padding=5)
        body.add(left, weight=1)
        body.add(right, weight=4)

        ttk.Label(left, text="Available light curves").pack(anchor="w")
        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            width=38,
            selectmode=tk.EXTENDED,
            exportselection=False,
        )
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self.on_selection)
        self.listbox.bind("<Double-Button-1>", lambda _event: self.load_active_selection())
        self.root.bind("<Up>", self.on_arrow_key)
        self.root.bind("<Down>", self.on_arrow_key)

        selection_buttons = ttk.Frame(left)
        selection_buttons.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(
            selection_buttons, text="Load active", command=self.load_active_selection
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(
            selection_buttons, text="Same star", command=self.select_same_star
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(
            selection_buttons, text="All", command=self.select_all_lightcurves
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(
            selection_buttons, text="Clear", command=self.clear_lightcurve_selection
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        self.quick_batch_button = ttk.Button(
            left, text="Analyze Selected LCs (Batch)", command=self.start_batch_analysis
        )
        self.quick_batch_button.pack(fill=tk.X, pady=(3, 0))
        self.selection_var = tk.StringVar(value="Selected: 0")
        ttk.Label(left, textvariable=self.selection_var).pack(anchor="w", pady=(2, 0))

        controls = ttk.Notebook(left)
        controls.pack(fill=tk.X, pady=(6, 0))
        period_tab = ttk.Frame(controls, padding=6)
        gp_tab = ttk.Frame(controls, padding=6)
        metadata_tab = ttk.Frame(controls, padding=6)
        rossby_tab = ttk.Frame(controls, padding=6)
        batch_tab = ttk.Frame(controls, padding=6)
        controls.add(period_tab, text="Periods")
        controls.add(gp_tab, text="GP")
        controls.add(metadata_tab, text="Metadata")
        controls.add(rossby_tab, text="Rossby")
        controls.add(batch_tab, text="Batch")

        self.vars = {
            "pmin": tk.StringVar(value="0.1"),
            "pmax": tk.StringVar(value="13.0"),
            "clip_upper": tk.StringVar(value="5.0"),
            "clip_lower": tk.StringVar(value="8.0"),
            "segment_selection": tk.StringVar(value="All (full+A+B)"),
            "method_selection": tk.StringVar(value="All methods"),
            "aggregation": tk.StringVar(value="median"),
            "rossby_relation": tk.StringVar(value="Noyes1984"),
            "v_ks": tk.StringVar(value=""),
            "teff": tk.StringVar(value=""),
            "allow_extrapolation": tk.BooleanVar(value=False),
            "bgls_bin_minutes": tk.StringVar(value="30"),
            "bgls_max_steps": tk.StringVar(value="3000"),
            "smooth_hours": tk.StringVar(value="6"),
            "gp_prior_source": tk.StringVar(value="final_averaged_prot"),
            "gp_backend": tk.StringVar(value="GProtation (Angus+17)"),
            "celerite_kernel": tk.StringVar(value="RotationTerm"),
            "celerite_max_points": tk.StringVar(value="5000"),
            "gp_points": tk.StringVar(value="500"),
            "chunk_size": tk.StringVar(value="250"),
            "optimizer_starts": tk.StringVar(value="10"),
            "walkers": tk.StringVar(value="40"),
            "steps": tk.StringVar(value="600"),
            "burn": tk.StringVar(value="250"),
            "run_mcmc": tk.BooleanVar(value=False),
            "paper_priors": tk.BooleanVar(value=False),
            "member_id": tk.StringVar(value=""),
            "sector": tk.StringVar(value=""),
            "lc_source": tk.StringVar(value="unknown"),
            "rossby_period_source": tk.StringVar(value="final_median_period"),
            "b_v": tk.StringVar(value=""),
            "bv_resolved_name": tk.StringVar(value="Resolved object: —"),
            "batch_run_gp": tk.BooleanVar(value=False),
            "batch_query_bv": tk.BooleanVar(value=False),
        }

        period_fields = [
            ("P min [d]", "pmin"),
            ("P max [d]", "pmax"),
            ("Upper clip [σ]", "clip_upper"),
            ("Lower clip [σ]", "clip_lower"),
            ("BGLS bin [min]", "bgls_bin_minutes"),
            ("BGLS max steps", "bgls_max_steps"),
            ("Smooth window [h]", "smooth_hours"),
        ]
        for row, (label, key) in enumerate(period_fields):
            ttk.Label(period_tab, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(period_tab, textvariable=self.vars[key], width=13).grid(
                row=row, column=1, sticky="ew", padx=(5, 0), pady=1
            )
        row = len(period_fields)
        ttk.Label(period_tab, text="Segments").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            period_tab,
            textvariable=self.vars["segment_selection"],
            values=("All (full+A+B)", "full_sector", "sectorA", "sectorB"),
            state="readonly",
            width=17,
        ).grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=1)
        row += 1
        ttk.Label(period_tab, text="Method").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            period_tab,
            textvariable=self.vars["method_selection"],
            values=("All methods", "Both", "ACF", "BGLS", "LS", "GLS", "FFT"),
            state="readonly",
            width=17,
        ).grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=1)
        row += 1
        ttk.Label(period_tab, text="Combine estimates").grid(row=row, column=0, sticky="w")
        ttk.Combobox(period_tab, textvariable=self.vars["aggregation"], values=("median", "mean"), state="readonly", width=17).grid(row=row,column=1,sticky="ew")
        row += 1
        self.period_button = ttk.Button(
            period_tab,
            text="Calculate Selected Periods",
            command=self.start_period_analysis,
        )
        self.period_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        ttk.Label(gp_tab, text="GP engine").grid(row=0, column=0, sticky="w")
        backend_box = ttk.Combobox(
            gp_tab, textvariable=self.vars["gp_backend"], values=GP_BACKENDS,
            state="readonly", width=25,
        )
        backend_box.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=1)

        ttk.Label(gp_tab, text="GP prior source").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            gp_tab, textvariable=self.vars["gp_prior_source"],
            values=("ACF_combined", "BGLS_combined", "final_averaged_prot",
                    "final_mean_period", "final_median_period"),
            state="readonly", width=25,
        ).grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=1)

        ttk.Label(gp_tab, text="celerite2 kernel").grid(row=2, column=0, sticky="w")
        kernel_box = ttk.Combobox(
            gp_tab, textvariable=self.vars["celerite_kernel"], values=CELERITE_KERNELS,
            state="readonly", width=25,
        )
        kernel_box.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=1)
        self.gp_kernel_help = tk.StringVar(value=CELERITE_KERNEL_HELP["RotationTerm"])
        ttk.Label(
            gp_tab, textvariable=self.gp_kernel_help, wraplength=300, justify=tk.LEFT
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(3, 5))

        gp_fields = [
            ("Angus GP points", "gp_points"),
            ("Angus chunk size", "chunk_size"),
            ("celerite max pts (0=all)", "celerite_max_points"),
            ("Optimizer starts", "optimizer_starts"),
            ("MCMC walkers", "walkers"),
            ("MCMC steps", "steps"),
            ("MCMC burn", "burn"),
        ]
        for offset, (label, key) in enumerate(gp_fields, start=4):
            ttk.Label(gp_tab, text=label).grid(row=offset, column=0, sticky="w")
            ttk.Entry(gp_tab, textvariable=self.vars[key], width=13).grid(
                row=offset, column=1, sticky="ew", padx=(5, 0), pady=1
            )
        gp_row = 4 + len(gp_fields)
        ttk.Checkbutton(
            gp_tab, text="MCMC refinement (slower)", variable=self.vars["run_mcmc"],
        ).grid(row=gp_row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            gp_tab, text="Exact Angus+17 Kepler hyperpriors (GProtation only)",
            variable=self.vars["paper_priors"],
        ).grid(row=gp_row + 1, column=0, columnspan=2, sticky="w")
        self.fit_button = ttk.Button(
            gp_tab, text="Fit GP Using Selected Engine", command=self.start_fit
        )
        self.fit_button.grid(row=gp_row + 2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        gp_terms = ttk.LabelFrame(gp_tab, text="GP output glossary", padding=5)
        gp_terms.grid(row=gp_row + 3, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Label(
            gp_terms,
            text=(
                "MAP period = period at the highest posterior probability point.  "
                "Posterior median = 50th percentile of the marginalized period posterior.\n"
                "P16/P84 = 16th/84th percentiles (roughly a 68% credible interval).  "
                "Prior = period information supplied before the GP fit; posterior = prior × likelihood.\n"
                "Kernel = covariance model describing how stellar variability remains correlated in time; "
                "hyperparameters control amplitude, coherence/evolution, jitter, and related behavior."
            ),
            wraplength=315, justify=tk.LEFT,
        ).pack(anchor="w")

        def _refresh_kernel_note(*_args):
            backend = self.vars["gp_backend"].get()
            if backend == "GProtation (Angus+17)":
                self.gp_kernel_help.set(
                    "Angus+17 exact quasi-periodic kernel: exp[-dt²/(2l²) - Γ sin²(πdt/P)/2]. "
                    "The celerite kernel selector is ignored for this engine."
                )
            else:
                self.gp_kernel_help.set(
                    CELERITE_KERNEL_HELP.get(
                        self.vars["celerite_kernel"].get(), "Select a celerite2 kernel."
                    )
                )
        backend_box.bind("<<ComboboxSelected>>", _refresh_kernel_note)
        kernel_box.bind("<<ComboboxSelected>>", _refresh_kernel_note)
        _refresh_kernel_note()

        for row, (label, key) in enumerate(
            (("Member ID", "member_id"), ("Sector", "sector"), ("LC source", "lc_source"))
        ):
            ttk.Label(metadata_tab, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(metadata_tab, textvariable=self.vars[key], width=24).grid(
                row=row, column=1, sticky="ew", padx=(5, 0), pady=2
            )
        ttk.Label(
            metadata_tab,
            text="These are inferred from the filename and can be corrected before export.",
            wraplength=280,
            justify=tk.LEFT,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(rossby_tab, text="Turnover relation").grid(row=0,column=0,sticky="w")
        ttk.Combobox(rossby_tab,textvariable=self.vars["rossby_relation"],values=ROSSBY_RELATIONS,state="readonly",width=22).grid(row=0,column=1,sticky="ew")
        ttk.Label(rossby_tab,text="Period source").grid(row=1,column=0,sticky="w")
        ttk.Combobox(rossby_tab,textvariable=self.vars["rossby_period_source"],values=ROSSBY_PERIOD_SOURCES,state="readonly",width=22).grid(row=1,column=1,sticky="ew")
        for row,(label,key) in enumerate((("B−V [mag]","b_v"),("V−Ks [mag]","v_ks"),("Teff [K]","teff")),2):
            ttk.Label(rossby_tab,text=label).grid(row=row,column=0,sticky="w")
            ttk.Entry(rossby_tab,textvariable=self.vars[key],width=16).grid(row=row,column=1,sticky="ew")
        ttk.Checkbutton(rossby_tab,text="Allow extrapolation (flagged in export)",variable=self.vars["allow_extrapolation"]).grid(row=5,column=0,columnspan=2,sticky="w")
        self.rossby_query_button=ttk.Button(rossby_tab,text="Query required inputs + calculate",command=lambda:self.start_rossby_calculation(True))
        self.rossby_query_button.grid(row=6,column=0,columnspan=2,sticky="ew",pady=3)
        self.rossby_manual_button=ttk.Button(rossby_tab,text="Calculate from entered values",command=lambda:self.start_rossby_calculation(False))
        self.rossby_manual_button.grid(row=7,column=0,columnspan=2,sticky="ew")
        ttk.Label(rossby_tab,textvariable=self.vars["bv_resolved_name"],wraplength=280,justify=tk.LEFT).grid(row=8,column=0,columnspan=2,sticky="w",pady=5)
        ttk.Label(rossby_tab,text="Noyes: B−V; Wright: V−Ks; Cranmer: Teff. Queries use SIMBAD and verified 2MASS Ks. No extinction correction. Inspect suitability for young stars.",wraplength=280,justify=tk.LEFT).grid(row=9,column=0,columnspan=2,sticky="w")

        ttk.Checkbutton(
            batch_tab,
            text="Run GP for every selected LC",
            variable=self.vars["batch_run_gp"],
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Checkbutton(
            batch_tab,
            text="Query chosen Rossby inputs per star",
            variable=self.vars["batch_query_bv"],
        ).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(
            batch_tab,
            text=(
                "Batch uses the selected algorithms and segments. Finite estimates are combined "
                "within each star/sector, then across sectors with equal sector weight. "
                "A membership list, when loaded, canonicalizes star names before grouping/querying."
            ),
            wraplength=280, justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 2))
        self.batch_run_button = ttk.Button(
            batch_tab, text="Analyze Selected LCs", command=self.start_batch_analysis
        )
        self.batch_run_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 2))
        self.batch_cancel_button = ttk.Button(
            batch_tab, text="Cancel Batch", command=self.cancel_batch, state="disabled"
        )
        self.batch_cancel_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)
        self.batch_save_button = ttk.Button(
            batch_tab, text="Save Batch CSV Tables", command=self.save_batch_tables
        )
        self.batch_save_button.grid(row=5, column=0, columnspan=2, sticky="ew", pady=2)
        self.batch_progress_var = tk.DoubleVar(value=0.0)
        self.batch_progress = ttk.Progressbar(
            batch_tab, variable=self.batch_progress_var, maximum=100.0
        )
        self.batch_progress.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        summary = ttk.LabelFrame(left, text="Current LC / star summary", padding=6)
        summary.pack(fill=tk.X, pady=6)
        self.summary_vars = {
            "acf": tk.StringVar(value="ACF combined: —"),
            "bgls": tk.StringVar(value="BGLS combined: —"),
            "final": tk.StringVar(value="LC candidate: —"),
            "star_mean": tk.StringVar(value="Star mean period: —"),
            "star_median": tk.StringVar(value="Star median period: —"),
            "gp": tk.StringVar(value="GP period: —"),
            "rossby": tk.StringVar(value="Rossby: —"),
        }
        for key in ("acf", "bgls", "final", "star_mean", "star_median", "gp", "rossby"):
            ttk.Label(summary, textvariable=self.summary_vars[key]).pack(anchor="w")

        export_box = ttk.Frame(left)
        export_box.pack(fill=tk.X)
        ttk.Button(export_box, text="Save Figure", command=self.save_figure).pack(
            fill=tk.X, pady=2
        )
        self.export_table_button = ttk.Button(
            export_box, text="Export Active Table CSV", command=self.export_table
        )
        self.export_table_button.pack(fill=tk.X, pady=2)
        ttk.Button(
            export_box, text="Save All Batch CSVs", command=self.save_batch_tables
        ).pack(fill=tk.X, pady=2)
        ttk.Button(export_box, text="Export Result JSON", command=self.export_json).pack(
            fill=tk.X, pady=2
        )

        self.status_var = tk.StringVar(value="Choose a light curve.")
        ttk.Label(
            left, textvariable=self.status_var, wraplength=320, justify=tk.LEFT
        ).pack(fill=tk.X, pady=(4, 0))

        right_tabs = ttk.Notebook(right)
        right_tabs.pack(fill=tk.BOTH, expand=True)
        plot_tab = ttk.Frame(right_tabs)
        table_tab = ttk.Frame(right_tabs)
        batch_detail_tab = ttk.Frame(right_tabs)
        batch_lc_tab = ttk.Frame(right_tabs)
        batch_star_tab = ttk.Frame(right_tabs)
        right_tabs.add(plot_tab, text="Plots")
        right_tabs.add(batch_detail_tab, text="Batch detailed")
        right_tabs.add(batch_lc_tab, text="Batch LC summary")
        right_tabs.add(batch_star_tab, text="Final star summary")
        self.right_tabs = right_tabs
        self.plot_tab = plot_tab
        self.table_tab = table_tab
        self.batch_detail_tab = batch_detail_tab
        self.batch_lc_tab = batch_lc_tab
        self.batch_star_tab = batch_star_tab
        self.right_tabs.bind("<<NotebookTabChanged>>", self._update_export_table_button_text)

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self.peak_tab = ttk.Frame(right_tabs)
        right_tabs.add(self.peak_tab,text="Peaks & phase")
        bar=ttk.Frame(self.peak_tab);bar.pack(fill=tk.X,padx=6,pady=4)
        self.diagnostic_var=tk.StringVar()
        self.diagnostic_choice=ttk.Combobox(bar,textvariable=self.diagnostic_var,state="readonly",width=25)
        self.diagnostic_choice.pack(side=tk.LEFT)
        self.diagnostic_choice.bind("<<ComboboxSelected>>",lambda e:self.refresh_diagnostics())
        ttk.Label(bar,text="Fold period [d]").pack(side=tk.LEFT,padx=6)
        self.fold_var=tk.StringVar()
        ttk.Entry(bar,textvariable=self.fold_var,width=13).pack(side=tk.LEFT)
        ttk.Button(bar,text="Fold / compare P/2, P, 2P",command=self.draw_diagnostics).pack(side=tk.LEFT,padx=6)
        ttk.Button(bar,text="Export diagnostic",command=self.export_diagnostic).pack(side=tk.LEFT)
        self.peak_tree=ttk.Treeview(self.peak_tab,columns=("rank","period","power","ratio","flag"),show="headings",height=5)
        for key,title in (("rank","Rank"),("period","Period [d]"),("power","Power / score"),("ratio","P / strongest"),("flag","Harmonic / boundary flag")):
            self.peak_tree.heading(key,text=title);self.peak_tree.column(key,width=140)
        self.peak_tree.pack(fill=tk.X,padx=6)
        self.peak_tree.bind("<<TreeviewSelect>>",self.select_peak)
        self.diagnostic_note=tk.StringVar(value="Calculate a period first. Select a ranked peak or enter a fold period.")
        ttk.Label(self.peak_tab,textvariable=self.diagnostic_note,wraplength=900).pack(fill=tk.X,padx=6)
        self.diagnostic_fig=Figure(figsize=(10,7),dpi=100)
        self.diagnostic_canvas=FigureCanvasTkAgg(self.diagnostic_fig,master=self.peak_tab)
        self.diagnostic_canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        NavigationToolbar2Tk(self.diagnostic_canvas,self.peak_tab).update()

        columns = (
            "member_id", "sector", "prot", "method", "note", "lc_source",
            "gp_backend", "gp_kernel",
            "b_v", "tau_c_days", "rossby_number", "rossby_period_source",
            "rossby_period_days", "bv_query_name", "bv_resolved_name",
        )
        table_frame = ttk.Frame(table_tab, padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        xscroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
        )
        yscroll.config(command=self.table.yview)
        xscroll.config(command=self.table.xview)
        widths = {
            "member_id": 180,
            "sector": 70,
            "prot": 120,
            "method": 110,
            "note": 220,
            "lc_source": 130,
            "gp_backend": 180,
            "gp_kernel": 190,
            "b_v": 85,
            "tau_c_days": 105,
            "rossby_number": 105,
            "rossby_period_source": 155,
            "rossby_period_days": 120,
            "bv_query_name": 150,
            "bv_resolved_name": 170,
        }
        for column in columns:
            self.table.heading(column, text=column)
            self.table.column(column, width=widths[column], anchor="center")
        self.table.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.batch_detailed_columns = (
            "batch_index", "member_id", "sector", "lc_source", "lc_file",
            "prot", "method", "note",
        )
        self.batch_detailed_table = self._create_table_widget(
            batch_detail_tab, self.batch_detailed_columns
        )

        self.batch_lc_columns = (
            "member_id", "sector", "lc_source", "acf_combined",
            "bgls_combined", "final_averaged_prot",
            *tuple(f"{m.lower()}_{seg}" for m in ("LS","GLS","FFT") for seg in SEGMENT_ORDER),
            "final_mean_period", "final_median_period",
            "gp_map", "gp_median", "gp_p16", "gp_p84",
            "gp_backend", "gp_kernel", "gp_prior_source",
            "selected_lc_prot", "selected_period_source", "status", "lc_file",
        )
        self.batch_lc_table = self._create_table_widget(
            batch_lc_tab, self.batch_lc_columns
        )
        self.batch_star_columns = (
            "member_id", "final_mean_period", "final_median_period",
            "period_source_per_lc", "n_lcs_selected", "n_lcs_used", "n_sectors_used", "aggregation",
            "std_prot", "min_prot", "max_prot", "consistency_flag",
            "sectors", "lc_sources", "gp_backends", "gp_kernels",
            "b_v", "tau_c_days", "rossby_number",
            "rossby_relation", "rossby_input_name", "rossby_input_value", "v_ks", "teff",
            "rossby_period_source", "rossby_period_days",
        )
        self.batch_star_table = self._create_table_widget(
            batch_star_tab, self.batch_star_columns
        )

    def _create_table_widget(self, parent, columns):
        tk, ttk = self.tk, self.ttk
        frame = ttk.Frame(parent, padding=5)
        frame.pack(fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
        )
        yscroll.config(command=tree.yview)
        xscroll.config(command=tree.xview)
        for column in columns:
            tree.heading(column, text=column)
            width = 170 if column in {"member_id", "lc_file", "consistency_flag"} else 120
            if column in {"selected_period_source", "period_source_per_lc"}:
                width = 165
            if column in {"gp_backend", "gp_backends"}:
                width = 190
            if column in {"gp_kernel", "gp_kernels"}:
                width = 200
            if column in {"sectors", "lc_sources"}:
                width = 150
            if column == "note":
                width = 210
            if column == "prot":
                width = 115
            tree.column(column, width=width, anchor="center")
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def settings_from_gui(self) -> FitSettings:
        return FitSettings(
            aggregation=self.vars["aggregation"].get(),
            pmin=float(self.vars["pmin"].get()),
            pmax=float(self.vars["pmax"].get()),
            clip_upper=float(self.vars["clip_upper"].get()),
            clip_lower=float(self.vars["clip_lower"].get()),
            gp_points=int(self.vars["gp_points"].get()),
            chunk_size=int(self.vars["chunk_size"].get()),
            optimizer_starts=int(self.vars["optimizer_starts"].get()),
            run_mcmc=bool(self.vars["run_mcmc"].get()),
            mcmc_walkers=int(self.vars["walkers"].get()),
            mcmc_steps=int(self.vars["steps"].get()),
            mcmc_burn=int(self.vars["burn"].get()),
            use_paper_hyperpriors=bool(self.vars["paper_priors"].get()),
            bgls_bin_minutes=float(self.vars["bgls_bin_minutes"].get()),
            bgls_max_steps=int(self.vars["bgls_max_steps"].get()),
            rotation_smooth_hours=float(self.vars["smooth_hours"].get()),
            gp_backend=self.vars["gp_backend"].get(),
            celerite_kernel=self.vars["celerite_kernel"].get(),
            celerite_max_points=int(self.vars["celerite_max_points"].get()),
        )

    def metadata_from_gui(self) -> LCMetadata:
        return LCMetadata(
            member_id=self.vars["member_id"].get().strip() or "unknown",
            sector=self.vars["sector"].get().strip(),
            lc_source=self.vars["lc_source"].get().strip() or "unknown",
        )

    def _load_membership_path(self, path: str, quiet: bool = False) -> None:
        frame = load_membership_table(path)
        designation_col = membership_designation_column(frame)
        self.membership_df = frame
        self.membership_path = path
        self._save_last_directory()
        self.membership_status_var.set(
            f"Membership: {Path(path).name} ({len(frame)} rows; ID={designation_col})"
        )
        if not quiet:
            self.status_var.set(
                f"Loaded membership list {Path(path).name}. It will canonicalize star IDs "
                "and provide better identifiers for SIMBAD B−V queries."
            )

    def choose_membership_list(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.askopenfilename(
            initialdir=str(Path(self.membership_path).parent) if self.membership_path else (self.directory or None),
            filetypes=[
                ("Membership tables", "*.parquet *.pq *.csv *.tsv *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._load_membership_path(path)
            # Re-canonicalize currently loaded LC metadata immediately.
            if self.current_path:
                metadata = self.infer_metadata(self.current_path)
                self.vars["member_id"].set(metadata.member_id)
                self.vars["lc_source"].set(metadata.lc_source)
        except Exception as exc:
            messagebox.showerror("Membership-list error", str(exc))

    def infer_metadata(self, path: str) -> LCMetadata:
        metadata = infer_lc_metadata(path)
        canonical, _ = match_membership_row(self.membership_df, metadata.member_id)
        return LCMetadata(canonical, metadata.sector, metadata.lc_source)

    def membership_row_for_member(self, member_id: str) -> Optional[pd.Series]:
        _, row = match_membership_row(self.membership_df, member_id)
        return row

    def membership_query_candidates_for_member(self, member_id: str) -> list[str]:
        canonical, row = match_membership_row(self.membership_df, member_id)
        return membership_query_names(canonical, row)

    def choose_directory(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(initialdir=self.directory or None)
        if selected:
            self.directory = selected
            self.dir_label.config(text=selected)
            self._save_last_directory()
            self.load_directory()

    def open_one_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            initialdir=self.directory or None,
            filetypes=[
                ("Light curves", "*.fits *.fit *.fts *.fz *.csv *.txt *.dat *.tsv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.directory = str(Path(path).parent)
            self.dir_label.config(text=self.directory)
            self._save_last_directory()
            self.load_selected_path(path)

    def load_directory(self) -> None:
        self.listbox.delete(0, self.tk.END)
        if not self.directory or not os.path.isdir(self.directory):
            return
        valid = {".fits", ".fit", ".fts", ".fz", ".csv", ".txt", ".dat", ".tsv"}
        files = sorted(
            name for name in os.listdir(self.directory) if Path(name).suffix.lower() in valid
        )
        for name in files:
            self.listbox.insert(self.tk.END, name)
        self.status_var.set(f"Found {len(files)} light-curve file(s).")

    def on_arrow_key(self, event) -> None:
        size = self.listbox.size()
        if size == 0:
            return
        current = self.listbox.curselection()
        index = current[0] if current else 0
        if event.keysym == "Up" and index > 0:
            index -= 1
        elif event.keysym == "Down" and index < size - 1:
            index += 1
        else:
            return
        self.listbox.selection_clear(0, self.tk.END)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)
        self.on_selection(None)

    def selected_lightcurve_paths(self) -> list[str]:
        return [
            os.path.join(self.directory, self.listbox.get(index))
            for index in self.listbox.curselection()
        ]

    def active_lightcurve_path(self) -> Optional[str]:
        if self.listbox.size() == 0:
            return None
        try:
            index = int(self.listbox.index(self.tk.ACTIVE))
        except Exception:
            selection = self.listbox.curselection()
            if not selection:
                return None
            index = int(selection[-1])
        if index < 0 or index >= self.listbox.size():
            return None
        return os.path.join(self.directory, self.listbox.get(index))

    def on_selection(self, _event) -> None:
        count = len(self.listbox.curselection())
        self.selection_var.set(f"Selected: {count}")
        if count > 1:
            self.period_button.config(text=f"Calculate ACF/BGLS for {count} Selected LCs")
        else:
            self.period_button.config(text="Calculate Selected Periods")
        if count == 1:
            path = self.selected_lightcurve_paths()[0]
            if path != self.current_path:
                self.load_selected_path(path)
        elif count > 1:
            unique_stars = {
                normalize_member_key(self.infer_metadata(path).member_id)
                for path in self.selected_lightcurve_paths()
            }
            self.status_var.set(
                f"Selected {count} light curves from {len(unique_stars)} inferred star(s). "
                "Use the Batch tab to analyze them; double-click or Load active to view one."
            )

    def load_active_selection(self) -> None:
        path = self.active_lightcurve_path()
        if path is not None:
            self.load_selected_path(path)

    def select_all_lightcurves(self) -> None:
        if self.listbox.size() == 0:
            return
        self.listbox.selection_set(0, self.tk.END)
        self.on_selection(None)

    def clear_lightcurve_selection(self) -> None:
        self.listbox.selection_clear(0, self.tk.END)
        self.selection_var.set("Selected: 0")
        self.status_var.set("Selection cleared.")

    def select_same_star(self) -> None:
        from tkinter import messagebox

        path = self.active_lightcurve_path()
        if path is None:
            messagebox.showwarning("No active LC", "Select one light curve first.")
            return
        target_metadata = self.infer_metadata(path)
        target_key = normalize_member_key(target_metadata.member_id)
        matches = []
        for index in range(self.listbox.size()):
            candidate_path = os.path.join(self.directory, self.listbox.get(index))
            candidate_key = normalize_member_key(
                self.infer_metadata(candidate_path).member_id
            )
            if candidate_key == target_key:
                matches.append(index)
        self.listbox.selection_clear(0, self.tk.END)
        for index in matches:
            self.listbox.selection_set(index)
        if matches:
            self.listbox.activate(matches[0])
            self.listbox.see(matches[0])
        self.on_selection(None)
        self.status_var.set(
            f"Selected {len(matches)} LC(s) inferred as {target_metadata.member_id}."
        )

    def load_selected_path(self, path: str) -> None:
        from tkinter import messagebox

        try:
            self.current_path = path
            self.raw_lc = load_lightcurve(path)
            self.clean_lc = None
            self.period_analysis = None
            self.result = None
            self.rossby_result = None
            self.result_table_df = pd.DataFrame()
            metadata = self.infer_metadata(path)
            self.vars["member_id"].set(metadata.member_id)
            self.vars["sector"].set(metadata.sector)
            self.vars["lc_source"].set(metadata.lc_source)
            self.vars["b_v"].set("")
            self.vars["v_ks"].set("")
            self.vars["teff"].set("")
            self.refresh_diagnostics(reset=True)
            self.vars["bv_resolved_name"].set("Resolved object: —")
            self._reset_summary()
            self._refresh_table()
            self.plot_raw()
            self.status_var.set(
                f"Loaded {Path(path).name}: {self.raw_lc.time.size} points. "
                "Calculate full/A/B periods, then fit the GP."
            )
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def plot_raw(self) -> None:
        if self.raw_lc is None:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self.raw_lc.time, self.raw_lc.flux, ".", ms=3, alpha=1.0)
        #ax.plot(self.raw_lc.time, self.raw_lc.flux, linewidth = 1, color = "blue")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Flux")
        ax.set_title(self.raw_lc.name)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _batch_record_for_path(self, path: Optional[str]) -> Optional[dict]:
        if not path:
            return None
        target = os.path.abspath(path)
        for record in self.batch_records:
            if os.path.abspath(str(record.get("path", ""))) == target:
                return record
        return None

    def _active_analysis_and_gp(self) -> tuple[Optional[PeriodAnalysis], Optional[FitResult]]:
        if self.period_analysis is not None:
            return self.period_analysis, self.result
        record = self._batch_record_for_path(self.current_path or self.active_lightcurve_path())
        if record is None:
            return None, None
        return record.get("analysis"), record.get("gp_result")

    def _star_summary_row(self, member_id: Optional[str] = None) -> Optional[pd.Series]:
        if self.batch_star_summary_df.empty:
            return None
        member_id = member_id or self.vars["member_id"].get().strip()
        if not member_id or member_id == "unknown":
            path = self.current_path or self.active_lightcurve_path()
            if path:
                member_id = self.infer_metadata(path).member_id
        key = normalize_member_key(member_id)
        matches = self.batch_star_summary_df.loc[
            self.batch_star_summary_df["member_key"].astype(str) == key
        ] if "member_key" in self.batch_star_summary_df.columns else pd.DataFrame()
        return None if matches.empty else matches.iloc[0]

    def _resolve_star_reference_period(self, source: str) -> float:
        if source not in {"final_mean_period", "final_median_period"}:
            raise ValueError(f"Not a star-level period source: {source}")
        row = self._star_summary_row()
        if row is not None:
            value = float(row.get(source, np.nan))
            if np.isfinite(value) and value > 0:
                return value
        # With exactly one current LC, mean = median = its per-LC final period.
        analysis, _ = self._active_analysis_and_gp()
        selected = self.selected_lightcurve_paths()
        if analysis is not None and len(selected) <= 1:
            value = float(analysis.final_averaged_prot)
            if np.isfinite(value) and value > 0:
                return value
        raise ValueError(
            f"{source} is not available yet. Run ACF/BGLS on the selected LCs first "
            "so GHOST can build the one-row-per-star summary."
        )

    def _resolve_gui_period_source(self, source: str) -> float:
        if source in {"final_mean_period", "final_median_period"}:
            return self._resolve_star_reference_period(source)
        analysis, gp_result = self._active_analysis_and_gp()
        return resolve_rossby_period(source, analysis, gp_result)

    def _sync_current_result_into_batch(self) -> None:
        """Keep batch tables synchronized when a single-LC GP/period fit is run."""
        if not self.current_path or self.period_analysis is None:
            return
        metadata = self.metadata_from_gui()
        record = self._batch_record_for_path(self.current_path)
        if record is None:
            record = {
                "path": self.current_path, "metadata": metadata,
                "analysis": self.period_analysis, "gp_result": self.result, "error": "",
            }
            self.batch_records.append(record)
        else:
            record.update({
                "metadata": metadata, "analysis": self.period_analysis,
                "gp_result": self.result, "error": "",
            })
        old_star = self.batch_star_summary_df.copy()
        detailed, lc_summary, star_summary = build_batch_tables(
            self.batch_records,
            period_source="final_averaged_prot",
            aggregation=self.last_settings.aggregation if self.last_settings else "median",
        )
        # Preserve existing catalog/Rossby results when a later GP fit refreshes
        # the period tables.
        if not old_star.empty and not star_summary.empty:
            preserve_cols = [
                "b_v", "b_mag", "v_mag", "tau_c_days", "rossby_number",
                "rossby_period_source", "rossby_period_days",
                "bv_query_name", "bv_resolved_name", "rossby_note",
        "rossby_relation", "rossby_input_name", "rossby_input_value", "v_ks", "ks_mag", "teff",
        "in_calibration", "input_provenance",
            ]
            old_lookup = old_star.set_index("member_key")
            for idx, row in star_summary.iterrows():
                key = row["member_key"]
                if key not in old_lookup.index:
                    continue
                old_row = old_lookup.loc[key]
                if isinstance(old_row, pd.DataFrame):
                    old_row = old_row.iloc[0]
                unchanged_periods = key != normalize_member_key(metadata.member_id) and all(
                    np.isclose(float(row[name]), float(old_row.get(name, np.nan)), equal_nan=True)
                    for name in ("final_mean_period", "final_median_period")
                )
                safe_inputs = {"b_v", "b_mag", "v_mag", "v_ks", "ks_mag", "teff",
                               "bv_query_name", "bv_resolved_name", "input_provenance"}
                for column in preserve_cols:
                    if not unchanged_periods and column not in safe_inputs:
                        continue
                    if column in old_row.index:
                        value = old_row[column]
                        if not (isinstance(value, float) and np.isnan(value)) and str(value) not in {"", "nan"}:
                            star_summary.at[idx, column] = value
            detailed, lc_summary = propagate_star_rossby_to_batch_tables(
                detailed, lc_summary, star_summary
            )
        self.batch_detailed_df = detailed
        self.batch_lc_summary_df = lc_summary
        self.batch_star_summary_df = star_summary
        self._populate_batch_tree(self.batch_detailed_table, detailed, self.batch_detailed_columns)
        self._populate_batch_tree(self.batch_lc_table, lc_summary, self.batch_lc_columns)
        self._populate_batch_tree(self.batch_star_table, star_summary, self.batch_star_columns)

    def _reset_summary(self) -> None:
        self.summary_vars["acf"].set("ACF combined: —")
        self.summary_vars["bgls"].set("BGLS combined: —")
        self.summary_vars["final"].set("LC candidate: —")
        self.summary_vars["star_mean"].set("Star mean period: —")
        self.summary_vars["star_median"].set("Star median period: —")
        self.summary_vars["gp"].set("GP period: —")
        self.summary_vars["rossby"].set("Rossby: —")

    def _set_busy(self, busy: bool) -> None:
        self.worker_running = busy
        state = "disabled" if busy else "normal"
        self.fit_button.config(state=state)
        self.period_button.config(state=state)
        self.rossby_query_button.config(state=state)
        self.rossby_manual_button.config(state=state)
        self.batch_run_button.config(state=state)
        self.batch_save_button.config(state=state)
        self.quick_batch_button.config(state=state)
        if not busy:
            self.batch_cancel_button.config(state="disabled")

    def _thread_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def start_period_analysis(self) -> None:
        from tkinter import messagebox

        selected_paths = self.selected_lightcurve_paths()
        if len(selected_paths) > 1:
            # Multi-selection means batch mode.  This is deliberately automatic so
            # the ordinary ACF/BGLS button cannot silently operate on only the
            # previously loaded LC while several files are selected.
            self.start_batch_analysis(quick_acf_bgls=True)
            return

        if self.raw_lc is None:
            messagebox.showwarning("No light curve", "Select a light curve first.")
            return
        if self.worker_running:
            return
        try:
            settings = self.settings_from_gui()
        except Exception as exc:
            messagebox.showerror("Settings error", str(exc))
            return
        self._set_busy(True)
        segment = self.vars["segment_selection"].get()
        method = self.vars["method_selection"].get()

        def worker():
            try:
                clean, analysis = calculate_period_analysis(
                    self.raw_lc,
                    settings,
                    segment_selection=segment,
                    method_selection=method,
                    progress=self._thread_status,
                )
                self.root.after(
                    0,
                    lambda: self._show_period_analysis(clean, analysis, settings),
                )
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(
                    0,
                    lambda exc=exc, tb=tb: self._show_worker_error(
                        "Period-analysis error", exc, tb
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _show_period_analysis(
        self,
        clean: LightCurveData,
        analysis: PeriodAnalysis,
        settings: FitSettings,
    ) -> None:
        self._set_busy(False)
        self.clean_lc = clean
        self.period_analysis = analysis
        self.result = None
        self.rossby_result = None
        self.last_settings = settings
        draw_period_analysis_figure(self.fig, clean, analysis)
        self.canvas.draw_idle()
        self.refresh_diagnostics(reset=True)
        self._sync_current_result_into_batch()
        self._update_summary()
        self._refresh_table()
        self.status_var.set(
            "Period calculation complete. GP priors can use ACF/BGLS combined, the "
            "per-LC final period, or the current star mean/median when available."
        )

    def start_fit(self) -> None:
        from tkinter import messagebox

        if self.raw_lc is None:
            messagebox.showwarning("No light curve", "Select a light curve first.")
            return
        if self.worker_running:
            return
        try:
            settings = self.settings_from_gui()
        except Exception as exc:
            messagebox.showerror("Settings error", str(exc))
            return
        prior_source = self.vars["gp_prior_source"].get()
        existing_analysis = self.period_analysis
        self._set_busy(True)

        def worker():
            try:
                analysis = existing_analysis
                if analysis is None:
                    self._thread_status(
                        "No combined periods yet; calculating full/A/B ACF and BGLS first"
                    )
                    _, analysis = calculate_period_analysis(
                        self.raw_lc, settings, segment_selection="all",
                        method_selection="both", progress=self._thread_status,
                    )

                prior_override = None
                if prior_source in {"final_mean_period", "final_median_period"}:
                    star_row = self._star_summary_row()
                    if star_row is not None:
                        prior_override = float(star_row.get(prior_source, np.nan))
                    elif len(self.selected_lightcurve_paths()) <= 1:
                        # One LC => across-LC mean and median are identical.
                        prior_override = float(analysis.final_averaged_prot)
                    if prior_override is None or not np.isfinite(prior_override) or prior_override <= 0:
                        raise ValueError(
                            f"{prior_source} is unavailable. Run the ACF/BGLS batch on the "
                            "selected light curves first, then fit the GP."
                        )
                else:
                    try:
                        analysis.prior_center(prior_source)
                    except ValueError:
                        self._thread_status(
                            f"{prior_source} unavailable; recalculating all full/A/B methods"
                        )
                        _, analysis = calculate_period_analysis(
                            self.raw_lc, settings, segment_selection="all",
                            method_selection="both", progress=self._thread_status,
                        )

                clean, result = fit_rotation_period(
                    self.raw_lc, settings, progress=self._thread_status,
                    period_analysis=analysis, gp_prior_source=prior_source,
                    gp_prior_override=prior_override,
                )
                self.root.after(0, lambda: self._show_fit_result(clean, result, settings))
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(
                    0, lambda exc=exc, tb=tb: self._show_worker_error("GP fit error", exc, tb)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _show_fit_result(
        self,
        clean: LightCurveData,
        result: FitResult,
        settings: FitSettings,
    ) -> None:
        self._set_busy(False)
        self.clean_lc = clean
        self.period_analysis = result.period_analysis
        self.result = result
        self.rossby_result = None
        self.last_settings = settings
        draw_fit_figure(self.fig, clean, result, settings)
        self.canvas.draw_idle()
        self.refresh_diagnostics(reset=True)
        self._sync_current_result_into_batch()
        self._update_summary()
        self._refresh_table()
        text = (
            f"{result.gp_backend} / {result.gp_kernel}: GP MAP P={result.map_period:.6g} d; posterior median "
            f"{result.period_median:.6g} (-{result.period_minus:.3g}/"
            f"+{result.period_plus:.3g}) d; prior={result.gp_prior_source}."
        )
        if result.acceptance_fraction is not None:
            text += f" MCMC acceptance={result.acceptance_fraction:.3f}."
        self.status_var.set(text)

    def start_rossby_calculation(self, query_bv: bool) -> None:
        from tkinter import messagebox

        if self.worker_running:
            return
        period_source = self.vars["rossby_period_source"].get()
        try:
            period_value = self._resolve_gui_period_source(period_source)
            relation=self.vars["rossby_relation"].get()
            extrapolate=bool(self.vars["allow_extrapolation"].get())
            manual={key:float(self.vars[key].get()) if self.vars[key].get().strip() else np.nan for key in ("b_v","v_ks","teff")}
        except Exception as exc:
            messagebox.showerror("Rossby settings error", str(exc))
            return

        metadata = self.metadata_from_gui()
        canonical, membership_row = match_membership_row(self.membership_df, metadata.member_id)
        query_candidates = membership_query_names(canonical, membership_row)
        analysis, gp_result = self._active_analysis_and_gp()
        self._set_busy(True)

        def worker():
            try:
                if query_bv:
                    inputs=query_rossby_inputs(canonical,relation,progress=self._thread_status,extra_candidates=query_candidates)
                else:
                    inputs={**manual,"queried_name":"manual","resolved_name":canonical,"input_provenance":{"source":"manual; user-supplied color/Teff"}}
                rossby=calculate_rossby_number(period_source=period_source,analysis=analysis,gp_result=gp_result,
                    period_override=period_value,relation=relation,allow_extrapolation=extrapolate,**inputs)
                self.root.after(0, lambda: self._show_rossby_result(rossby))
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(
                    0, lambda exc=exc, tb=tb: self._show_worker_error(
                        "Rossby calculation error", exc, tb
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def _show_rossby_result(self, rossby: RossbyResult) -> None:
        self._set_busy(False)
        self.rossby_result = rossby
        for key in ("b_v","v_ks","teff"):
            value=getattr(rossby,key)
            self.vars[key].set(f"{value:.6g}" if np.isfinite(value) else "")
        resolved = rossby.resolved_name or rossby.queried_name or "manual"
        self.vars["bv_resolved_name"].set(f"Resolved object: {resolved}")

        # If batch/star tables already exist, write the active star's Rossby
        # result back into them so the exported summary stays synchronized.
        if not self.batch_star_summary_df.empty:
            key = normalize_member_key(self.vars["member_id"].get())
            mask = self.batch_star_summary_df["member_key"].astype(str) == key
            if mask.any():
                idx = self.batch_star_summary_df.index[mask][0]
                updates = {
                    "b_v": rossby.b_v, "b_mag": rossby.b_mag, "v_mag": rossby.v_mag,
                    "tau_c_days": rossby.tau_c_days, "rossby_number": rossby.rossby_number,
                    "rossby_period_source": rossby.period_source,
                    "rossby_period_days": rossby.period_days,
                    "bv_query_name": rossby.queried_name,
                    "bv_resolved_name": rossby.resolved_name,
                    "rossby_note": rossby.note,
                    "rossby_relation":rossby.relation,"rossby_input_name":rossby.input_name,
                    "rossby_input_value":rossby.input_value,"v_ks":rossby.v_ks,"ks_mag":rossby.ks_mag,
                    "teff":rossby.teff,"in_calibration":rossby.in_calibration,
                    "input_provenance":json.dumps(rossby.input_provenance),
                }
                for column, value in updates.items():
                    self.batch_star_summary_df.at[idx, column] = value
                self.batch_detailed_df, self.batch_lc_summary_df = propagate_star_rossby_to_batch_tables(
                    self.batch_detailed_df, self.batch_lc_summary_df, self.batch_star_summary_df
                )
                self._populate_batch_tree(
                    self.batch_lc_table, self.batch_lc_summary_df, self.batch_lc_columns
                )
                self._populate_batch_tree(
                    self.batch_star_table, self.batch_star_summary_df, self.batch_star_columns
                )

        self._update_summary()
        self._refresh_table()
        self.status_var.set(
            f"{rossby.relation}: {rossby.input_name}={rossby.input_value:.5g}, "
            f"tau_c={rossby.tau_c_days:.5g} d, "
            f"P={rossby.period_days:.5g} d ({rossby.period_source}), "
            f"Ro={rossby.rossby_number:.5g}. {rossby.note}"
        )

    def _update_summary(self) -> None:
        if self.period_analysis is not None:
            analysis = self.period_analysis
            self.summary_vars["acf"].set(
                f"ACF combined: {_period_text(analysis.acf_combined)} "
                f"({analysis.acf_n_used}/3)"
            )
            self.summary_vars["bgls"].set(
                f"BGLS combined: {_period_text(analysis.bgls_combined)} "
                f"({analysis.bgls_n_used}/3)"
            )
            self.summary_vars["final"].set(
                f"LC candidate ({analysis.aggregation}): {_period_text(analysis.final_averaged_prot)}"
            )

        star_row = self._star_summary_row()
        if star_row is not None:
            self.summary_vars["star_mean"].set(
                f"Star mean period: {_period_text(float(star_row.get('final_mean_period', np.nan)))}"
            )
            self.summary_vars["star_median"].set(
                f"Star median period: {_period_text(float(star_row.get('final_median_period', np.nan)))}"
            )
        elif self.period_analysis is not None and len(self.selected_lightcurve_paths()) <= 1:
            value = self.period_analysis.final_averaged_prot
            self.summary_vars["star_mean"].set(f"Star mean period: {_period_text(value)}")
            self.summary_vars["star_median"].set(f"Star median period: {_period_text(value)}")
        else:
            self.summary_vars["star_mean"].set("Star mean period: —")
            self.summary_vars["star_median"].set("Star median period: —")

        self.summary_vars["gp"].set(
            (
                f"GP MAP: {_period_text(self.result.map_period)} | "
                f"posterior median: {_period_text(self.result.period_median)} "
                f"({self.result.gp_backend} / {self.result.gp_kernel})"
            )
            if self.result is not None else "GP period: —"
        )
        if self.rossby_result is not None:
            r = self.rossby_result
            self.summary_vars["rossby"].set(
                f"Rossby ({r.relation}): {r.rossby_number:.5g} | {r.input_name}={r.input_value:.4g}, "
                f"tau_c={r.tau_c_days:.4g} d ({r.period_source})"
            )
        else:
            self.summary_vars["rossby"].set("Rossby: —")

    def _refresh_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        if self.period_analysis is None:
            self.result_table_df = pd.DataFrame()
            return
        self.result_table_df = period_analysis_dataframe(
            self.metadata_from_gui(), self.period_analysis, self.result, self.rossby_result
        )
        for _, row in self.result_table_df.iterrows():
            self.table.insert("","end",values=tuple(self._tree_value(row.get(col,"")) for col in self.table["columns"]))

    @staticmethod
    def _tree_value(value) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.10g}" if np.isfinite(value) else ""
        return str(value)

    def _populate_batch_tree(self, tree, dataframe: pd.DataFrame, columns) -> None:
        for item in tree.get_children():
            tree.delete(item)
        if dataframe.empty:
            return
        for _, row in dataframe.iterrows():
            tree.insert(
                "",
                "end",
                values=tuple(self._tree_value(row.get(column, "")) for column in columns),
            )

    def _set_batch_progress(self, completed: int, total: int) -> None:
        percent = 100.0 * completed / max(total, 1)
        self.root.after(0, lambda: self.batch_progress_var.set(percent))

    def start_batch_analysis(self, quick_acf_bgls: bool = False) -> None:
        """Analyze all selected LCs, then optionally fit GP/query B−V in a second pass.

        The two-pass design is important when the GP prior is the across-LC
        stellar mean/median: those values do not exist until every selected LC
        has first completed its ACF/BGLS analysis.
        """
        from tkinter import messagebox

        if self.worker_running:
            return
        paths = self.selected_lightcurve_paths()
        if not paths:
            messagebox.showwarning("No selected LCs", "Select one or more light curves first.")
            return
        try:
            settings = self.settings_from_gui()
        except Exception as exc:
            messagebox.showerror("Settings error", str(exc))
            return

        if quick_acf_bgls:
            run_gp = False
            query_bv = False
            period_source = "final_averaged_prot"
        else:
            run_gp = bool(self.vars["batch_run_gp"].get())
            query_bv = bool(self.vars["batch_query_bv"].get())
            period_source = "final_averaged_prot"
        gp_prior_source = self.vars["gp_prior_source"].get()
        rossby_source = self.vars["rossby_period_source"].get()
        rossby_relation = self.vars["rossby_relation"].get()
        extrapolate=bool(self.vars["allow_extrapolation"].get())
        batch_method=self.vars["method_selection"].get()
        batch_segment=self.vars["segment_selection"].get()

        if period_source.upper().startswith("GP_") and not run_gp:
            messagebox.showerror(
                "Batch settings error",
                f"{period_source} requires 'Run GP for every selected LC'.",
            )
            return
        if run_gp and settings.run_mcmc and len(paths) > 2:
            if not messagebox.askyesno(
                "Slow batch run",
                "MCMC refinement is enabled and will run for every selected LC. Continue?",
            ):
                return

        self.batch_cancel_event.clear()
        self.batch_progress_var.set(0.0)
        self.batch_detailed_df = pd.DataFrame()
        self.batch_lc_summary_df = pd.DataFrame()
        self.batch_star_summary_df = pd.DataFrame()
        self._populate_batch_tree(self.batch_detailed_table, pd.DataFrame(), self.batch_detailed_columns)
        self._populate_batch_tree(self.batch_lc_table, pd.DataFrame(), self.batch_lc_columns)
        self._populate_batch_tree(self.batch_star_table, pd.DataFrame(), self.batch_star_columns)
        self._set_busy(True)
        self.batch_cancel_button.config(state="normal")
        self.status_var.set(f"Starting batch analysis for {len(paths)} selected light curve(s)...")

        def worker():
            records: list[dict] = []
            total = len(paths)
            cancelled = False
            try:
                # Pass 1: ACF/BGLS for all LCs.
                for index, path in enumerate(paths, start=1):
                    if self.batch_cancel_event.is_set():
                        cancelled = True
                        break
                    metadata = self.infer_metadata(path)
                    self._thread_status(
                        f"Periods {index}/{total}: {metadata.member_id}, sector {metadata.sector or '?'}"
                    )
                    record = {
                        "path": path, "metadata": metadata, "analysis": None,
                        "gp_result": None, "error": "", "raw_lc": None,
                    }
                    try:
                        raw_lc = load_lightcurve(path)
                        record["raw_lc"] = raw_lc
                        _, analysis = calculate_period_analysis(
                            raw_lc, settings, segment_selection=batch_segment, method_selection=batch_method,
                            progress=lambda text, i=index, n=total, m=metadata: self._thread_status(
                                f"Periods {i}/{n}: {m.member_id} — {text}"
                            ),
                        )
                        record["analysis"] = analysis
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"[batch skip] {path}: {traceback.format_exc()}")
                    records.append(record)
                    self._set_batch_progress(index, total * (2 if run_gp else 1))

                # Use the same equal-sector hierarchy for GP priors and exports.
                _, _, reference_table = build_batch_tables(
                    records, period_source="final_averaged_prot", aggregation=settings.aggregation
                )
                star_reference = {
                    row["member_key"]: {
                        "final_mean_period": row["final_mean_period"],
                        "final_median_period": row["final_median_period"],
                    }
                    for _, row in reference_table.iterrows()
                }

                # Pass 2: optional GP. This makes star mean/median valid prior choices.
                if run_gp and not cancelled:
                    for gp_index, record in enumerate(records, start=1):
                        if self.batch_cancel_event.is_set():
                            cancelled = True
                            break
                        analysis = record.get("analysis")
                        raw_lc = record.get("raw_lc")
                        if analysis is None or raw_lc is None:
                            continue
                        metadata = record["metadata"]
                        key = normalize_member_key(metadata.member_id)
                        prior_override = None
                        if gp_prior_source in {"final_mean_period", "final_median_period"}:
                            prior_override = star_reference.get(key, {}).get(gp_prior_source, np.nan)
                            if not np.isfinite(prior_override):
                                record["error"] = (record.get("error", "") + "; " if record.get("error") else "") + \
                                    f"GP prior {gp_prior_source} unavailable"
                                continue
                        self._thread_status(
                            f"GP {gp_index}/{total}: {metadata.member_id} — {gp_prior_source}"
                        )
                        try:
                            _, gp_result = fit_rotation_period(
                                raw_lc, settings,
                                progress=lambda text, i=gp_index, n=total, m=metadata: self._thread_status(
                                    f"GP {i}/{n}: {m.member_id} — {text}"
                                ),
                                period_analysis=analysis,
                                gp_prior_source=gp_prior_source,
                                gp_prior_override=prior_override,
                            )
                            record["gp_result"] = gp_result
                        except Exception as exc:
                            record["error"] = (record.get("error", "") + "; " if record.get("error") else "") + \
                                f"GP {type(exc).__name__}: {exc}"
                            print(f"[batch GP skip] {record['path']}: {traceback.format_exc()}")
                        self._set_batch_progress(total + gp_index, total * 2)

                detailed, lc_summary, star_summary = build_batch_tables(
                    records, period_source=period_source, aggregation=settings.aggregation
                )

                if query_bv and not cancelled and not star_summary.empty:
                    n_stars = len(star_summary)
                    for star_index in range(n_stars):
                        if self.batch_cancel_event.is_set():
                            cancelled = True
                            break
                        member_id = str(star_summary.at[star_index, "member_id"])
                        key = str(star_summary.at[star_index, "member_key"])
                        group = lc_summary.loc[lc_summary["member_key"].astype(str) == key]

                        # Resolve the requested Rossby period at star level.
                        if rossby_source in {"final_mean_period", "final_median_period"}:
                            rossby_period = float(star_summary.at[star_index, rossby_source])
                        else:
                            source_to_column = {
                                "final_averaged_prot": "final_averaged_prot",
                                "ACF_combined": "acf_combined", "BGLS_combined": "bgls_combined",
                                "GP_MAP": "gp_map", "GP_median": "gp_median",
                                "ACF_full_sector": "acf_full_sector", "ACF_sectorA": "acf_sectorA",
                                "ACF_sectorB": "acf_sectorB", "BGLS_full_sector": "bgls_full_sector",
                                "BGLS_sectorA": "bgls_sectorA", "BGLS_sectorB": "bgls_sectorB",
                            }
                            source_to_column.update({f"{m}_{seg}":f"{m.lower()}_{seg}" for m in METHOD_ORDER for seg in SEGMENT_ORDER})
                            col = source_to_column.get(rossby_source)
                            vals = pd.to_numeric(group[col], errors="coerce") if col in group else pd.Series(dtype=float)
                            vals = vals[np.isfinite(vals) & (vals > 0)]
                            rossby_period = float(np.median(vals)) if len(vals) else np.nan

                        self._thread_status(f"Rossby inputs {star_index + 1}/{n_stars}: {member_id}")
                        try:
                            canonical, membership_row = match_membership_row(self.membership_df, member_id)
                            candidates = membership_query_names(canonical, membership_row)
                            inputs=query_rossby_inputs(canonical,rossby_relation,progress=self._thread_status,extra_candidates=candidates)
                            rr=calculate_rossby_number(period_source=rossby_source,analysis=None,period_override=rossby_period,
                                relation=rossby_relation,allow_extrapolation=extrapolate,**inputs)
                            updates={"b_v":rr.b_v,"b_mag":rr.b_mag,"v_mag":rr.v_mag,"tau_c_days":rr.tau_c_days,
                                     "rossby_number":rr.rossby_number,"rossby_period_source":rossby_source,
                                     "rossby_period_days":rossby_period,"bv_query_name":rr.queried_name,
                                     "bv_resolved_name":rr.resolved_name,"rossby_note":rr.note,
                                     "rossby_relation":rr.relation,"rossby_input_name":rr.input_name,
                                     "rossby_input_value":rr.input_value,"v_ks":rr.v_ks,"ks_mag":rr.ks_mag,
                                     "teff":rr.teff,"in_calibration":rr.in_calibration,
                                     "input_provenance":json.dumps(rr.input_provenance)}
                        except Exception as exc:
                            updates = {
                                "rossby_period_source": rossby_source,
                                "rossby_period_days": rossby_period,
                                "rossby_note": f"Catalog query failed: {type(exc).__name__}: {exc}",
                            }
                        for column, value in updates.items():
                            star_summary.at[star_index, column] = value

                    detailed, lc_summary = propagate_star_rossby_to_batch_tables(
                        detailed, lc_summary, star_summary
                    )

                self.root.after(
                    0, lambda: self._show_batch_result(
                        records, detailed, lc_summary, star_summary, cancelled
                    )
                )
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(
                    0, lambda exc=exc, tb=tb: self._show_worker_error("Batch-analysis error", exc, tb)
                )

        threading.Thread(target=worker, daemon=True).start()

    def cancel_batch(self) -> None:
        if self.worker_running:
            self.batch_cancel_event.set()
            self.batch_cancel_button.config(state="disabled")
            self.status_var.set(
                "Batch cancellation requested. The current calculation will finish first."
            )

    def _show_batch_result(
        self,
        records: list[dict],
        detailed: pd.DataFrame,
        lc_summary: pd.DataFrame,
        star_summary: pd.DataFrame,
        cancelled: bool,
    ) -> None:
        self._set_busy(False)
        self.batch_records = records
        self.batch_detailed_df = detailed
        self.batch_lc_summary_df = lc_summary
        self.batch_star_summary_df = star_summary
        self._populate_batch_tree(
            self.batch_detailed_table, detailed, self.batch_detailed_columns
        )
        self._populate_batch_tree(
            self.batch_lc_table, lc_summary, self.batch_lc_columns
        )
        self._populate_batch_tree(
            self.batch_star_table, star_summary, self.batch_star_columns
        )
        self._update_summary()
        if not cancelled:
            self.batch_progress_var.set(100.0)
        successful = int((lc_summary.get("status", pd.Series(dtype=str)) != "failed").sum())
        failures = int((lc_summary.get("status", pd.Series(dtype=str)) == "failed").sum())
        unique_stars = len(star_summary)
        prefix = "Batch stopped" if cancelled else "Batch complete"
        self.status_var.set(
            f"{prefix}: {successful} LC(s) processed, {failures} failed, "
            f"{len(detailed)} detailed measurement row(s), "
            f"{unique_stars} final star row(s)."
        )
        # Show the comprehensive all-measurements table first; the neighboring
        # tabs provide one row per LC and one row per star.
        if not detailed.empty:
            self.right_tabs.select(self.batch_detail_tab)
        elif not lc_summary.empty:
            self.right_tabs.select(self.batch_lc_tab)
        elif not star_summary.empty:
            self.right_tabs.select(self.batch_star_tab)

    def save_batch_tables(self) -> None:
        from tkinter import filedialog, messagebox

        if self.batch_lc_summary_df.empty and self.batch_star_summary_df.empty:
            messagebox.showwarning(
                "No batch result", "Run Analyze Selected LCs first."
            )
            return
        output_dir = filedialog.askdirectory(initialdir=self.directory or None)
        if not output_dir:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {
            "detailed measurements": os.path.join(
                output_dir, f"rotation_batch_{stamp}_detailed_measurements.csv"
            ),
            "LC summary": os.path.join(
                output_dir, f"rotation_batch_{stamp}_lightcurve_summary.csv"
            ),
            "final star summary": os.path.join(
                output_dir, f"rotation_batch_{stamp}_final_period_per_star.csv"
            ),
        }
        self.batch_detailed_df.to_csv(paths["detailed measurements"], index=False)
        self.batch_lc_summary_df.to_csv(paths["LC summary"], index=False)
        self.batch_star_summary_df.to_csv(paths["final star summary"], index=False)
        self.status_var.set(
            f"Saved three batch CSV tables in {output_dir}."
        )
        messagebox.showinfo(
            "Batch tables saved",
            "Saved:\n" + "\n".join(Path(path).name for path in paths.values()),
        )

    def _show_worker_error(self, title: str, exc: Exception, tb: str) -> None:
        from tkinter import messagebox

        self._set_busy(False)
        print(tb)
        messagebox.showerror(title, str(exc))
        self.status_var.set(f"{title}: {exc}")

    def save_figure(self) -> None:
        from tkinter import filedialog, messagebox

        if self.period_analysis is None:
            messagebox.showwarning("No result", "Calculate periods first.")
            return
        stem = safe_filename(self.metadata_from_gui().member_id)
        suffix = "gp_rotation" if self.result is not None else "period_analysis"
        path = filedialog.asksaveasfilename(
            initialdir=self.directory or None,
            initialfile=f"{stem}_{suffix}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
        )
        if path:
            self.fig.savefig(path, dpi=180, bbox_inches="tight")
            self.status_var.set(f"Saved figure: {path}")

    def _active_table_export_target(self):
        """Return (label, dataframe, suggested_filename) for the visible table tab."""
        selected = self.right_tabs.select()


        if selected == str(self.batch_detail_tab):
            return (
                "Batch detailed",
                self.batch_detailed_df,
                "rotation_batch_detailed_measurements.csv",
            )

        if selected == str(self.batch_lc_tab):
            return (
                "Batch LC summary",
                self.batch_lc_summary_df,
                "rotation_batch_lightcurve_summary.csv",
            )

        if selected == str(self.batch_star_tab):
            return (
                "Final star summary",
                self.batch_star_summary_df,
                "rotation_batch_final_period_per_star.csv",
            )

        return (None, None, None)

    def _update_export_table_button_text(self, event=None) -> None:
        if not hasattr(self, "export_table_button") or not hasattr(self, "right_tabs"):
            return
        try:
            selected = self.right_tabs.select()
            labels = {
                str(self.batch_detail_tab): "Export Batch Detailed CSV",
                str(self.batch_lc_tab): "Export Batch LC Summary CSV",
                str(self.batch_star_tab): "Export Final Star Summary CSV",
            }
            self.export_table_button.config(
                text=labels.get(selected, "Export Active Table CSV")
            )
        except Exception:
            self.export_table_button.config(text="Export Active Table CSV")

    def export_table(self) -> None:
        from tkinter import filedialog, messagebox

        label, df, initialfile = self._active_table_export_target()

        if label is None:
            messagebox.showwarning(
                "No table selected",
                "Open Batch detailed, Batch LC summary, or Final star summary, then export again.",
            )
            return

        if df is None or df.empty:
            messagebox.showwarning(
                "No table data", "Run the period/batch analysis first so this table has rows to save."
            )
            return

        path = filedialog.asksaveasfilename(
            initialdir=self.directory or None,
            initialfile=initialfile,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            df.to_csv(path, index=False)
            self.status_var.set(f"Saved {label}: {path}")

    def refresh_diagnostics(self, reset=False):
        if not hasattr(self,"diagnostic_choice"): return
        analysis=self.period_analysis
        if analysis is None or self.clean_lc is None:
            self.diagnostic_choice["values"]=()
            self.diagnostic_var.set("");self.fold_var.set("")
            self.peak_tree.delete(*self.peak_tree.get_children())
            self.diagnostic_fig.clear();self.diagnostic_canvas.draw_idle()
            return
        choices=[f"{m.method} | {m.segment}" for m in analysis.measurements.values()
                 if np.isfinite(m.period) and m.period>0]
        self.diagnostic_choice["values"]=choices
        if not choices:return
        if reset or self.diagnostic_var.get() not in choices:
            self.diagnostic_var.set(next((v for v in choices if v.startswith("GLS | full")),choices[0]))
        method,segment=self.diagnostic_var.get().split(" | ")
        m=analysis.measurement(segment,method)
        self.fold_var.set(f"{m.period:.10g}")
        self.peak_tree.delete(*self.peak_tree.get_children())
        for peak in m.peaks:
            flag=peak["harmonic_candidate"]+("; search edge" if peak["at_search_boundary"] else "")
            self.peak_tree.insert("","end",values=(peak["rank"],f"{peak['period_days']:.10g}",f"{peak['power']:.6g}",f"{peak['ratio_to_strongest']:.4g}",flag))
        self.diagnostic_note.set(m.note+". Harmonic flags use a 5% period-ratio tolerance; they do not establish the physical rotation period.")
        self.draw_diagnostics()

    def draw_diagnostics(self):
        from tkinter import messagebox
        if self.period_analysis is None or self.clean_lc is None:return
        try:
            method,segment=self.diagnostic_var.get().split(" | ")
            m=self.period_analysis.measurement(segment,method)
            draw_peak_phase_figure(self.diagnostic_fig,self.clean_lc,m,float(self.fold_var.get()))
            self.diagnostic_canvas.draw_idle()
        except Exception as exc:messagebox.showerror("Phase-fold settings",str(exc))

    def select_peak(self, event=None):
        selection=self.peak_tree.selection()
        if selection:
            self.fold_var.set(self.peak_tree.item(selection[0],"values")[1])
            self.draw_diagnostics()

    def export_diagnostic(self):
        from tkinter import filedialog,messagebox
        if self.period_analysis is None or self.clean_lc is None:
            messagebox.showwarning("No analysis","Run a period search first.");return
        try:
            method,segment=self.diagnostic_var.get().split(" | ")
            period=float(self.fold_var.get())
            if not np.isfinite(period) or period<=0:raise ValueError("Fold period must be positive")
            path=filedialog.asksaveasfilename(defaultextension=".png",initialfile=f"{safe_filename(self.clean_lc.name)}_{method}_{segment}_phase.png",filetypes=[("PNG","*.png")])
            if not path:return
            self.draw_diagnostics();self.diagnostic_fig.savefig(path,dpi=180,bbox_inches="tight")
            base=str(Path(path).with_suffix(""))
            peak_dataframe(self.period_analysis).to_csv(base+"_peaks.csv",index=False)
            segment_lc=split_lightcurve_halves(self.clean_lc)[segment]
            folded_dataframe(segment_lc,period,epoch=float(self.clean_lc.time[0])).to_csv(base+"_folded.csv",index=False)
            self.status_var.set(f"Saved diagnostic PNG, ranked peaks, and folded samples: {base}")
        except Exception as exc:messagebox.showerror("Export diagnostic",str(exc))


    def export_json(self) -> None:
        from tkinter import filedialog, messagebox

        if self.period_analysis is None or self.clean_lc is None or self.last_settings is None:
            messagebox.showwarning("No result", "Run a period search first.")
            return
        metadata = self.metadata_from_gui()
        path = filedialog.asksaveasfilename(
            initialdir=self.directory or None,
            initialfile=f"{safe_filename(metadata.member_id)}_acf_bgls_gp_rotation.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            payload = analysis_json(self.clean_lc,self.period_analysis,self.last_settings,self.rossby_result,metadata)
            if self.result is not None:
                payload["gp"]=result_to_json_dict(self.clean_lc,self.result,self.last_settings,metadata,self.rossby_result)
            payload["inspected_fold_period_days"]=self.fold_var.get()
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(strict_json(payload), handle, indent=2, allow_nan=False)
            self.status_var.set(f"Saved result: {path}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def folded_dataframe(lc, period, epoch=None):
    period=float(period)
    if not np.isfinite(period) or period<=0:raise ValueError("Fold period must be finite and positive")
    epoch=float(lc.time[0]) if epoch is None else float(epoch)
    return pd.DataFrame({"time_days":lc.time,"flux":lc.flux,"flux_err":lc.flux_err,
        "phase_Phalf":((lc.time-epoch)/(period/2))%1,
        "phase_P":((lc.time-epoch)/period)%1,
        "phase_Pdouble":((lc.time-epoch)/(2*period))%1,
        "fold_period_days":period,"epoch_days":epoch})


def run_cli(args) -> int:
    settings = FitSettings(
        pmin=args.pmin,pmax=args.pmax,clip_upper=args.clip_upper,clip_lower=args.clip_lower,
        gp_points=args.gp_points,chunk_size=args.chunk_size,optimizer_starts=args.optimizer_starts,
        use_paper_hyperpriors=args.paper_priors,run_mcmc=args.mcmc,mcmc_walkers=args.walkers,
        mcmc_steps=args.steps,mcmc_burn=args.burn,random_seed=args.seed,
        posterior_scan_points=args.scan_points,prediction_train_points=args.prediction_train_points,
        prediction_points=args.prediction_points,bgls_bin_minutes=args.bgls_bin_minutes,
        bgls_max_steps=args.bgls_max_steps,rotation_smooth_hours=args.smooth_hours,
        gp_backend="exoplanet / celerite2" if args.gp_backend=="celerite2" else "GProtation (Angus+17)",
        celerite_kernel=args.celerite_kernel,celerite_max_points=args.celerite_max_points,
        aggregation=args.aggregation,peak_count=args.top_peaks)
    raw=load_lightcurve(args.file)
    clean,analysis=calculate_period_analysis(raw,settings,args.segment,args.period_method,print)
    result=None
    if args.run_gp:
        clean,result=fit_rotation_period(raw,settings,progress=print,period_analysis=analysis,gp_prior_source=args.gp_prior_source)
    metadata=infer_lc_metadata(args.file)
    if args.member_id: metadata.member_id=args.member_id
    if args.sector is not None:metadata.sector=str(args.sector)
    if args.lc_source:metadata.lc_source=args.lc_source
    rossby=None
    if args.calculate_rossby:
        if args.query_bv or args.query_photometry:
            inputs=query_rossby_inputs(metadata.member_id,args.rossby_relation,progress=print)
        else:
            inputs={"b_v":args.b_v if args.b_v is not None else np.nan,
                    "v_ks":args.v_ks if args.v_ks is not None else np.nan,
                    "teff":args.teff if args.teff is not None else np.nan,
                    "queried_name":"manual","resolved_name":metadata.member_id}
        rossby=calculate_rossby_number(period_source=args.rossby_period_source,analysis=analysis,gp_result=result,
                  relation=args.rossby_relation,allow_extrapolation=args.allow_extrapolation,**inputs)
    matplotlib.use("Agg",force=True)
    from matplotlib import pyplot as plt
    outbase=args.output or str(Path(args.file).with_suffix(""))+"_ghost"
    Path(outbase).parent.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(12,8));draw_period_analysis_figure(fig,clean,analysis)
    fig.savefig(outbase+".png",dpi=160,bbox_inches="tight");plt.close(fig)
    payload=analysis_json(clean,analysis,settings,rossby,metadata)
    if result is not None:
        payload["gp"]=result_to_json_dict(clean,result,settings,metadata,rossby)
        fig=plt.figure(figsize=(12,8));draw_fit_figure(fig,clean,result,settings)
        fig.savefig(outbase+"_gp.png",dpi=160,bbox_inches="tight");plt.close(fig)
    choices=[m for m in analysis.measurements.values() if np.isfinite(m.period) and m.period>0]
    if choices:
        if args.fold_source:
            method,segment=args.fold_source.split("_",1)
            chosen=analysis.measurement(segment,method)
            if chosen is None or not np.isfinite(chosen.period):raise ValueError("Requested fold source is unavailable")
        else:chosen=next((m for m in choices if m.method=="GLS" and m.segment=="full_sector"),choices[0])
        period=chosen.period if args.fold_period is None else args.fold_period
        fig=plt.figure(figsize=(12,8));draw_peak_phase_figure(fig,clean,chosen,period)
        fig.savefig(outbase+"_phase.png",dpi=160,bbox_inches="tight");plt.close(fig)
        folded_dataframe(split_lightcurve_halves(clean)[chosen.segment],period,epoch=clean.time[0]).to_csv(outbase+"_folded.csv",index=False)
        payload["phase_fold"]={"period_days":period,"source":f"{chosen.method}_{chosen.segment}","epoch_days":float(clean.time[0]),"manual_override":args.fold_period is not None}
    elif args.fold_period is not None:raise ValueError("No valid measurement to inspect")
    with open(outbase+".json","w",encoding="utf-8") as handle:json.dump(strict_json(payload),handle,indent=2,allow_nan=False)
    period_analysis_dataframe(metadata,analysis,result,rossby).to_csv(outbase+".csv",index=False)
    peak_dataframe(analysis).to_csv(outbase+"_peaks.csv",index=False)
    print(f"Candidate ({analysis.aggregation}): {analysis.final_averaged_prot:.8g} d; outputs: {outbase}*")
    if rossby:print(f"{rossby.relation}: tau={rossby.tau_c_days:g} d; Ro={rossby.rossby_number:g}; {rossby.note}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GHOST v0.2: ACF/BGLS/LS/GLS/FFT, ranked peaks, phase folds, optional GP and Rossby"
    )
    parser.add_argument("--file", help="FITS/CSV/TXT light curve")
    parser.add_argument("--no-gui", action="store_true", help="Run command-line mode")
    parser.add_argument("--output", help="Output basename without extension")
    parser.add_argument("--pmin", type=float, default=0.1)
    parser.add_argument("--pmax", type=float, default=13.0)
    parser.add_argument("--clip-upper", type=float, default=5.0)
    parser.add_argument("--clip-lower", type=float, default=8.0)
    parser.add_argument(
        "--segment",
        choices=("all", "full_sector", "sectorA", "sectorB"),
        default="all",
    )
    parser.add_argument(
        "--period-method", default="all", help="all, both (ACF+BGLS), or comma-separated ACF,BGLS,LS,GLS,FFT"
    )
    parser.add_argument("--aggregation", choices=("median","mean"),default="median")
    parser.add_argument("--top-peaks",type=int,default=5)
    parser.add_argument("--run-gp",action="store_true",help="Opt in to the slower GP fit")
    parser.add_argument("--fold-period",type=float,help="Manual diagnostic period; does not change measured periods")
    parser.add_argument("--fold-source",choices=tuple(f"{m}_{seg}" for m in METHOD_ORDER for seg in SEGMENT_ORDER))
    parser.add_argument("--rossby-relation",choices=ROSSBY_RELATIONS,default="Noyes1984")
    parser.add_argument("--v-ks",type=float,help="Johnson V minus verified 2MASS Ks")
    parser.add_argument("--teff",type=float,help="Effective temperature in K for Cranmer2011")
    parser.add_argument("--query-photometry",action="store_true",help="Query the inputs required by the selected Rossby relation")
    parser.add_argument("--allow-extrapolation",action="store_true")
    parser.add_argument("--bgls-bin-minutes", type=float, default=30.0)
    parser.add_argument("--bgls-max-steps", type=int, default=3000)
    parser.add_argument("--smooth-hours", type=float, default=6.0)
    parser.add_argument(
        "--gp-prior-source",
        choices=("ACF_combined", "BGLS_combined", "final_averaged_prot",
                 "final_mean_period", "final_median_period"),
        default="final_averaged_prot",
    )
    parser.add_argument("--gp-backend", choices=("gprotation", "celerite2"), default="gprotation")
    parser.add_argument("--celerite-kernel", choices=CELERITE_KERNELS, default="RotationTerm")
    parser.add_argument("--celerite-max-points", type=int, default=5000, help="0 uses all clean points")
    parser.add_argument("--gp-points", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--optimizer-starts", type=int, default=10)
    parser.add_argument("--paper-priors", action="store_true")
    parser.add_argument("--mcmc", action="store_true")
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--burn", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-points", type=int, default=120)
    parser.add_argument("--prediction-train-points", type=int, default=500)
    parser.add_argument("--prediction-points", type=int, default=1200)
    parser.add_argument("--member-id")
    parser.add_argument("--sector")
    parser.add_argument("--lc-source")
    parser.add_argument(
        "--calculate-rossby",
        action="store_true",
        help="Calculate tau_c and Rossby number using the chosen relation",
    )
    parser.add_argument("--b-v", type=float, help="Manual Johnson B-V color")
    parser.add_argument(
        "--query-bv",
        action="store_true",
        help="Query SIMBAD for Johnson B and V using the inferred/member ID",
    )
    parser.add_argument(
        "--rossby-period-source",
        choices=ROSSBY_PERIOD_SOURCES,
        default="final_averaged_prot",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.no_gui or args.file:
        if not args.file:
            parser.error("--file is required in command-line mode")
        return run_cli(args)

    import tkinter as tk

    root = tk.Tk()
    RotationGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
