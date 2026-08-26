"""Compute light-curve quality metrics and science-specific usability flags.

Candidate metrics:
- duty cycle
- NaN/masked fraction
- cadence
- observing baseline
- large gaps
- discontinuities
- robust scatter
- saturation/crowding indicators
- flat/noisy behavior

Later, separate flags may indicate whether a curve is suitable for rotation,
flare detection, or other analyses.
"""
