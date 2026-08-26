"""Resolve target inputs to persistent astronomical identifiers.

Planned responsibilities:
- Accept RA/Dec, TIC IDs, Gaia IDs, or target names.
- Handle coordinate epochs and proper motion where required.
- Crossmatch against Gaia/TIC/SIMBAD or other services.
- Return a normalized internal Star record.

Important: RA/Dec should be treated as a lookup input, not the sole permanent ID.
"""
