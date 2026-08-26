"""Very small end-to-end smoke test used while the repository is young."""

from stellar_lc_platform import __version__


if __name__ == "__main__":
    print(f"stellar-lc-platform {__version__} imports successfully")
