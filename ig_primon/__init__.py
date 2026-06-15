"""IG-PRIMON-T1 — operational layer over the research receipts.

The experimental program lives in the top-level ``module_*.py`` receipts and the
markdown ledgers. This package ports them to an operational footing WITHOUT editing
them (the program's no-silent-edit discipline): it adds

  * ``ig_primon.anchors``  — the exact reference values each receipt pins, as callable checks
  * ``ig_primon.harness``  — run the anchors, report pass/fail (the ``verify --all`` engine)
  * ``ig_primon.hardware`` — scan the real device, assign the Tier-C / Tier-E split
  * ``ig_primon.firewall`` — the Precision–Certification Firewall (Tier-E proposes, Tier-C certifies)
  * ``ig_primon.cli``      — the ``igprimon`` command-line entry point

Nothing here may award a ``[V]`` tag that the underlying receipt did not already earn;
this layer only re-checks the anchors programmatically and reports drift.
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
