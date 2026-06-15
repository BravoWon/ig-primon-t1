"""Ensure the project root is importable so the top-level receipt modules
(module_L_*, module_e_*, audit_independent, module_hw_firewall) resolve during tests
even without an editable install."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
