# IG-PRIMON-T1 operational targets. On Windows use the documented `igprimon` /
# `python -m pytest` commands directly if `make` is unavailable.

PY ?= python

.PHONY: install install-gpu verify verify-quick test test-quick hwscan firewall clean

install:        ## editable install with the certification stack (numpy/scipy/mpmath)
	$(PY) -m pip install -e ".[dev]"

install-gpu:    ## also install the CUDA FP32 Tier-E explorer (CuPy + CUDA headers)
	$(PY) -m pip install -e ".[dev,gpu]" && $(PY) -m pip install "cupy-cuda12x[ctk]"

verify:         ## run every anchor (Tier-C / CPU) — the operational acceptance gate
	$(PY) -m ig_primon.cli verify

verify-quick:   ## skip the slow high-precision anchors
	$(PY) -m ig_primon.cli verify --quick

test:           ## full pytest (anchors + hardware)
	$(PY) -m pytest

test-quick:     ## pytest without the slow-marked anchors
	$(PY) -m pytest -m "not slow"

hwscan:         ## scan the device and print the Tier-C / Tier-E map
	$(PY) -m ig_primon.cli hwscan

firewall:       ## run the Precision-Certification Firewall (GPU Tier-E if available)
	$(PY) -m ig_primon.cli firewall

clean:
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['build','dist','.pytest_cache']+glob.glob('*.egg-info')]"
