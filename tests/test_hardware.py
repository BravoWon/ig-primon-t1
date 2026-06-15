"""Hardware-scan and firewall smoke tests (CPU path — no GPU required for CI)."""

from ig_primon.hardware import scan
from ig_primon.firewall import run_firewall, certify_alpha_c


def test_scan_returns_device_map():
    dm = scan()
    assert dm.tier_c, "Tier-C must always be assigned (the CPU certifier)"
    assert dm.tier_e_backend in ("cuda", "cpu-fp32")


def test_tier_c_certifies_gardner_capacity():
    # the certifier alone must reproduce alpha_c(0) = 2 exactly
    assert abs(float(certify_alpha_c(0.0, dps=40)) - 2.0) < 1e-12


def test_firewall_has_precision_teeth_on_cpu():
    # force the CPU FP32 explorer so this runs anywhere.
    res = run_firewall(kappa=0.0, backend="cpu")
    honest, near, gross = res.kernels
    # honest: within the float32 noise floor -> Tier-C certifies it
    assert honest.certify_pass, "honest FP32 kernel must certify (deviation within FP32 noise floor)"
    # near-miss: invisible to an FP32-grade tolerance, but its error exceeds the noise floor ...
    assert near.explore_pass, "near-miss should pass the loose FP32-grade tolerance (it is camouflaged)"
    assert not near.certify_pass, "near-miss must be REJECTED by Tier-C (error exceeds FP32 noise floor)"
    assert near.eps_multiple > 2.0, "near-miss must sit above the float32 noise floor to be a real error"
    # gross: visibly wrong, fails even the loose tolerance
    assert not gross.explore_pass, "gross kernel must fail even the FP32-grade tolerance"
    # the load-bearing claim: FP32 agreement cannot catch what dps>=50 catches
    assert res.precision_teeth and res.firewall_intact
