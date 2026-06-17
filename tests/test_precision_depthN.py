def test_receipt_imports():
    import module_T1_precision_depthN
    assert hasattr(module_T1_precision_depthN, "main")


def test_block_error_recursion():
    import numpy as np
    from module_T1_precision_depthN import compute_block_error
    prev = np.array([0.1, 0.2])
    J = np.eye(2) * 0.5
    delta = np.array([0.01, 0.02])
    result = compute_block_error(prev, J, delta)
    expected = prev + J @ prev + delta
    np.testing.assert_allclose(result, expected)


def test_depth_map_uses_firewall():
    from module_T1_precision_depthN import run_depth_error_map
    result = run_depth_error_map()
    assert "firewall" in result


def test_c2_random_weights_reproduces_exponential():
    """C2 HARD GATE test per pre-reg and plan Task 5.
    On random weights (Budzinskiy regime d=n=20, L=40), the certified error
    via the recursion must grow exponentially: positive log-slope and median << mean.
    Matches 'Budzinskiy numbers' within the pre-reg tolerance (slope>0.05, mm@L20>5
    as used in the reference C2 gate logic).
    """
    import numpy as np
    from module_T1_precision_depthN import run_c2_random_weight_depthN
    res = run_c2_random_weight_depthN(d=20, L=40, n_samples=64, seed=20260616)
    slope = res["slope"]
    mm_L20 = res.get("mean_over_med_L20", res.get("mm_L20", 0.0))
    growth = res.get("growth", 0.0)
    # Assert exponential mean growth
    assert slope > 0.05, f"C2 failed: log-slope {slope:+.3f} not >0.05 (no exp growth)"
    # median << mean (heavy tail, Budzinskiy signature)
    assert mm_L20 > 5.0, f"C2 failed: mean/median at L20 {mm_L20:.1f} not >>1 (no heavy tail)"
    # Rough match to observed Budzinskiy reproduction numbers within tolerance (e.g. growth factor)
    # observed ~7e4 x at L40 for the regime; allow loose tolerance for synthetic J model
    assert growth > 100.0, f"C2 failed: end growth {growth:.2g}x too small for exp regime"
    # Mark as reproduced within tolerance
    assert res.get("reproduced", False) or (slope > 0.05 and mm_L20 > 5), "C2 did not reproduce within tolerance"


def test_cli_run_depth_map_direct_or_help():
    """Failing test (Step 2) for igprimon run depth-map (or via direct main call).
    Per Task 6: exercises that 'depth-map' is registered in RECEIPTS so parser accepts it
    and _cmd_run can invoke the receipt (python -m module_T1_precision_depthN).
    Test written before final registration to demonstrate failure -> implement flow.
    Uses direct main(argv=...) call (avoids needing shell 'igprimon' and --help edge cases).
    """
    from ig_primon.cli import main, RECEIPTS
    # direct evidence of registration
    assert "depth-map" in RECEIPTS, "depth-map must be in RECEIPTS for CLI integration"
    # direct call: should succeed (return 0) once registered; will raise/return 2 on unknown now
    rc = main(["run", "depth-map"])
    assert rc == 0, f"igprimon run depth-map via direct call should exit 0, got {rc}"
