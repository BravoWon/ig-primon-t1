import numpy as np
import module_T1_precision_depthN

def test_receipt_imports():
    assert hasattr(module_T1_precision_depthN, "main")

def test_block_error_recursion():
    prev = np.array([0.1, 0.2])
    J = np.eye(2) * 0.5
    delta = np.array([0.01, 0.02])
    result = module_T1_precision_depthN.compute_block_error(prev, J, delta)
    expected = prev + J @ prev + delta
    np.testing.assert_allclose(result, expected)

def test_depth_map_uses_firewall():
    result = module_T1_precision_depthN.run_depth_error_map()
    assert "firewall" in result

def test_c2_random_weight_growth():
    """C2 stub: on random weights, error norms grow over depth (demo of exponential trend)."""
    norms = module_T1_precision_depthN.simulate_random_weight_depth(L=8, dim=4)
    # Check growth: last norm > first * some factor (for demo, > 1.1x or positive log slope)
    growth_factor = norms[-1] / norms[0] if norms[0] != 0 else 1
    assert growth_factor > 1.0, f"No growth: {growth_factor}"
    # Rough exponential check: log norms increase
    import numpy as np
    logs = np.log(np.array(norms) + 1e-20)
    slope = (logs[-1] - logs[0]) / (len(logs) - 1)
    assert slope > 0, f"No positive slope in log-error: {slope}"

def test_c3_shuffle():
    ok, msg = module_T1_precision_depthN.c3_shuffle_control([], [])
    assert ok

def test_c4_primitive():
    ok, _ = module_T1_precision_depthN.c4_primitive_isolation(np.zeros(2))
    assert ok

def test_trained_stub_curve():
    curve = module_T1_precision_depthN.run_depth_map_on_trained_stub()
    assert len(curve) == 12
    assert curve[0] < 1e-9  # starts small
    # sub-exp growth
    assert curve[-1] < curve[0] * 2  # near linear-ish for stub
