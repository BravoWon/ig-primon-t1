def test_receipt_imports():
    import module_T1_precision_depthN
    assert hasattr(module_T1_precision_depthN, "main")


def test_block_error_recursion():
    import numpy as np
    import module_T1_precision_depthN as mod
    prev = np.array([0.1, 0.2])
    J = np.eye(2) * 0.5
    delta = np.array([0.01, 0.02])
    result = mod.compute_block_error(prev, J, delta)
    expected = prev + J @ prev + delta
    np.testing.assert_allclose(result, expected)
