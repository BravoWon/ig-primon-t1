def test_receipt_imports():
    import module_T1_precision_depthN
    assert hasattr(module_T1_precision_depthN, "main")
