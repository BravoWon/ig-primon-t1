# T1_precision_map_v0_2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the software to support the canonical locked pre-reg `t1_precision_map_v0_2.md` (from main docs; worktree copy in root): the four receipts (precision_recursion_gate.py for [GATE] derivation, precision_depth_map.py for certified depth map + controls/falsifiers, precision_allocator.py for the shippable LAMP-style allocator on certified ref, allocator_bakeoff.py for bakeoff vs uniform/LAMP). Start with skeleton/harness per stages, TDD, reuse ig_primon (firewall as certified ref engine), following the plan, pre-reg walls, and design (Approach 1 fidelity).

**Architecture:** Extend the existing precision and firewall infrastructure (Tier-E explorer proposes, Tier-C mpmath certifies) to support full forward-pass error composition tracking. Add a new top-level receipt module (following the pattern of module_L_* and module_hw_firewall.py) that can run a model in reduced precision (bf16/fp8/FP4), accumulate per-block errors per the frozen recursion, and support subsampled certification. Add anchors for the key [V] metrics and falsifier checks. Reuse hardware scan, precision matrix primitives, and the Precision–Certification Firewall. No changes to existing [V] receipts. The pre-reg T1_precision_map_v0_2.md is the immutable spec.

**Tech Stack:** Python 3.11+, numpy, scipy, mpmath (Tier-C), torch (for model execution and low-prec), cupy optional for GPU explorer. Follow existing patterns in ig_primon/ and top-level modules. Use subprocess for guard-less receipts where appropriate.

---

### Task 1: Project Setup and File Structure Confirmation

**Files:**
- Review: T1_precision_map_v0_2.md (locked spec)
- Review: docs/superpowers/specs/2026-06-16-t1-precision-map-v0-2-design.md (approved design)
- Create: docs/superpowers/plans/2026-06-16-t1-precision-map-v0-2-implementation.md (this plan)

- [ ] **Step 1: Verify the locked pre-registration and design doc are present and committed**
  ```bash
  git log --oneline -5
  ls T1_precision_map_v0_2.md
  ls docs/superpowers/specs/2026-06-16-t1-precision-map-v0-2-design.md
  ```
  Expected: The two files exist and latest commits reference the pre-reg and design.

- [ ] **Step 2: Confirm existing patterns by inspecting similar files**
  ```bash
  head -50 ig_primon/precision.py
  head -30 module_hw_firewall.py
  ls ig_primon/*.py
  ```
  Expected: See the precision matrix, firewall structure, and how receipts are wrapped without editing the core logic.

- [ ] **Step 3: Create the plans directory (if not already) and ensure .gitignore if needed**
  (Already handled; confirm no large binaries are tracked.)

- [ ] **Step 4: Commit any setup changes**
  ```bash
  git add docs/superpowers/plans/
  git commit -m "docs: add implementation plan skeleton for T1_precision_map_v0_2"
  ```

### Task 2: Design the New Receipt Module Structure

**Files:**
- Create: module_T1_precision_depthN.py (the main receipt, analogous to module_L_perceptron_finiteT.py or module_hw_firewall.py)
- Modify: ig_primon/cli.py (add entry for the new receipt if needed for `igprimon run`)

- [ ] **Step 1: Write a minimal skeleton for the receipt and a failing test for import**
  Create module_T1_precision_depthN.py with basic structure and a guard for direct run.

  ```python
  """T1_precision_map_v0_2 depth-N error composition receipt.

  Implements the locked pre-registration: certified error curves through depth
  on small decoder-only models using the Precision-Certification Firewall.
  """

  from __future__ import annotations

  import sys
  from typing import Any

  def main() -> int:
      """Entry point when run as `python -m module_T1_precision_depthN`."""
      print("T1_precision_map_v0_2 receipt skeleton")
      # TODO: implement per plan tasks
      return 0

  if __name__ == "__main__":
      raise SystemExit(main())
  ```

  Write a test in tests/test_precision_depthN.py (create the file).

  ```python
  def test_receipt_imports():
      import module_T1_precision_depthN
      assert hasattr(module_T1_precision_depthN, "main")
  ```

- [ ] **Step 2: Run the test to see it fail (import error or no test yet)**
  ```bash
  pytest tests/test_precision_depthN.py -v --tb=short
  ```
  Expected: FAIL (file or test not found or import error).

- [ ] **Step 3: Make the test pass by creating the test file with the import test**
  (The skeleton above should make the import test pass once the py file is there.)

- [ ] **Step 4: Run test to confirm PASS**
  ```bash
  pytest tests/test_precision_depthN.py::test_receipt_imports -v
  ```
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add module_T1_precision_depthN.py tests/test_precision_depthN.py
  git commit -m "feat: add skeleton for T1_precision_depthN receipt and basic import test"
  ```

### Task 3: Add the Error Recursion Accumulator (Core of the Frozen H1)

**Files:**
- Modify: module_T1_precision_depthN.py (add the recursion logic)
- Create: tests/test_precision_depthN.py (add tests for the accumulator)

- [ ] **Step 1: Add the per-block recursion function and a failing test for it**
  In module_T1_precision_depthN.py, add:

  ```python
  import numpy as np
  from mpmath import mp

  def compute_block_error(
      prev_error: np.ndarray,
      block_jacobian_approx: np.ndarray,
      local_delta: np.ndarray,
  ) -> np.ndarray:
      """Implements ε_{l+1} = (I + J_f) ε_l + δ_l per the locked pre-reg."""
      return prev_error + block_jacobian_approx @ prev_error + local_delta
  ```

  Add to test file:

  ```python
  def test_block_error_recursion():
      prev = np.array([0.1, 0.2])
      J = np.eye(2) * 0.5
      delta = np.array([0.01, 0.02])
      result = compute_block_error(prev, J, delta)
      expected = prev + J @ prev + delta
      np.testing.assert_allclose(result, expected)
  ```

- [ ] **Step 2: Run to verify it fails (function not in module or test fails)**
  ```bash
  pytest tests/test_precision_depthN.py::test_block_error_recursion -v
  ```
  Expected: FAIL with AttributeError or assertion fail.

- [ ] **Step 3: Implement the function in the receipt and export it**
  Update the function to be more complete (support for subsampling later), make the test pass.

- [ ] **Step 4: Run test**
  ```bash
  pytest tests/test_precision_depthN.py::test_block_error_recursion -v
  ```
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add module_T1_precision_depthN.py tests/test_precision_depthN.py
  git commit -m "feat: implement frozen per-block error recursion with test"
  ```

### Task 4: Integrate with Existing Firewall and Hardware for Certification

**Files:**
- Modify: module_T1_precision_depthN.py (add explorer/certify loop using ig_primon.firewall and hardware)
- Modify: ig_primon/anchors.py (add anchors for C1, C2, depth error metrics)
- Modify: ig_primon/harness.py (if needed for new group)
- Test: tests/test_precision_depthN.py (add certification tests)

- [ ] **Step 1: Add a function that runs a tiny model forward in low prec and certifies error vs mpmath reference. Write failing test.**
  Sketch in receipt:

  ```python
  from ig_primon.firewall import run_firewall
  from ig_primon.hardware import scan

  def run_depth_error_map(model_name: str = "gpt2-small", prec: str = "bf16") -> dict:
      dm = scan()
      # Use dm.tier_e_backend to choose explorer
      # For now, stub that calls firewall for a block error
      res = run_firewall(kappa=0.0, backend=dm.tier_e_backend)
      return {"firewall": res, "device": dm}
  ```

  Test:

  ```python
  def test_depth_map_uses_firewall():
      result = run_depth_error_map()
      assert "firewall" in result
  ```

- [ ] **Step 2: Run test (expect failure on missing integration)**
  ```bash
  pytest tests/test_precision_depthN.py::test_depth_map_uses_firewall -v
  ```

- [ ] **Step 3: Implement minimal integration, reusing existing precision matrix primitives for the block ops.**
  Fill in the function to actually compute a small chain of errors using torch low prec vs mpmath for one block.

- [ ] **Step 4: Run and pass the test**
- [ ] **Step 5: Add basic anchor in anchors.py for C1 (identity)**
- [ ] **Step 6: Run full verify to check anchor**
  ```bash
  python -m ig_primon.harness --group precision-depth
  ```
- [ ] **Step 7: Commit**

### Task 5: Implement C2 (Random Weights Exponential Reproduction)

**Files:**
- Modify: module_T1_precision_depthN.py (add random weight model runner that reproduces the published exponential on synthetic data)
- Test: tests/test_precision_depthN.py (test that C2 reproduces the known bound within tolerance)

- [ ] **Step 1: Add the random-weight depth-N runner and failing test that checks exponential growth**
  The test should assert that on random weights the certified error grows exponentially (matching Budzinskiy numbers within the pre-reg tolerance).

- [ ] **Step 2: Run to fail**
- [ ] **Step 3: Implement using the recursion + random J_f matrices**
- [ ] **Step 4: Pass the test (this is the hard gate for Stage 1)**
- [ ] **Step 5: Commit**

### Task 6: Add CLI and Harness Integration

**Files:**
- Modify: ig_primon/cli.py (add "depth-map" to RECEIPTS and command)
- Modify: ig_primon/anchors.py (register new anchors with status [GATE] until C2 passes)

- [ ] **Step 1: Update RECEIPTS dict and parser**
- [ ] **Step 2: Write failing test for `igprimon run depth-map --help` or direct call**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Test `igprimon verify --group precision-depth`**
- [ ] **Step 5: Commit**

### Task 7: Full Controls, C3/C4, and Basic Depth Map on Trained Weights

**Files:**
- Extend module_T1_precision_depthN.py with shuffle control (C3) and primitive isolation (C4)
- Add anchors for the main [V] depth-error curve metrics
- Update tests

- [ ] Steps for each control + the first trained-weight (tiny) depth curve
- Ensure F3 path (range vs mantissa) is instrumented
- Commit after each control passes

### Task 8: Documentation and Final Polish

**Files:**
- Update README.md with the new igprimon command
- Add note in T1_precision_map_v0_2.md that the software is now available (post-lock)
- Ensure all anchors have proper slow/ status tags

- [ ] Run full `igprimon verify`
- [ ] Run `igprimon hwscan`
- [ ] Commit

### Task 9: Self-Review and Hand-off

- [ ] Run the full plan self-review checklist against the locked pre-reg and design doc
- [ ] Ensure no placeholders, exact paths, TDD followed
- [ ] Final commit
- [ ] Announce plan complete

**Plan complete and saved to `docs/superpowers/plans/2026-06-16-t1-precision-map-v0-2-implementation.md`.**

Two execution options:
1. Subagent-Driven (recommended) - dispatch fresh subagent per task with review.
2. Inline Execution - execute tasks in this session using executing-plans.

Which approach? (If subagent, use the subagent-driven-development skill.)