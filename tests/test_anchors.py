"""Anchor regression tests — every receipt's pinned reference value must still reproduce.

The quick anchors run on every CI push; the slow (high-precision) ones are marked and can
be deselected with ``-m 'not slow'``.
"""

import pytest

from ig_primon.anchors import get_anchors


def _ids(specs):
    return [s.id for s in specs]


QUICK = get_anchors(include_slow=False)
SLOW = [s for s in get_anchors(include_slow=True) if s.slow]


@pytest.mark.parametrize("spec", QUICK, ids=_ids(QUICK))
def test_quick_anchor(spec):
    r = spec.run()
    assert r.ok, f"{r.id} drifted: {r.value} vs {r.expected} | {r.detail} {r.error}"


@pytest.mark.slow
@pytest.mark.parametrize("spec", SLOW, ids=_ids(SLOW))
def test_slow_anchor(spec):
    r = spec.run()
    assert r.ok, f"{r.id} drifted: {r.value} vs {r.expected} | {r.detail} {r.error}"
