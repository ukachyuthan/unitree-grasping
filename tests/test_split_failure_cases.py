"""Demo / held-out split of failure cases: stratified by family, deterministic, disjoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split_failure_cases import split_cases  # noqa: E402


def test_split_is_stratified_disjoint_and_deterministic():
    cases = {f"spoon.v{i % 3:03d}__q{i}": f"spoon.v{i % 3:03d}" for i in range(10)}
    cases |= {f"bowl__q{i}": "bowl" for i in range(4)}
    cases["torus__q0"] = "torus"
    demo, held = split_cases(cases, 0.5, seed=0)
    assert not set(demo) & set(held) and set(demo) | set(held) == set(cases)
    assert sum(c.startswith("spoon") for c in held) == 5      # variants pooled into one family
    assert sum(c.startswith("bowl") for c in held) == 2
    assert "torus__q0" in demo                                # a lone case is always demonstrable
    assert split_cases(cases, 0.5, seed=0) == (demo, held)
