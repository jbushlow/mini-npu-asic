"""Generate placeholder inputs for the core smoke testbench.

The first core PNR bring-up test only checks that the gate-level simulation can
reset and run a few idle cycles. Functional program vectors can replace this
once the core-level workload testbench is stable.
"""

from pathlib import Path

Path("test_vectors.txt").write_text("", encoding="ascii")
Path("design.args").write_text("", encoding="ascii")
