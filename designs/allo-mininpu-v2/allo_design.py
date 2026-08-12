"""Expose the rebuilt MiniNPU v2 through the Allo ASIC-flow contract."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "allo-rebuild" / "mininpu_v2.py"
SPEC = spec_from_file_location("allo_mininpu_v2_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load MiniNPU v2 source: {SOURCE}")

_source = module_from_spec(SPEC)
SPEC.loader.exec_module(_source)

# These names are the public design interface consumed by the compilation node
# and by future backend-independent workload/testbench generation.
mininpu_v2 = _source.mininpu_v2
build = _source.build
workload = _source.workload
run_workload = _source.run_workload

