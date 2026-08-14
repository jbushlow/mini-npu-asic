"""Labmate blocking EVA design through the Catapult-backed ASIC flow."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EVA_DIR = THIS_DIR.parent / "allo-eva-rebuild"
REFERENCE_CONSTRUCT = EVA_DIR / "construct-commercial.py"
EVA_SOURCE = EVA_DIR / "eva_sb_syscredit_rtprime_skid_leanalu.py"


def _load_reference_construct():
    spec = spec_from_file_location(
        "allo_eva_catapult_reference_construct",
        REFERENCE_CONSTRUCT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load reference EVA ASIC construct: {REFERENCE_CONSTRUCT}"
        )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.construct


def construct():
    """Return the blocking EVA graph with Catapult selected for Allo HLS."""
    graph = _load_reference_construct()()
    compilation = graph.get_node("allo-asic-compilation")
    compilation.update_params(
        {
            "backend": "catapult",
            "build_mode": "csyn",
            "backend_options": "device=nangate-45nm_beh",
        }
    )
    graph.update_params(
        {
            "construct_path": __file__,
            "allo_design_file": str(EVA_SOURCE),
            "allo_entrypoint": "build",
            "backend": "catapult",
            "build_mode": "csyn",
            "clock_period": 10.0,
            "macro_clock_period": 8.0,
            "backend_options": "device=nangate-45nm_beh",
            "report_design_name": "allo-eva-blocking_2x2_fp16_FIFO8_catapult",
        }
    )
    return graph


if __name__ == "__main__":
    construct()
