"""MiniNPU v2 through the complete two-stage commercial ASIC flow."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REFERENCE_CONSTRUCT = (
    THIS_DIR.parent / "allo-test" / "construct-commercial.py"
)


def _load_reference_construct():
    spec = spec_from_file_location(
        "allo_mininpu_v2_reference_construct",
        REFERENCE_CONSTRUCT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load reference ASIC construct: {REFERENCE_CONSTRUCT}"
        )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.construct


def construct():
    """Return the standard graph specialized for the 5x5 MiniNPU v2."""
    graph = _load_reference_construct()()
    graph.get_node("allo-asic-compilation").update_params(
        {
            "allo_array_size": 5,
            "allo_reduction_size": 5,
            "allo_dtype_bits": 16,
            "allo_fifo_depth": 4,
        },
        allow_new=True,
    )
    graph.update_params(
        {
            "construct_path": __file__,
            "allo_design_file": str(THIS_DIR / "allo_design.py"),
            "allo_entrypoint": "build",
            "allo_array_size": 5,
            "allo_reduction_size": 5,
            "allo_dtype_bits": 16,
            "allo_fifo_depth": 4,
            "backend": "vitis",
            "build_mode": "csyn",
            "clock_period": 10.0,
            "macro_clock_period": 8.0,
            "backend_options": "device=u280",
            "python_bin": "/home/jb2698/.conda/envs/allo/bin/python",
            "allo_setup_script": "/work/shared/common/allo/setup-llvm-main.sh",
            "top_module": "mininpu_v2",
            "design_name": "mininpu_v2",
            "report_design_name": "allo-mininpu-v2_5_lane",
            "adk": "freepdk-45nm",
            "adk_view": "view-standard",
            "min_macro_reuse": 2,
            "harden_repeated_hls_submodules": False,
            "bypass_macro_generation": False,
            "enable_kernel_rotation": True,
            "enable_gui": False,
            "nthreads": 4,
            "local_cpus": 4,
            "postroute_max_local_cpus": 4,
            "antenna_check_policy": "report",
            "drc_check_policy": "report",
            "stop_after_step": "place",
            "core_density_target": 0.70,
            "floorplan_aspect_ratio": 2.0,
            "pin_primary_fraction": 0.70,
            "pin_corner_keepout": 2.0,
            "highest_macro_routing_layer": 5,
            "drc_nthreads": 4,
            "lvs_nthreads": 4,
            "flatten_effort": 1,
            "macro_separation_x": 30.0,
            "macro_separation_y": 30.0,
            "kernel_separation_x": 30.0,
            "kernel_separation_y": 30.0,
            "sram_separation": 20.0,
            "google_sheets_enabled": True,
            "google_sheets_required": True,
            "google_sheets_credentials":
                "/home/jb2698/.config/gspread/service_account.json",
            "google_spreadsheet_id":
                "1F5ck4kwl_YjXzAMYE-LmGeHsYYSfzZKWXKVH9LVJ9zU",
            "google_worksheet_name": "Results_Landing",
        }
    )
    return graph


if __name__ == "__main__":
    construct()
