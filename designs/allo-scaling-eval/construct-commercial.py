"""Allo scaling evaluation through Stage 1 and Stage 2 commercial flows."""

import os

from mflowgen.components import Graph, Node


def construct():
    graph = Graph()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_dir = os.path.join(os.path.dirname(os.path.dirname(this_dir)), "nodes")

    graph.set_adk("freepdk-45nm")
    adk = graph.get_adk_node()

    allo_build = Node(os.path.join(nodes_dir, "allo-asic-compilation"))
    # These design-specific parameters are intentionally local to this graph.
    # Register them on the compilation node so mflowgen exports them while it
    # imports allo_design.py; Graph.update_params only updates declared keys.
    allo_build.update_params(
        {
            "allo_array_size": 4,
            "allo_reduction_size": 4,
            "allo_dtype_bits": 32,
            "allo_fifo_depth": 4,
        },
        allow_new=True,
    )
    rtl_normalize = Node(os.path.join(nodes_dir, "sv2v-rtl-allo"))

    stage1_dir = os.path.join(nodes_dir, "allo-macro-generation")
    macro_plan = Node(os.path.join(stage1_dir, "allo-asic-macro-plan"))
    macro_synthesis = Node(
        os.path.join(stage1_dir, "commercial-batch-synthesis")
    )
    macro_pnr = Node(os.path.join(stage1_dir, "commercial-macro-pnr"))
    macro_physical_verify = Node(
        os.path.join(stage1_dir, "commercial-batch-physical-verify")
    )
    macro_signoff = Node(
        os.path.join(stage1_dir, "commercial-batch-signoff")
    )
    macro_publish = Node(
        os.path.join(stage1_dir, "commercial-macro-publish")
    )

    stage2_dir = os.path.join(nodes_dir, "allo-full-chip")
    assembly_plan = Node(os.path.join(stage2_dir, "allo-asic-assembly-plan"))
    rtl_assembly = Node(os.path.join(stage2_dir, "allo-asic-rtl-assembly"))
    full_chip_synthesis = Node(
        os.path.join(stage2_dir, "commercial-full-chip-synthesis")
    )
    physical_intent = Node(os.path.join(stage2_dir, "allo-asic-physical-intent"))
    full_chip_pnr = Node(os.path.join(stage2_dir, "commercial-full-chip-pnr"))
    full_chip_gdsmerge = Node(
        os.path.join(stage2_dir, "commercial-full-chip-gdsmerge")
    )
    full_chip_drc = Node(os.path.join(stage2_dir, "commercial-full-chip-drc"))
    full_chip_lvs = Node(os.path.join(stage2_dir, "commercial-full-chip-lvs"))
    flow_summary = Node(os.path.join(nodes_dir, "allo-asic-flow-summary"))

    for node in [
        allo_build,
        rtl_normalize,
        macro_plan,
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
        macro_publish,
        assembly_plan,
        rtl_assembly,
        full_chip_synthesis,
        physical_intent,
        full_chip_pnr,
        full_chip_gdsmerge,
        full_chip_drc,
        full_chip_lvs,
        flow_summary,
    ]:
        graph.add_node(node)

    # Allo/Vitis compilation and normalized RTL production.
    graph.connect_by_name(allo_build, rtl_normalize)

    # Stage 1: select, harden, verify, characterize, and publish reusable PEs.
    graph.connect_by_name(rtl_normalize, macro_plan)
    graph.connect_by_name(macro_plan, macro_synthesis)
    graph.connect_by_name(macro_synthesis, macro_pnr)
    graph.connect_by_name(macro_pnr, macro_physical_verify)
    graph.connect_by_name(macro_physical_verify, macro_signoff)
    graph.connect_by_name(macro_signoff, macro_publish)
    graph.connect_by_name(rtl_normalize, macro_publish)
    for node in [
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
    ]:
        graph.connect_by_name(adk, node)

    # Stage 2: substitute the published macro views and implement the chip.
    graph.connect_by_name(rtl_normalize, assembly_plan)
    graph.connect_by_name(macro_publish, assembly_plan)
    graph.connect_by_name(rtl_normalize, rtl_assembly)
    graph.connect_by_name(assembly_plan, rtl_assembly)
    graph.connect_by_name(macro_publish, rtl_assembly)
    graph.connect_by_name(rtl_assembly, full_chip_synthesis)
    graph.connect_by_name(macro_publish, full_chip_synthesis)
    graph.connect_by_name(adk, full_chip_synthesis)
    graph.connect_by_name(assembly_plan, physical_intent)
    graph.connect_by_name(rtl_assembly, physical_intent)
    graph.connect_by_name(macro_publish, physical_intent)
    graph.connect_by_name(full_chip_synthesis, physical_intent)
    graph.connect_by_name(rtl_assembly, full_chip_pnr)
    graph.connect_by_name(macro_publish, full_chip_pnr)
    graph.connect_by_name(full_chip_synthesis, full_chip_pnr)
    graph.connect_by_name(physical_intent, full_chip_pnr)
    graph.connect_by_name(adk, full_chip_pnr)

    # Full-chip physical verification.
    graph.connect_by_name(full_chip_pnr, full_chip_gdsmerge)
    graph.connect_by_name(macro_publish, full_chip_gdsmerge)
    graph.connect_by_name(adk, full_chip_gdsmerge)
    graph.connect_by_name(full_chip_gdsmerge, full_chip_drc)
    graph.connect_by_name(adk, full_chip_drc)
    graph.connect_by_name(full_chip_gdsmerge, full_chip_lvs)
    graph.connect_by_name(full_chip_pnr, full_chip_lvs)
    graph.connect_by_name(macro_publish, full_chip_lvs)
    graph.connect_by_name(adk, full_chip_lvs)

    # Terminal summary waits for both macro and full-chip results.
    graph.connect_by_name(macro_publish, flow_summary)
    for node in [
        allo_build,
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
        full_chip_synthesis,
        full_chip_pnr,
        full_chip_gdsmerge,
        full_chip_drc,
        full_chip_lvs,
    ]:
        graph.connect_by_name(node, flow_summary)
    graph.connect_by_name(physical_intent, flow_summary)

    graph.update_params(
        {
            "construct_path": __file__,
            "allo_design_file": os.path.join(this_dir, "allo_design.py"),
            "allo_entrypoint": "build",
            "allo_array_size": 6,
            "allo_reduction_size": 4,
            "allo_dtype_bits": 32,
            "allo_fifo_depth": 4,
            "backend": "vitis",
            "build_mode": "csyn",
            "clock_period": 10.0,
            "macro_clock_period": 8.0,
            "backend_options": "device=u280",
            "python_bin": "/home/jb2698/.conda/envs/allo/bin/python",
            "allo_setup_script": "/work/shared/common/allo/setup-llvm-main.sh",
            "top_module": "top",
            "design_name": "top",
            "report_design_name": "allo-scaling-eval, N=6, int32",
            "adk": "freepdk-45nm",
            "adk_view": "view-standard",
            "min_macro_reuse": 2,
            "bypass_macro_generation": False,
            "enable_gui": False,
            "nthreads": 4,
            "local_cpus": 4,
            "postroute_max_local_cpus": 4,
            "antenna_check_policy": "report",
            "kernel_cluster_max_density_percent": 55,
            "drc_nthreads": 4,
            "lvs_nthreads": 4,
            "flatten_effort": 1,
            "macro_separation_x": 25.0,
            "macro_separation_y": 25.0,
            "kernel_separation_x": 30.0,
            "kernel_separation_y": 30.0,
            "sram_separation": 20.0,
            "google_sheets_enabled": True,
            "google_sheets_required": True,
            "google_sheets_credentials":
                "/home/jb2698/.config/gspread/service_account.json",
            "google_spreadsheet_id": "1F5ck4kwl_YjXzAMYE-LmGeHsYYSfzZKWXKVH9LVJ9zU",
            "google_worksheet_name": "Results_Landing",
        }
    )
    return graph


if __name__ == "__main__":
    construct()
