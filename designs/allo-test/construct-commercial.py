"""Allo front end plus Stage-1 macro and Stage-2 full-chip research flow."""

import os

from mflowgen.components import Graph, Node


def construct():
    graph = Graph()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_dir = os.path.join(os.path.dirname(os.path.dirname(this_dir)), "nodes")

    graph.set_adk("freepdk-45nm")
    adk = graph.get_adk_node()

    allo_build = Node(os.path.join(nodes_dir, "allo-asic-compilation"))
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
    flow_summary = Node(os.path.join(nodes_dir, "allo-asic-flow-summary"))
    stage2_dir = os.path.join(nodes_dir, "allo-full-chip")
    assembly_plan = Node(os.path.join(stage2_dir, "allo-asic-assembly-plan"))
    rtl_assembly = Node(os.path.join(stage2_dir, "allo-asic-rtl-assembly"))
    full_chip_synthesis = Node(
        os.path.join(stage2_dir, "commercial-full-chip-synthesis")
    )
    for node in [
        allo_build,
        rtl_normalize,
        macro_plan,
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
        macro_publish,
        flow_summary,
        assembly_plan,
        rtl_assembly,
        full_chip_synthesis,
    ]:
        graph.add_node(node)

    graph.connect_by_name(allo_build, rtl_normalize)
    graph.connect_by_name(rtl_normalize, macro_plan)
    graph.connect_by_name(macro_plan, macro_synthesis)
    graph.connect_by_name(macro_synthesis, macro_pnr)
    graph.connect_by_name(macro_pnr, macro_physical_verify)
    graph.connect_by_name(macro_physical_verify, macro_signoff)
    graph.connect_by_name(macro_signoff, macro_publish)
    graph.connect_by_name(rtl_normalize, macro_publish)
    graph.connect_by_name(macro_publish, flow_summary)
    for node in [
        allo_build,
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
    ]:
        graph.connect_by_name(node, flow_summary)
    graph.connect_by_name(rtl_normalize, assembly_plan)
    graph.connect_by_name(macro_publish, assembly_plan)
    graph.connect_by_name(rtl_normalize, rtl_assembly)
    graph.connect_by_name(assembly_plan, rtl_assembly)
    graph.connect_by_name(macro_publish, rtl_assembly)
    graph.connect_by_name(rtl_assembly, full_chip_synthesis)
    graph.connect_by_name(macro_publish, full_chip_synthesis)
    graph.connect_by_name(adk, full_chip_synthesis)
    graph.connect_by_name(full_chip_synthesis, flow_summary)
    for node in [
        macro_synthesis,
        macro_pnr,
        macro_physical_verify,
        macro_signoff,
    ]:
        graph.connect_by_name(adk, node)

    graph.update_params(
        {
            "construct_path": __file__,
            "allo_design_file": os.path.join(this_dir, "allo_design.py"),
            "allo_entrypoint": "build",
            "backend": "vitis",
            "build_mode": "csyn",
            "clock_period": 10.0,
            "macro_clock_period": 8.0,
            "backend_options": "device=u280",
            "python_bin": "/home/jb2698/.conda/envs/allo/bin/python",
            "allo_setup_script": "/work/shared/common/allo/setup-llvm-main.sh",
            "top_module": "top",
            "adk": "freepdk-45nm",
            "adk_view": "view-standard",
            "min_macro_reuse": 2,
            "enable_gui": False,
            "nthreads": 4,
            "local_cpus": 4,
            "postroute_max_local_cpus": 4,
            "drc_nthreads": 4,
            "lvs_nthreads": 4,
        }
    )
    return graph


if __name__ == "__main__":
    construct()
