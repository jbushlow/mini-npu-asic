"""Front-end portion of the commercial hierarchical ASIC research flow."""

import os

from mflowgen.components import Graph, Node


def construct():
    graph = Graph()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_dir = os.path.join(os.path.dirname(os.path.dirname(this_dir)), "nodes")

    allo_build = Node(os.path.join(nodes_dir, "allo-asic-compilation"))
    rtl_normalize = Node(os.path.join(nodes_dir, "sv2v-rtl-allo"))
    graph.add_node(allo_build)
    graph.add_node(rtl_normalize)
    graph.connect_by_name(allo_build, rtl_normalize)

    graph.update_params(
        {
            "construct_path": __file__,
            "allo_design_file": os.path.join(this_dir, "allo_design.py"),
            "allo_entrypoint": "build",
            "backend": "vitis",
            "build_mode": "csyn",
            "clock_period": 10.0,
            "backend_options": "device=u280",
            "python_bin": "/home/jb2698/.conda/envs/allo/bin/python",
            "allo_setup_script": "/work/shared/common/allo/setup-llvm-main.sh",
            "top_module": "top",
        }
    )
    return graph


if __name__ == "__main__":
    construct()
