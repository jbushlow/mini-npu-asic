"""Commercial ASIC flow for the Allo SystemC hello_channel example."""

import os

from mflowgen.components import Graph, Node


def construct():
  g = Graph()

  adk_name = 'freepdk-45nm'
  parameters = {
    'construct_path': __file__,
    'design_name': 'hello_channel',
    'clock_period': 10.0,
    'adk': adk_name,
    'adk_view': 'view-standard',
    'enable_gui': True,
    'saif_instance': 'HelloChannelTb/dut',
    'flatten_effort': 0,
    'topographical': False,
    'setup_target_slack': 0.050,
    'hold_target_slack': 0.050,
    'core_density_target': 0.70,
    'floorplan_mode': 'auto',
    'floorplan_aspect_ratio': 1.0,
    'floorplan_width': '',
    'floorplan_height': '',
    'stop_after_step': 'none',
    'process_node': 45,
    'max_route_layer': 7,
    'base_layer_idx': 0,
    'pin_layer_offset': 3,
    'power_mesh_bot_layer': 8,
    'power_mesh_top_layer': 9,
    'local_cpus': 16,
    'postroute_max_local_cpus': 8,
    'well_tap_cell': 'WELLTAP_X1',
    'well_tap_interval': 120,
    'primary_power_net': 'VDD',
    'primary_ground_net': 'VSS',
    'power_pin_names': 'VDD,vdd',
    'ground_pin_names': 'VSS,gnd',
  }

  this_dir = os.path.dirname(os.path.abspath(__file__))
  asic_dir = os.path.dirname(os.path.dirname(this_dir))
  nodes_dir = os.path.join(asic_dir, 'nodes')

  g.set_adk(adk_name)
  adk = g.get_adk_node()

  rtl = Node(os.path.join(this_dir, 'rtl'))
  testbench = Node(os.path.join(this_dir, 'testbench'))
  constraints = Node(os.path.join(this_dir, 'constraints'))
  info = Node('info', default=True)
  synth = Node(os.path.join(nodes_dir, 'synopsys-dc-synthesis'))
  pnr = Node(os.path.join(nodes_dir, 'cadence-innovus-pnr'))
  pt_signoff = Node(os.path.join(nodes_dir, 'synopsys-pt-timing-signoff'))
  genlibdb = Node(os.path.join(nodes_dir, 'synopsys-ptpx-genlibdb'))
  gdsmerge = Node(os.path.join(nodes_dir, 'mentor-calibre-gdsmerge'))
  drc = Node(os.path.join(nodes_dir, 'mentor-calibre-drc'))
  lvs = Node(os.path.join(nodes_dir, 'mentor-calibre-lvs'))
  vcs_sim = Node(os.path.join(nodes_dir, 'synopsys-vcs-sim-old'))
  power_est = Node(os.path.join(nodes_dir, 'synopsys-pt-power'))

  vcs_sim.extend_inputs(['test_vectors.txt'])
  vcs_sim.update_params(testbench.params())

  for node in [info, rtl, testbench, constraints, synth, pnr, pt_signoff,
               genlibdb, gdsmerge, drc, lvs, vcs_sim, power_est]:
    g.add_node(node)

  for node in [synth, pnr, pt_signoff, genlibdb, gdsmerge, drc, lvs, vcs_sim,
               power_est]:
    g.connect_by_name(adk, node)

  g.connect_by_name(rtl, synth)
  g.connect_by_name(constraints, synth)
  g.connect_by_name(synth, pnr)
  g.connect_by_name(constraints, pnr)

  for node in [pt_signoff, genlibdb, gdsmerge, drc, lvs, vcs_sim, power_est]:
    g.connect_by_name(pnr, node)

  g.connect_by_name(gdsmerge, drc)
  g.connect_by_name(gdsmerge, lvs)
  g.connect_by_name(testbench, vcs_sim)
  g.connect_by_name(vcs_sim, power_est)

  g.update_params(parameters)
  return g


if __name__ == '__main__':
  construct()
