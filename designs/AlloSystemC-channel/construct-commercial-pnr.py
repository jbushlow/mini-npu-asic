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
    'flatten_effort': 3,
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
    'power_pin_names': 'VDD',
    'ground_pin_names': 'VSS',
    'consume_upstream_testbench': True,
    'testbench_top': 'HelloChannelTb',
    'dut_instance': 'dut',
    'activity_source': 'bagl_vcd',
    'pass_marker': 'PASS',
  }

  this_dir = os.path.dirname(os.path.abspath(__file__))
  asic_dir = os.path.dirname(os.path.dirname(this_dir))
  nodes_dir = os.path.join(asic_dir, 'nodes')

  g.sys_path.append(os.path.join(asic_dir, 'adks'))
  g.set_adk(adk_name)
  adk = g.get_adk_node()

  rtl = Node(os.path.join(this_dir, 'rtl'))
  testbench = Node(os.path.join(this_dir, 'testbench'))
  testbench_collect = Node(os.path.join(nodes_dir, 'testbench-collector'))
  constraints = Node(os.path.join(this_dir, 'constraints'))
  synth = Node(os.path.join(nodes_dir, 'synopsys-dc-synthesis'))
  pnr = Node(os.path.join(nodes_dir, 'cadence-innovus-pnr'))
  pt_signoff = Node(os.path.join(nodes_dir, 'synopsys-pt-timing-signoff'))
  genlibdb = Node(os.path.join(nodes_dir, 'synopsys-ptpx-genlibdb'))
  gdsmerge = Node(os.path.join(nodes_dir, 'mentor-calibre-gdsmerge'))
  drc = Node(os.path.join(nodes_dir, 'mentor-calibre-drc'))
  lvs = Node(os.path.join(nodes_dir, 'mentor-calibre-lvs'))
  utilities = Node(os.path.join(nodes_dir, 'asic-flow-utilities'))
  rtl_sim = Node(os.path.join(nodes_dir, 'commercial-rtl-sim'))
  ffgl_sim = Node(os.path.join(nodes_dir, 'commercial-ffgl-sim'))
  bagl_sim = Node(os.path.join(nodes_dir, 'commercial-bagl-sim'))
  power_est = Node(os.path.join(nodes_dir, 'synopsys-pt-power'))
  summary = Node(os.path.join(nodes_dir, 'asic-flow-summary'))
  finalize = Node(os.path.join(nodes_dir, 'asic-flow-finalize'))

  for node in [rtl, testbench, testbench_collect, constraints, utilities,
               rtl_sim, synth, ffgl_sim, pnr, pt_signoff, genlibdb, gdsmerge,
               drc, lvs, bagl_sim, power_est, summary, finalize]:
    g.add_node(node)

  for node in [synth, pnr, pt_signoff, genlibdb, gdsmerge, drc, lvs, ffgl_sim,
               bagl_sim, power_est]:
    g.connect_by_name(adk, node)

  g.connect_by_name(rtl, synth)
  g.connect_by_name(rtl, rtl_sim)
  g.connect_by_name(testbench, testbench_collect)
  for node in [rtl_sim, ffgl_sim, bagl_sim, power_est]:
    g.connect_by_name(testbench_collect, node)
  for node in [rtl_sim, ffgl_sim, bagl_sim, lvs]:
    g.connect_by_name(utilities, node)
  g.connect_by_name(constraints, synth)
  g.connect_by_name(synth, pnr)
  g.connect_by_name(synth, power_est)

  for node in [pt_signoff, genlibdb, gdsmerge, drc, lvs, bagl_sim, power_est]:
    g.connect_by_name(pnr, node)
  g.connect_by_name(synth, ffgl_sim)

  g.connect_by_name(gdsmerge, drc)
  g.connect_by_name(gdsmerge, lvs)
  g.connect_by_name(bagl_sim, power_est)
  g.connect(synth.o('synthesis-metrics.json'), summary.i('synthesis-metrics.json'))
  g.connect(pnr.o('pnr-metrics.json'), summary.i('pnr-metrics.json'))
  g.connect(pt_signoff.o('timing-metrics.json'), summary.i('timing-metrics.json'))
  g.connect(gdsmerge.o('gdsmerge-metrics.json'), summary.i('gdsmerge-metrics.json'))
  g.connect(drc.o('drc-metrics.json'), summary.i('drc-metrics.json'))
  g.connect(drc.o('drc-policy.json'), summary.i('drc-policy.json'))
  g.connect(lvs.o('lvs-metrics.json'), summary.i('lvs-metrics.json'))
  g.connect(power_est.o('power.rpt'), summary.i('power.rpt'))
  g.connect(power_est.o('activity-source.json'), summary.i('activity-source.json'))
  g.connect(rtl_sim.o('simulation-report.json'), summary.i('rtl-simulation-report.json'))
  g.connect(ffgl_sim.o('simulation-report.json'), summary.i('ffgl-simulation-report.json'))
  g.connect(bagl_sim.o('simulation-report.json'), summary.i('bagl-simulation-report.json'))
  g.connect_by_name(summary, finalize)

  g.update_params(parameters)
  return g


if __name__ == '__main__':
  construct()
