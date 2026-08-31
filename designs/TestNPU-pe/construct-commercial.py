#=========================================================================
# construct.py
#=========================================================================
# Commercial ASIC flow for the PE block using single-node Innovus PNR
#
# Author : Julian Bushlow
# Date   : June 15, 2026
#

import os

from mflowgen.components import Graph, Node

def construct():

  g = Graph()

  #-----------------------------------------------------------------------
  # Parameters
  #-----------------------------------------------------------------------

  adk_name = 'freepdk-45nm'
  adk_view = 'view-standard'

  parameters = {
    'construct_path'      : __file__,
    'design_name'         : 'pe',
    'clock_period'        : 5.0,
    'adk'                 : adk_name,
    'adk_view'            : adk_view,
    # Enable GUIs
    'enable_gui'          : True,
    # GLS Testbench
    'saif_instance'       : 'PETb/pe_inst',
    # Synthesis
    # Flatten effort 0 is strict hierarchy, 3 is full flattening
    'flatten_effort'      : 3,
    'topographical'       : True,
    # Postroute timing target slack
    'setup_target_slack'  : 0.000,
    'hold_target_slack'   : 0.050,
    # Utilization target
    'core_density_target' : 0.70,
    # Floorplan controls
    'floorplan_mode'      : 'auto',
    'floorplan_aspect_ratio' : 1.0,
    'floorplan_width'     : '',
    'floorplan_height'    : '',
    # Stop after floorplan, power, place, cts, route, or none
    'stop_after_step'     : 'none',
    # Innovus/ADK controls
    'process_node'        : 45,
    'max_route_layer'     : 7,
    'base_layer_idx'      : 0,
    'pin_layer_offset'    : 3,
    'power_mesh_bot_layer' : 8,
    'power_mesh_top_layer' : 9,
    'local_cpus'          : 16,
    'postroute_max_local_cpus' : 8,
    'macro_halo'          : 2.0,
    'macro_pg_resource_util' : 0.2,
    'macro_forbidden_space_to_macro' : '20,20',
    'macro_min_space_to_core' : '30,30',
    'macro_corner_keepout' : '5,5',
    'well_tap_cell'       : 'WELLTAP_X1',
    'well_tap_interval'   : 120,
    'primary_power_net'   : 'VDD',
    'primary_ground_net'  : 'VSS',
    'power_pin_names'     : 'VDD',
    'ground_pin_names'    : 'VSS',
    'top_module'          : 'pe',
    'design_path'         : '../../../mininpu/hardware/rtl',
    'sv2v_include_dirs'   : '.:pkg:core/mxu:core/vpu/fpu',
    'normalize_rtl'       : False,
    'consume_upstream_testbench' : True,
    'testbench_top'       : 'PETb',
    'dut_instance'        : 'pe_inst',
    'activity_source'     : 'bagl_vcd',
    'pass_marker'         : 'PASS',
  }

  #-----------------------------------------------------------------------
  # Create nodes
  #-----------------------------------------------------------------------

  this_dir  = os.path.dirname( os.path.abspath( __file__ ) )
  asic_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  nodes_dir = os.path.join(asic_dir, "nodes")

  # ADK node

  g.sys_path.append(os.path.join(asic_dir, "adks"))
  g.set_adk( adk_name )
  adk = g.get_adk_node()

  testbench      = Node( this_dir + '/testbench'   )
  testbench_collect = Node( os.path.join(nodes_dir, 'testbench-collector')        )
  constraints    = Node( this_dir + '/constraints' )

  sv2v           = Node( os.path.join(nodes_dir, 'sv2v-design-collector')         )
  synth          = Node( os.path.join(nodes_dir, 'synopsys-dc-synthesis')         )
  pnr            = Node( os.path.join(nodes_dir, 'cadence-innovus-pnr')           )
  pt_signoff     = Node( os.path.join(nodes_dir, 'synopsys-pt-timing-signoff')    )
  genlibdb       = Node( os.path.join(nodes_dir, 'synopsys-ptpx-genlibdb')        )
  gdsmerge       = Node( os.path.join(nodes_dir, 'mentor-calibre-gdsmerge')       )
  drc            = Node( os.path.join(nodes_dir, 'mentor-calibre-drc')            )
  lvs            = Node( os.path.join(nodes_dir, 'mentor-calibre-lvs')            )
  utilities      = Node( os.path.join(nodes_dir, 'asic-flow-utilities')            )
  rtl_sim        = Node( os.path.join(nodes_dir, 'commercial-rtl-sim')             )
  ffgl_sim       = Node( os.path.join(nodes_dir, 'commercial-ffgl-sim')            )
  bagl_sim       = Node( os.path.join(nodes_dir, 'commercial-bagl-sim')            )
  power_est      = Node( os.path.join(nodes_dir, 'synopsys-pt-power')             )
  summary        = Node( os.path.join(nodes_dir, 'asic-flow-summary')              )
  finalize       = Node( os.path.join(nodes_dir, 'asic-flow-finalize')             )

  #-----------------------------------------------------------------------
  # Modify Nodes
  #-----------------------------------------------------------------------

  #-----------------------------------------------------------------------
  # Graph -- Add nodes
  #-----------------------------------------------------------------------

  g.add_node( sv2v              )
  g.add_node( testbench         )
  g.add_node( testbench_collect )
  g.add_node( utilities         )
  g.add_node( rtl_sim           )
  g.add_node( constraints       )
  g.add_node( synth             )
  g.add_node( pnr               )
  g.add_node( gdsmerge          )
  g.add_node( drc               )
  g.add_node( lvs               )
  g.add_node( ffgl_sim          )
  g.add_node( bagl_sim          )
  g.add_node( pt_signoff        )
  g.add_node( power_est         )
  g.add_node( genlibdb          )
  g.add_node( summary           )
  g.add_node( finalize          )

  #-----------------------------------------------------------------------
  # Graph -- Add edges
  #-----------------------------------------------------------------------

  # Connect by name

  g.connect_by_name( adk,            synth          )
  g.connect_by_name( adk,            pnr            )
  g.connect_by_name( adk,            pt_signoff     )
  g.connect_by_name( adk,            gdsmerge       )
  g.connect_by_name( adk,            drc            )
  g.connect_by_name( adk,            lvs            )
  g.connect_by_name( adk,            genlibdb       )
  g.connect_by_name( adk,            ffgl_sim       )
  g.connect_by_name( adk,            bagl_sim       )

  g.connect_by_name( sv2v,           synth          )
  g.connect_by_name( sv2v,           rtl_sim        )
  g.connect_by_name( testbench,      testbench_collect )
  for node in [rtl_sim, ffgl_sim, bagl_sim, power_est]:
    g.connect_by_name( testbench_collect, node )
  for node in [rtl_sim, ffgl_sim, bagl_sim, lvs]:
    g.connect_by_name( utilities, node )
  g.connect_by_name( constraints,    synth          )

  g.connect_by_name( synth,          pnr            )
  g.connect_by_name( synth,          ffgl_sim       )
  g.connect_by_name( constraints,    pnr            )

  g.connect_by_name( pnr,            pt_signoff     )
  g.connect_by_name( pnr,            genlibdb       )
  g.connect_by_name( pnr,            gdsmerge       )
  g.connect_by_name( pnr,            drc            )
  g.connect_by_name( pnr,            lvs            )

  g.connect_by_name( gdsmerge,       drc            )
  g.connect_by_name( gdsmerge,       lvs            )

  g.connect_by_name( pnr,            bagl_sim       )

  g.connect_by_name( adk,            power_est      )
  g.connect_by_name( pnr,            power_est      )
  g.connect_by_name( bagl_sim,       power_est      )
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
  g.connect_by_name( summary,         finalize       )

  #-----------------------------------------------------------------------
  # Parameterize
  #-----------------------------------------------------------------------

  g.update_params( parameters )

  return g


if __name__ == '__main__':
  g = construct()
#  g.plot()
