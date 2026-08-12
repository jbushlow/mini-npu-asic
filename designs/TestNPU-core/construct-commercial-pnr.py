#=========================================================================
# construct.py
#=========================================================================
# Commercial ASIC flow for the MiniNPU core block using single-node Innovus PNR
#
# Author : Julian Bushlow
# Date   : July 29, 2026
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
    'design_name'         : 'core',
    'clock_period'        : 15.0,
    'adk'                 : adk_name,
    'adk_view'            : adk_view,
    # Enable GUIs
    'enable_gui'          : True,
    # GLS Testbench
    'saif_instance'       : 'CoreTb/core_inst',
    # Synthesis
    # Flatten effort 0 is strict hierarchy, 3 is full flattening
    'flatten_effort'      : 0,
    'topographical'       : False,
    # Postroute timing target slack
    'setup_target_slack'  : 0.050,
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
    'power_pin_names'     : 'VDD,vdd',
    'ground_pin_names'    : 'VSS,gnd',
    # SV2V params
    'top_module'          : 'core',
    'design_path'         : '~/mininpu/hardware/rtl',
    'sv2v_include_dirs'   : '.:pkg:common:core/top:core/sequencer:core/spad:core/mxu:core/vpu:core/vpu/fpu:core/dmu:uncore/dma',
    # Reuse pre-generated SRAM views from mini-npu-asic/srams.
    'sram_manifest'       : 'rtl/sram_manifest.yml',
    'use_sram_cache'      : True,
    'sram_cache_path'     : '../../srams',
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
  constraints    = Node( this_dir + '/constraints' )

  info           = Node( 'info',                            default=True          )
  openram        = Node( os.path.join(nodes_dir, 'openram-sram-generation')       )
  sv2v           = Node( os.path.join(nodes_dir, 'sv2v-design-collector')         )
  synth          = Node( os.path.join(nodes_dir, 'synopsys-dc-synthesis')         )
  pnr            = Node( os.path.join(nodes_dir, 'cadence-innovus-pnr')           )
  pt_signoff     = Node( os.path.join(nodes_dir, 'synopsys-pt-timing-signoff')    )
  genlibdb       = Node( os.path.join(nodes_dir, 'synopsys-ptpx-genlibdb')        )
  gdsmerge       = Node( os.path.join(nodes_dir, 'mentor-calibre-gdsmerge')       )
  drc            = Node( os.path.join(nodes_dir, 'mentor-calibre-drc')            )
  lvs            = Node( os.path.join(nodes_dir, 'mentor-calibre-lvs')            )
  vcs_sim        = Node( os.path.join(nodes_dir, 'synopsys-vcs-sim-old')          )
  power_est      = Node( os.path.join(nodes_dir, 'synopsys-pt-power')             )

  #-----------------------------------------------------------------------
  # Modify Nodes
  #-----------------------------------------------------------------------

  vcs_sim.extend_inputs( ['test_vectors.txt'] )
  vcs_sim.update_params( testbench.params() )

  #-----------------------------------------------------------------------
  # Graph -- Add nodes
  #-----------------------------------------------------------------------

  g.add_node( info              )
  g.add_node( sv2v              )
  g.add_node( testbench         )
  g.add_node( openram           )
  g.add_node( constraints       )
  g.add_node( synth             )
  g.add_node( pnr               )
  g.add_node( gdsmerge          )
  g.add_node( drc               )
  g.add_node( lvs               )
  g.add_node( vcs_sim           )
  g.add_node( pt_signoff        )
  g.add_node( power_est         )
  g.add_node( genlibdb          )

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

  g.connect_by_name( openram,        synth          )
  g.connect_by_name( openram,        pnr            )
  g.connect_by_name( openram,        pt_signoff     )
  g.connect_by_name( openram,        genlibdb       )
  g.connect_by_name( openram,        gdsmerge       )
  g.connect_by_name( openram,        lvs            )
  g.connect_by_name( openram,        vcs_sim        )
  g.connect_by_name( openram,        power_est      )

  g.connect_by_name( sv2v,           synth          )
  g.connect_by_name( constraints,    synth          )

  g.connect_by_name( synth,          pnr            )
  g.connect_by_name( constraints,    pnr            )

  g.connect_by_name( pnr,            pt_signoff     )
  g.connect_by_name( pnr,            genlibdb       )
  g.connect_by_name( pnr,            gdsmerge       )
  g.connect_by_name( pnr,            drc            )
  g.connect_by_name( pnr,            lvs            )

  g.connect_by_name( gdsmerge,       drc            )
  g.connect_by_name( gdsmerge,       lvs            )

  g.connect_by_name( adk,            vcs_sim        )
  g.connect_by_name( pnr,            vcs_sim        )
  g.connect_by_name( testbench,      vcs_sim        )

  g.connect_by_name( adk,            power_est      )
  g.connect_by_name( pnr,            power_est      )
  g.connect_by_name( vcs_sim,        power_est      )

  #-----------------------------------------------------------------------
  # Parameterize
  #-----------------------------------------------------------------------

  g.update_params( parameters )

  return g


if __name__ == '__main__':
  g = construct()
#  g.plot()
