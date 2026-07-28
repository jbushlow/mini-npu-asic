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
    'flatten_effort'      : 0,
    'topographical'       : True,
    # Postroute timing target slack
    'setup_target_slack'  : 0.000,
    'hold_target_slack'   : 0.050,
    # Utilization target
    'core_density_target' : 0.70,
    'top_module'          : 'pe',
    'design_path'         : '../../../mininpu/npu/src/compute_tile',
    'sv2v_include_dirs'   : ".:.."
  }

  #-----------------------------------------------------------------------
  # Create nodes
  #-----------------------------------------------------------------------

  this_dir  = os.path.dirname( os.path.abspath( __file__ ) )
  asic_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  nodes_dir = os.path.join(asic_dir, "nodes")

  # ADK node

  g.set_adk( adk_name )
  adk = g.get_adk_node()

  testbench      = Node( this_dir + '/testbench'   )
  constraints    = Node( this_dir + '/constraints' )

  info           = Node( 'info',                            default=True          )
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
