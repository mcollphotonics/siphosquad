# 1. import the ring resonator
from technologies import silicon_photonics
from ipkiss3 import all as i3
from bm_topological_test import SingleResonator
from bm_topological_test import CROW_1D
from picazzo3.traces.wire_wg import WireWaveguideTemplate
from euler90_rounding_algorithm import EulerArbAlgorithm

# First the default of the class
default_ring_resonator = SingleResonator(name="default")
default_ring_resonator_layout = default_ring_resonator.Layout()
default_ring_resonator_layout.write_gdsii("default_testing.gds")

# now putting in a specific example, same as default but with Euler bends
euler_ring_resonator = SingleResonator(name="euler_default")
euler_ring_resonator_layout = euler_ring_resonator.Layout(ring_x_straight=5., rounding_algorithm=EulerArbAlgorithm)
euler_ring_resonator_layout.write_gdsii("euler_test.gds")

# putting in a detailed example for clarity
waveguide_template = WireWaveguideTemplate(name="the_waveguide")
waveguide_template_layout = waveguide_template.Layout(core_width=1.5, cladding_width=2.0)

full_ring_resonator = SingleResonator(name="all_vars", wg_coupler_template=waveguide_template,
                                      wg_ring_template=waveguide_template)

full_ring_layout = full_ring_resonator.Layout(bend_radius_ring=10., ring_x_straight=5., ring_y_straight=20.,
                                              external_straights=15., external_gap=1.,
                                              rounding_algorithm=EulerArbAlgorithm)

full_ring_layout.write_gdsii("full_euler_layout.gds")

# And try the Crow 1D
default_crow_ring_resonator = CROW_1D(name="crow_def")
default_crow_resonator_layout = default_crow_ring_resonator.Layout(num_rings=5)

print default_crow_resonator_layout.ports
print "Ports look good"

default_crow_resonator_layout.write_gdsii("crow_default_testing.gds")

# and do a 1D Crow with a custom gap list, need to add exception but seems to work fine for now
the_list = [1.5, 2.5, 1.0, 1.5]
gapped_crow_ring_resonator = CROW_1D(name="gapped")
gapped_crow_resonator_layout = gapped_crow_ring_resonator.Layout(num_rings=5, use_gap_list=True, ring_gap_list=the_list)

gapped_crow_resonator_layout.write_gdsii("crow_gapped_testing.gds")

print "fin"
