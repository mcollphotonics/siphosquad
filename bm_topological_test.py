"""
Playing around with classes which may be useful for topological and Crow structures
Blair Morrison USYD
26 March 2017
"""

from ipkiss3 import all as i3
from picazzo3.filters.ring import RingRect
from ipkiss3.all import SplineRoundingAlgorithm


class SingleResonator(i3.PCell):
    """ A single resonator test class
    """
    _name_prefix = "single_resonator"

    # defining the resonator as child cell with input and output waveguides
    resonator = i3.ChildCellProperty(restriction=i3.RestrictType(i3.PCell),
                                     doc="the input resonator")

    # define waveguide template and waveguide cells
    wg_coupler_template = i3.WaveguideTemplateProperty(default=i3.TECH.PCELLS.WG.DEFAULT)
    wg_ring_template = i3.WaveguideTemplateProperty(default=i3.TECH.PCELLS.WG.DEFAULT)
    wgs = i3.ChildCellListProperty(doc="list of waveguides")

    # define default MMI for class
    def _default_resonator(self):
        """
        Internal resonator which is repeated later
        """
        resonator = RingRect(name=self.name+"_resonator", ring_trace_template=self.wg_ring_template)
        return resonator

    # define rounded waveguides for the inputs and outputs of CROW
    def _default_wgs(self):
        wg_in = i3.RoundedWaveguide(name=self.name+"_wg_in", trace_template=self.wg_coupler_template)
        wg_pass = i3.RoundedWaveguide(name=self.name+"_wg_pass", trace_template=self.wg_coupler_template)
        # wg_ring = i3.RoundedWaveguide(name=self.name+"_wg_ring", trace_template=self.wg_coupler_template)
        return wg_in, wg_pass  # , wg_pass, wg_ring

    class Layout(i3.LayoutView):
        # specified parameters used for layout, lengths of various waveguides
        # using some default values if standard ring shape is used
        bend_radius_ring = i3.PositiveNumberProperty(default=10., doc="bend radius of ring")
        ring_x_straight = i3.PositiveNumberProperty(default=15., doc="straight between bends in x ring")
        ring_y_straight = i3.PositiveNumberProperty(default=25., doc="straight between bends in y ring")
        external_straights = i3.PositiveNumberProperty(default=10., doc="extra straight for outside structure")
        external_gap = i3.PositiveNumberProperty(default=0.5, doc="gap between outside waveguides and resonator")
        # external_radius = i3.PositiveNumberProperty(default=bend_radius_ring, doc="radius of outside coupler")
        use_rounding = i3.BoolProperty(default=False, doc="use non default bending algorithm")
        rounding_algorithm = i3.DefinitionProperty(default=SplineRoundingAlgorithm(), doc="secondary rounding algorithm")

        # define the layout of the internal coupler which we SRef below
        def _default_resonator(self):
            res_layout = self.cell.resonator.get_default_view(i3.LayoutView)  # Retrieve layout view following example

            # make the shape of the layout from the previous values. Assume (0, 0) is bottom middle!)
            # will do each corner for clarity
            # bottom_left = (-self.bend_radius_ring - self.ring_x_straight/2., 0.)
            # top_left = (-self.bend_radius_ring - self.ring_x_straight/2.,
            #             self.bend_radius_ring*2. + self.ring_y_straight)
            # top_right = (self.bend_radius_ring + self.ring_x_straight/2.,
            #              self.bend_radius_ring*2. + self.ring_y_straight)
            # bottom_right = (self.bend_radius_ring + self.ring_x_straight/2., 0.)
            # ring_shape = [bottom_left, top_left, top_right, bottom_right, bottom_left]
            # print ring_shape

            # tried to use generic round ring, but failed :P. Using ring rect instead
            # set the layout of the resonator. Stuck a bool for non default rounding algorithm
            if self.use_rounding is True:
                res_layout.set(bend_radius=self.bend_radius_ring, straights=(self.ring_x_straight, self.ring_y_straight)
                               , rounding_algorithm=self.rounding_algorithm)
            else:
                res_layout.set(bend_radius=self.bend_radius_ring,
                               straights=(self.ring_x_straight, self.ring_y_straight))  # , shape=ring_shape
            return res_layout

        # now we take the resonator which was just defined and stick it in the main *get components thing
        def _get_components(self):
            resonator = i3.SRef(name="another_res", reference=self.resonator)
            return resonator

        # setting the output shape of the access waveguides using a shape defined by ports from MMI (hopefully..)
        def _default_wgs(self):
            # bring in parts from rest of PCell Layout, used to grab positions
            resonator = self._get_components()
            wg_in_cell, wg_pass_cell = self.cell.wgs
            wg_template = self.wg_coupler_template
            wg_ring_template = self.wg_ring_template

            # using the ring radius for the external radius
            external_rad = self.bend_radius_ring
            external_str = self.external_straights

            # grabbing the position of the resonator to layout the rest of the coupler properly
            resonator_west_side = resonator.size_info().west
            resonator_south_side = resonator.size_info().south

            resonator_core_width = wg_ring_template.core_width
            resonator_clad_width = wg_ring_template.cladding_width
            coupler_core_width = wg_template.core_width

            # calculate the x position for center of input coupling waveguide when coupling, and make shape
            x_coup_spot = resonator_west_side + resonator_clad_width/2. - resonator_core_width/2. - self.external_gap \
                - coupler_core_width/2.

            # get bottom using the south and cladding information again
            bottom_left = (x_coup_spot - external_str - external_rad, resonator_south_side + resonator_clad_width/2.)
            bottom_right = (x_coup_spot, resonator_south_side + resonator_clad_width/2.)
            top_right = (x_coup_spot, bottom_right[1] + 2.*external_rad + self.ring_y_straight)
            top_left = (bottom_left[0], top_right[1])

            wg_shape = [bottom_left, bottom_right, top_right, top_left]

            # now make the instance using this shape info
            wg_in_layout = wg_in_cell.get_default_view(i3.LayoutView)
            if self.use_rounding is True:
                wg_in_layout.set(trace_template=wg_template, shape=wg_shape, bend_radius=external_rad,
                                 rounding_algorithm=self.rounding_algorithm)
            else:
                wg_in_layout.set(trace_template=wg_template, shape=wg_shape, bend_radius=external_rad)

            wg_pass_layout = wg_pass_cell.get_default_view(i3.LayoutView)
            # wg_in_layout.set()
            return wg_in_layout, wg_pass_layout  # wg_ring_layout

        # A few functions for grabbing waveguide parameters to determine lengths for FSR checking
        # def wg_lengths(self):
        #     # grab the lengths of internal waveguides to use for calculations later
        #     wg_in_layout, wg_pass_layout, wg_ring_layout = self.wgs
        #
        #     straights_and_bends = wg_ring_layout.trace_length()
        #     return straights_and_bends

        def _generate_instances(self, insts):
            # includes the get components and the new waveguides
            insts += self._get_components()
            wg_in_layout, wg_pass_layout = self.wgs  #  wg_pass_layout, wg_ring_layout

            insts += i3.SRef(reference=wg_in_layout, name="wg_in")
            # insts += i3.SRef(reference=wg_pass_layout, name="wg_pass")
            # insts += i3.SRef(reference=wg_ring_layout, name="wg_ring")
            return insts

        def _generate_ports(self, prts):
            # try to reuse the output waveguides following the example and change the names, looks good
            instances = self.instances
            prts += instances["wg_in"].ports["in"].modified_copy(name="in")
            prts += instances["wg_in"].ports["out"].modified_copy(name="pass")
            return prts

    class Netlist(i3.NetlistView):
        def _generate_terms(self, terms):
            terms += i3.OpticalTerm(name="in")
            # terms += i3.OpticalTerm(name="pass")
            return terms


class CROW_1D(i3.PCell):
    """ A single resonator test class
    """
    _name_prefix = "CROW_1D"

    # defining the resonator as child cell with input and output waveguides
    resonator = i3.ChildCellProperty(restriction=i3.RestrictType(i3.PCell),
                                     doc="the input resonator")

    # define waveguide template and waveguide cells
    wg_coupler_template = i3.WaveguideTemplateProperty(default=i3.TECH.PCELLS.WG.DEFAULT)
    wg_ring_template = i3.WaveguideTemplateProperty(default=i3.TECH.PCELLS.WG.DEFAULT)
    wgs = i3.ChildCellListProperty(doc="list of waveguides")

    # define default MMI for class
    def _default_resonator(self):
        """
        Internal resonator which is repeated later
        """
        resonator = RingRect(name=self.name+"_resonator", ring_trace_template=self.wg_ring_template)
        return resonator

    # define rounded waveguides for the inputs and outputs of CROW
    def _default_wgs(self):
        wg_in = i3.RoundedWaveguide(name=self.name+"_wg_in", trace_template=self.wg_coupler_template)
        wg_pass = i3.RoundedWaveguide(name=self.name+"_wg_pass", trace_template=self.wg_coupler_template)
        # wg_ring = i3.RoundedWaveguide(name=self.name+"_wg_ring", trace_template=self.wg_coupler_template)
        return wg_in, wg_pass  # , wg_pass, wg_ring

    class Layout(i3.LayoutView):
        # specified parameters used for layout, lengths of various waveguides
        # using some default values if standard ring shape is used
        bend_radius_ring = i3.PositiveNumberProperty(default=10., doc="bend radius of ring")
        ring_x_straight = i3.PositiveNumberProperty(default=15., doc="straight between bends in x ring")
        ring_y_straight = i3.PositiveNumberProperty(default=25., doc="straight between bends in y ring")
        external_straights = i3.PositiveNumberProperty(default=10., doc="extra straight for outside structure")
        external_gap = i3.PositiveNumberProperty(default=1., doc="gap between outside waveguides and resonator")
        # external_radius = i3.PositiveNumberProperty(default=bend_radius_ring, doc="radius of outside coupler")
        rounding_algorithm = i3.DefinitionProperty(default=SplineRoundingAlgorithm(), doc="secondary rounding algorithm")

        # extra layouting for the CROW
        num_rings = i3.IntProperty(default=3, doc="number of rings")
        ring_gap = i3.PositiveNumberProperty(default=0.5, doc="gap between internal rings")

        use_gap_list = i3.BoolProperty(default=False, doc="use non default bending algorithm")
        ring_gap_list = i3.ListProperty(default=[], doc="list of gaps for manual swapping, default empty!")


        # define the layout of the internal coupler which we SRef below
        def _default_resonator(self):
            res_layout = self.cell.resonator.get_default_view(i3.LayoutView)  # Retrieve layout view following example

            # make the shape of the layout from the previous values. Assume (0, 0) is bottom middle!)
            # will do each corner for clarity
            # bottom_left = (-self.bend_radius_ring - self.ring_x_straight/2., 0.)
            # top_left = (-self.bend_radius_ring - self.ring_x_straight/2.,
            #             self.bend_radius_ring*2. + self.ring_y_straight)
            # top_right = (self.bend_radius_ring + self.ring_x_straight/2.,
            #              self.bend_radius_ring*2. + self.ring_y_straight)
            # bottom_right = (self.bend_radius_ring + self.ring_x_straight/2., 0.)
            # ring_shape = [bottom_left, top_left, top_right, bottom_right, bottom_left]
            # print ring_shape

            # tried to use generic round ring, but failed :P. Using ring rect instead
            # set the layout of the resonator. Stuck a bool for non default rounding algorithm

            res_layout.set(bend_radius=self.bend_radius_ring, straights=(self.ring_x_straight, self.ring_y_straight)
                           , rounding_algorithm=self.rounding_algorithm)

            return res_layout

        def _dummy_resonator(self):
            dummy_res = i3.SRef(name="just_a_dummy", reference=self.resonator)
            return dummy_res
        # make a function for determining the distance between core size and

        def _resonator_size_core_to_core(self):
            # calls the get components function and then does math on the pulled in layout
            resonator = self._dummy_resonator()
            wg_ring_template = self.wg_ring_template

            # grabbing the position of the resonator to layout the rest of the coupler properly
            resonator_west_side = resonator.size_info().west
            resonator_east_side = resonator.size_info().east

            resonator_core_width = wg_ring_template.core_width
            resonator_clad_width = wg_ring_template.cladding_width

            resonator_x_dim = (resonator_east_side - resonator_west_side) - resonator_clad_width
            return resonator_x_dim

        # setting the output shape of the access waveguides using a shape defined by ports from MMI (hopefully..)
        def _default_wgs(self):
            # bring in parts from rest of PCell Layout, used dummy resonator to grab positions
            resonator = self._dummy_resonator()
            wg_in_cell, wg_pass_cell = self.cell.wgs
            wg_template = self.wg_coupler_template
            wg_ring_template = self.wg_ring_template

            # using the ring radius for the external radius
            external_rad = self.bend_radius_ring
            external_str = self.external_straights

            # grabbing the position of the resonator to layout the rest of the coupler properly
            resonator_west_side = resonator.size_info().west
            resonator_south_side = resonator.size_info().south

            resonator_core_width = wg_ring_template.core_width
            resonator_clad_width = wg_ring_template.cladding_width
            coupler_core_width = wg_template.core_width

            # calculate the x position for center of input coupling waveguide when coupling, and make shape
            x_coup_spot = resonator_west_side + resonator_clad_width/2. - resonator_core_width/2. - self.external_gap \
                - coupler_core_width/2.

            # get bottom using the south and cladding information again
            bottom_left = (x_coup_spot - external_str - external_rad, resonator_south_side + resonator_clad_width/2.)
            bottom_right = (x_coup_spot, resonator_south_side + resonator_clad_width/2.)
            top_right = (x_coup_spot, bottom_right[1] + 2.*external_rad + self.ring_y_straight)
            top_left = (bottom_left[0], top_right[1])

            wg_shape = [bottom_left, bottom_right, top_right, top_left]

            # now make the instance using this shape info
            wg_in_layout = wg_in_cell.get_default_view(i3.LayoutView)

            wg_in_layout.set(trace_template=wg_template, shape=wg_shape, bend_radius=external_rad,
                             rounding_algorithm=self.rounding_algorithm)

            # other waveguide for reference, can put in shape or mirror later
            wg_pass_layout = wg_pass_cell.get_default_view(i3.LayoutView)
            # wg_in_layout.set()
            return wg_in_layout, wg_pass_layout  # wg_ring_layout

        # A few functions for grabbing waveguide parameters to determine lengths for FSR checking
        # def wg_lengths(self):
        #     # grab the lengths of internal waveguides to use for calculations later
        #     wg_in_layout, wg_pass_layout, wg_ring_layout = self.wgs
        #
        #     straights_and_bends = wg_ring_layout.trace_length()
        #     return straights_and_bends

        # now we take the resonator and perform multiple translations for the CROW
        def _get_components(self):
            res_x_dim = self._resonator_size_core_to_core()
            ring_gap = self.ring_gap
            ring_core_width = self.wg_ring_template.core_width
            ring_gap_list = self.ring_gap_list

            shifting_list = [0.] + ring_gap_list
            all_components = []
            # and now crank an SRef for each Ring in a loop
            for ring in range(self.num_rings):
                # will translate the original ring over to the correct position, and iterate for number of rings
                # use an if statement for external gap list or not. Need an error
                if self.use_gap_list is False:
                    this_transform = i3.Translation(((res_x_dim + ring_gap + ring_core_width)*ring, 0.))
                    this_resonator = i3.SRef(name="R_" + str(ring), reference=self.resonator,
                                             transformation=this_transform)
                    all_components.append(this_resonator)
                else:
                    # sum previous elements of the shifting list for correct relative translation
                    total_shift = sum(shifting_list[:(ring+1)])

                    this_transform = i3.Translation(((res_x_dim + ring_core_width)*ring + total_shift, 0.))
                    this_resonator = i3.SRef(name="R_" + str(ring), reference=self.resonator,
                                             transformation=this_transform)
                    all_components.append(this_resonator)

            return all_components

        def _generate_instances(self, insts):
            # includes the get components and the waveguides
            the_rings = self._get_components()
            insts += the_rings
            wg_in_layout, wg_pass_layout = self.wgs  #  wg_pass_layout, wg_ring_layout
            insts += i3.SRef(reference=wg_in_layout, name="wg_in")

            # ok so now I grab the last ring from the rings and use it to determine its position
            last_ring = the_rings[-1]
            east_side_ring = last_ring.size_info().east

            # and I get the waveguide properties for ring and coupler, to give correct outside gap
            ring_core_width = self.wg_ring_template.core_width
            ring_clad_width = self.wg_ring_template.cladding_width

            bus_wg_core_width = self.wg_coupler_template.core_width
            bus_wg_clad_width = self.wg_coupler_template.cladding_width

            final_x_spot = (east_side_ring - ring_clad_width/2.) + ring_core_width/2. \
                           + self.external_gap + bus_wg_core_width/2.

            # rather than making a new waveguide we can mirror the previous structure into the final position
            # thus we need to determine the difference in the core position of the original structure
            # with the *negative* position of the final x position, and then the mirror will flip it around
            bus_core_pos = wg_in_layout.size_info().east - bus_wg_clad_width/2.

            # now we translate the original structure to the desired negative position, and horizontally mirror around 0
            output_transformation = i3.HMirror() + i3.Translation((-1.*(- final_x_spot - bus_core_pos), 0.))

            # finally we perform the SRef on the previous layout and transform it with a new name
            insts += i3.SRef(reference=wg_in_layout, name="wg_out", transformation=output_transformation)

            return insts

        def _generate_ports(self, prts):
            # try to reuse the output waveguides following the example and change the names, looks good
            instances = self.instances
            prts += instances["wg_in"].ports["in"].modified_copy(name="in1")
            prts += instances["wg_in"].ports["out"].modified_copy(name="in2")
            prts += instances["wg_out"].ports["in"].modified_copy(name="out1")
            prts += instances["wg_out"].ports["out"].modified_copy(name="out2")
            return prts

    class Netlist(i3.NetlistView):
        def _generate_terms(self, terms):
            terms += i3.OpticalTerm(name="in")
            # terms += i3.OpticalTerm(name="pass")
            return terms

