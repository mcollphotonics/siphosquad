"""
Bending algorithm based on Euler Curve, linear increasing curvature over half the bend
based on generic bending algorithm code provided by Luceda (Pierre) and paper from VTT.
Added capability for arbitrary Euler bends, allowing in particular for S bends in different
shapes. Use Pierre's original code to determine the intersect point for the bend positions and fit euler from
there
authors: Blair Morrison
updated: 10th April 2016
"""
from ipkiss.geometry.shape_modifier import __ShapeModifierAutoOpenClosed__
import numpy as np
from ipkiss3 import all as i3
norm = np.linalg.norm


# define a couple of the functions externally for reference and checks
# function for determining the starting coordinates relative to the bend point
def cutting_coordinates(point1, point2, distance):
    """ get new point a distance from point 2

    calculates a new coord in x,y between two input points based on the distance from the 2nd point
    :param point1: the starting point
    :param point2: the ending point
    :param distance: the desired distance from second point
    :return: A coordinate which is the desired distance along the line
    """

    # find the angle first between the two points
    angle_p1_p2 = np.arctan2((point2[1]-point1[1]), (point2[0]-point1[0]))

    # find the distance between the two input points
    dist_p1_to_p2 = np.sqrt(np.square(point2[0]-point1[0])+np.square(point2[1]-point1[1]))

    # use the angle and trig to find delta from original point
    delta_x_point1_newpoint = np.cos(angle_p1_p2)*(dist_p1_to_p2 - distance)
    delta_y_point1_newpoint = np.sin(angle_p1_p2)*(dist_p1_to_p2 - distance)

    # add deltas and return new coordinate, turned into the correct parameters (list of 1 tuple)
    new_point = [((delta_x_point1_newpoint + point1[0]), (delta_y_point1_newpoint + point1[1]))]
    return new_point


# quick function to wrap an angle to be between pi and -pi, based on np trig functions
def wrap_angle(angle):
    the_angle = np.arctan2(np.sin(angle), np.cos(angle))
    return the_angle


def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)


class Euler90Algorithm(__ShapeModifierAutoOpenClosed__):
    angle_step = i3.PositiveNumberProperty(default=1.0, doc= "Property required by RoundedWaveguide. The value can ignored if you do not need it")
    radius = i3.PositiveNumberProperty(default=1.0, doc= "Property required by RoundedWaveguide. The value can ignored if you do not need it")

    # this check is from original code but should still be valid for the Euler Algorithm
    def validate_properties(self):
        if len(self.original_shape) != 3.0:
            raise Exception("The length of the shape is not 3:{}".format(self.original_shape))

        return True

    # This method returns the points of the new shape
    def define_points(self, pts):
        original_shape = self.original_shape
        new_points = [original_shape[0]]
        new_points.extend(self._euler_spun_90())
        new_points.extend([original_shape[2]])
        return np.array(new_points)

    # function which determines the way the curve bends from input points
    def _left_or_right(self):
        # got some weird floating errors so had to convert everything to Ints for the boolean, needs to be exact!
        # add the grabbing of original shape from the self
        original_shape = self.original_shape
        start_point = original_shape[0]
        bend_point = original_shape[1]
        end_point = original_shape[2]
        # start by finding the angle from the start point to bend point and the angle from bend point to end point
        input_angle = int(np.arctan2((bend_point[1]-start_point[1]), (bend_point[0]-start_point[0]))*180/np.pi)
        output_angle = int(np.arctan2((end_point[1]-bend_point[1]), (end_point[0]-bend_point[0]))*180/np.pi)
        # now its just a matter of subtracting the two angles and if the final angle is pi/2 its left or -pi/2 right
        final_angle = int(wrap_angle(input_angle*np.pi/180 - output_angle*np.pi/180)*180/np.pi)

        if final_angle == 90:
            direction = -1.
        elif final_angle == -90:
            direction = +1.
        else:
            raise Exception("The output angle is not +/-90 compared to input")
        return direction

    # function for calculating Euler curve heading right from origin and turning left, scaled by a radius
    @i3.cache()
    def _euler_90_algorithm(self):
        """ Calculate Euler Curve

        Generates a parametric Euler curve using appropriate functions to 5th order, with scaling
        :return: An Euler curve between the input points with appropriate curvature
        """
        # define scale factor from min radius and output angle (which is ninety degrees), grab radius from input
        output_angle = np.pi/2.
        effective_radius = self.radius
        # Euler curvature scaling factor, determined from calculating a 1. radius term and looking at output
        min_radius = effective_radius/1.87009582269
        a_scale = 2.*min_radius*(output_angle/2.0)**0.5
        # too many points causes issues on gdsii, splitting over different sizes is probably most suitable way
        if effective_radius < 30.:
            points = 50
        else:
            points = 80
        # Create t array for calculating parametric curve
        end_t = (output_angle/2.0)**0.5
        all_t = np.linspace(0., end_t, points)
        # Create a list for x values and generate the x components of parametric curve using loop
        xs = list()
        for t in all_t:
            xs.append(a_scale*(t-(1/10.)*t**5.+(1/216.)*t**9.-(1/9360.)*t**13.+(1/685440.)*t**17.))
        # Do the same for y values
        ys = list()
        for t in all_t:
            ys.append(a_scale*(t**3.*(1/3.)-(1/42.)*t**7.+(1/1320.)*t**11.-(1/75600.)*t**15.))
        # Combine the xs and ys to perform the mirroring operation
        start_euler_xy = zip(xs, ys)
        # Calculating Mirror curve for X and Y, need x axis angle and end positions
        angle_x = np.pi/2. + output_angle/2.
        end_x = start_euler_xy[-1][0]
        end_y = start_euler_xy[-1][1]
        # initialising for loops, looping using checked equations from Mathematica for mirroring around line
        x_mirror = list()
        y_mirror = list()
        for elem in start_euler_xy:
            x_mirror.append(end_x + np.cos(2*angle_x)*(elem[0] - end_x) + np.sin(2*angle_x)*(elem[1]-end_y))

        for elem in start_euler_xy:
            y_mirror.append(end_y + np.sin(2*angle_x)*(elem[0] - end_x) - np.cos(2*angle_x)*(elem[1]-end_y))

        # takes output of mirrors, flips them and combines them
        mirror_xy = zip(x_mirror[::-1], y_mirror[::-1])

        # Combines initial and mirrored list to generate the euler curve
        euler_full = start_euler_xy + mirror_xy
        return euler_full

    # takes above euler algorithm and rotates and offsets relative to the correct cutting points
    @i3.cache()
    def _euler_spun_90(self):
        """ Euler Curve rotated to direction of input angle and curving correct direction

        :param effective_radius: minimum radius in euler curve, in microns
        :param start_point: starting position of curve, defaults to origin
        :param bend_point: middle point which is base of radius
        :param end_point: ending position of curve, for setting directionality of bend
        :return: Euler curve between start_point and end_point with appropriate radius
        """
        # specifying variables from shape using .self and the cutting coordinates to fit my old definition
        effective_radius = self.radius
        start_point = cutting_coordinates(self.original_shape[0], self.original_shape[1], self.radius)[0]
        bend_point = self.original_shape[1]
        end_point = cutting_coordinates(self.original_shape[2], self.original_shape[1], self.radius)[0]
        end_cut = cutting_coordinates(self.original_shape[2], self.original_shape[1], self.radius)

        # Uses Euler curve definition with 90 degree angle for l bend and rotates for input angle
        standard_curve = self._euler_90_algorithm()

        # determine which direction the curve is going by comparing the start and end point and input angle
        direction = self._left_or_right()
        input_angle = np.arctan2((bend_point[1]-start_point[1]), (bend_point[0]-start_point[0]))

        # calculates the cutting positions around the bend point from the input radius
        first_cut = start_point
        second_cut = end_point

        # Rotate the outputs by rotation matrix and offset, flip y depending on left or right
        spun_x = []
        for elems in standard_curve:
            spun_x.append(elems[0]*np.cos(input_angle) + elems[1]*-1.*np.sin(input_angle)*direction + first_cut[0])

        spun_y = []
        for elems in standard_curve:
            spun_y.append(elems[0]*np.sin(input_angle) + elems[1]*np.cos(input_angle)*direction + first_cut[1])

        # combine the flipped(or not) y's and normal x's
        full_spun = zip(spun_x, spun_y)

        # remove final point and replace it with the end point just to avoid the radius scaling issue
        # need another definition because the structure needs to a list of 1 tuple!
        full_shape = full_spun[:-1] + end_cut
        return full_shape


class EulerArbAlgorithm(__ShapeModifierAutoOpenClosed__):
    angle_step = i3.PositiveNumberProperty(default=1.0, doc= "Property required by RoundedWaveguide. The value can ignored if you do not need it")
    radius = i3.PositiveNumberProperty(default=1.0, doc= "Property required by RoundedWaveguide. The value can ignored if you do not need it")

    # this check is from original code but should still be valid for the Euler Algorithm
    def validate_properties(self):
        if len(self.original_shape) != 3.0:
            raise Exception("The length of the shape is not 3:{}".format(self.original_shape))

        return True

    # This method returns the points of the new shape
    def define_points(self, pts):
        original_shape = self.original_shape
        new_points = [original_shape[0]] # original_shape[0]
        new_points.extend(self._euler_spun_arb())
        new_points.extend([original_shape[2]])
        return np.array(new_points)

    # Luceda code for calculating intercept points for arbitrary curve, will do the same here
    @i3.cache()
    def _get_center_circle(self):
        """
        gets the center point of the arc, used to determine intersection points
        """
        original_shape = self.original_shape
        start_point = np.array(self.original_shape[0] - self.original_shape[1])
        end_point = np.array(self.original_shape[2] - self.original_shape[1])
        base_point = (start_point/norm(start_point) + end_point/norm(end_point))/2.0
        angle = get_angle(self.original_shape[0], self.original_shape[1] , self.original_shape[2])/2.0
        sin_angle = np.sin(angle*np.pi/180.0)

        d = self.radius/sin_angle
        factor = d/np.linalg.norm(base_point)
        center = abs(factor) * base_point + self.original_shape[1]
        return center

    @i3.cache()
    def _get_intersects(self):
        """
        intersection points are used for starting and ending positions of the curve
        """
        c = self._get_center_circle()
        d = (norm(c-self.original_shape[1])**2 - self.radius**2)**0.5
        intersects = []
        for cnt in [0, 2]:
            factor = d/norm(self.original_shape[cnt]-self.original_shape[1])
            intersects.append(abs(factor) * (self.original_shape[cnt]-self.original_shape[1]) + self.original_shape[1])
        return intersects

    # function which determines the way the curve bends from input points
    @i3.cache()
    def _left_or_right_arb(self):
        # got some weird floating errors so had to convert everything to Ints for the boolean, needs to be exact!
        # add the grabbing of original shape from the self
        original_shape = self.original_shape
        start_point = original_shape[0]
        bend_point = original_shape[1]
        end_point = original_shape[2]
        # start by finding the angle from the start point to bend point and the angle from bend point to end point
        input_angle = int(np.arctan2((bend_point[1]-start_point[1]), (bend_point[0]-start_point[0]))*180/np.pi)
        output_angle = int(np.arctan2((end_point[1]-bend_point[1]), (end_point[0]-bend_point[0]))*180/np.pi)
        # now its just a matter of subtracting the two angles and if the final angle is pi/2 its left or -pi/2 right
        final_angle = int(wrap_angle(input_angle*np.pi/180 - output_angle*np.pi/180)*180/np.pi)

        if final_angle > 0:
            direction = -1.
        elif final_angle < 0:
            direction = +1.
        else:
            direction = +1.  # raise Exception("The output angle is not bent! compared to input")
        return direction

    # add a function to determine the angle between the two points and use that in euler generation
    @i3.cache()
    def bend_angle(self):
        # got some weird floating errors so had to convert everything to Ints for the boolean, needs to be exact!
        # add the grabbing of original shape from the self
        original_shape = self.original_shape
        start_point = original_shape[0]
        bend_point = original_shape[1]
        end_point = original_shape[2]
        # start by finding the angle from the start point to bend point and the angle from bend point to end point
        input_angle = (np.arctan2((bend_point[1]-start_point[1]), (bend_point[0]-start_point[0]))*180/np.pi)
        output_angle = (np.arctan2((end_point[1]-bend_point[1]), (end_point[0]-bend_point[0]))*180/np.pi)
        # now its just a matter of subtracting the two angles and if the final angle is pi/2 its left or -pi/2 right
        final_angle = abs(wrap_angle(input_angle*np.pi/180 - output_angle*np.pi/180)*180/np.pi)
        return final_angle

    def euler_arb_scaling(self):
        """
        this function determines the appropriate scaling coefficient for a given angle. Calculates the minimum scaled
        radius, which is required in the general algorithm, for fixed angles known values can be used ie 90 deg = 1.87
        Look at euler_arb_algorithm first before understanding this
        """
        # calculate the final points of the first euler curve by directly subbing into the parametric equation
        min_radius = 1.0
        output_angle_rad = self.bend_angle()*(np.pi/180.)
        output_angle = output_angle_rad
        # output_angle_circ = output_angle_rad - np.pi/2.
        a_scale = 2.*min_radius*(output_angle/2.0)**0.5
        # parametric parameter t based on euler definition, using single value for final end point
        t = (output_angle/2.)**0.5
        x_end = a_scale*(t-(1/10.)*t**5.+(1/216.)*t**9.-(1/9360.)*t**13.+(1/685440.)*t**17.)
        y_end = a_scale*(t**3.*(1/3.)-(1/42.)*t**7.+(1/1320.)*t**11.-(1/75600.)*t**15.)
        # now we rotate these as we do in the final arb rotation below
        rot_angle = np.pi/2.0 + output_angle/2.0
        x_final = x_end + np.cos(2*rot_angle)*(-x_end) + np.sin(2*rot_angle)*(-y_end)
        y_final = y_end + np.sin(2*rot_angle)*(-x_end) - np.cos(2*rot_angle)*(-y_end)
        # now we compare these positions to a circle which is centered on TRUE radius on the y axis
        # to calculate the appropriate scaling factor. There may be some issues due to infinities....
        # convert to appropriate angle and grab terms for quadratic formula
        angle_circ = output_angle - np.pi/2.0
        square_coeff = np.cos(angle_circ)**2.0 + (1.0 + np.sin(angle_circ))**2.0
        constant = -1.0*(x_final**2.0 + y_final**2.0)
        # quadratic formula with the b term as zero, taking positive root
        the_radius = ((square_coeff*constant*-1.0)**0.5)/square_coeff
        return the_radius

    # function for calculating Euler curve heading right from origin and turning left, scaled by a radius
    @i3.cache()
    def _euler_arb_algorithm(self):
        """ Calculate Euler Curve

        Generates a parametric Euler curve using appropriate functions to 5th order, with scaling
        :return: An Euler curve between the input points with appropriate curvature
        """
        # define scale factor from min radius and output angle (which is ninety degrees), grab radius from input
        out_angle_deg = self.bend_angle()
        output_angle_rad = self.bend_angle()*(np.pi/180.)
        output_angle = output_angle_rad
        effective_radius = self.radius
        euler_scale = self.euler_arb_scaling()
        # Euler curvature scaling factor, determined from calculating a 1. radius term and looking at output
        # depends on the angle, generated test code above
        min_radius = effective_radius/euler_scale
        a_scale = 2.*min_radius*(output_angle/2.0)**0.5
        # too many points causes issues on gdsii, splitting over different sizes is probably most suitable way
        if effective_radius < 30.:
            points = int(50.*(out_angle_deg/90.))
        else:
            points = int(80.*(out_angle_deg/90.))
        # Create t array for calculating parametric curve
        end_t = (output_angle/2.0)**0.5
        all_t = np.linspace(0., end_t, points)
        # Create a list for x values and generate the x components of parametric curve using loop
        xs = list()
        for t in all_t:
            xs.append(a_scale*(t-(1/10.)*t**5.+(1/216.)*t**9.-(1/9360.)*t**13.+(1/685440.)*t**17.))
        # Do the same for y values
        ys = list()
        for t in all_t:
            ys.append(a_scale*(t**3.*(1/3.)-(1/42.)*t**7.+(1/1320.)*t**11.-(1/75600.)*t**15.))
        # Combine the xs and ys to perform the mirroring operation
        start_euler_xy = zip(xs, ys)
        # Calculating Mirror curve for X and Y, need x axis angle and end positions
        angle_x = np.pi/2. + output_angle/2.
        end_x = start_euler_xy[-1][0]
        end_y = start_euler_xy[-1][1]
        # initialising for loops, looping using checked equations from Mathematica for mirroring around line
        x_mirror = list()
        y_mirror = list()
        for elem in start_euler_xy:
            x_mirror.append(end_x + np.cos(2*angle_x)*(elem[0] - end_x) + np.sin(2*angle_x)*(elem[1]-end_y))

        for elem in start_euler_xy:
            y_mirror.append(end_y + np.sin(2*angle_x)*(elem[0] - end_x) - np.cos(2*angle_x)*(elem[1]-end_y))

        # takes output of mirrors, flips them and combines them
        mirror_xy = zip(x_mirror[::-1], y_mirror[::-1])

        # Combines initial and mirrored list to generate the euler curve
        euler_full = start_euler_xy + mirror_xy
        return euler_full

    # takes above euler algorithm and rotates and offsets relative to the correct cutting points
    @i3.cache()
    def _euler_spun_arb(self):
        # """ Euler Curve rotated to direction of input angle and curving correct direction
        #
        # :param effective_radius: minimum radius in euler curve, in microns
        # :param start_point: starting position of curve, defaults to origin
        # :param bend_point: middle point which is base of radius
        # :param end_point: ending position of curve, for setting directionality of bend
        # :return: Euler curve between start_point and end_point with appropriate radius
        # """
        # specifying variables from shape using .self and the cutting coordinates to fit my old definition
        intersects = self._get_intersects()
        # Using the intersect points for scaling the curve
        effective_radius = self.radius
        start_point = intersects[0]
        bend_point = self.original_shape[1]
        end_point = intersects[1]
        end_cut = cutting_coordinates(self.original_shape[2], self.original_shape[1], self.radius)
        # Uses Euler curve definition with 90 degree angle for l bend and rotates for input angle
        standard_curve = self._euler_arb_algorithm()

        # determine which direction the curve is going by comparing the start and end point and input angle
        direction = self._left_or_right_arb()
        input_angle = np.arctan2((bend_point[1]-start_point[1]), (bend_point[0]-start_point[0]))
        # calculates the cutting positions around the bend point from the input radius
        first_cut = start_point
        second_cut = end_point

        # Rotate the outputs by rotation matrix and offset, flip y depending on left or right
        spun_x = []
        for elems in standard_curve:
            spun_x.append(elems[0]*np.cos(input_angle) + elems[1]*-1.*np.sin(input_angle)*direction + first_cut[0])

        spun_y = []
        for elems in standard_curve:
            spun_y.append(elems[0]*np.sin(input_angle) + elems[1]*np.cos(input_angle)*direction + first_cut[1])

        # combine the flipped(or not) y's and normal x's
        full_spun = zip(spun_x, spun_y)

        # remove final point and replace it with the end point just to avoid the radius scaling issue
        # need another definition because the structure needs to a list of 1 tuple!
        full_shape = full_spun
        return full_shape
