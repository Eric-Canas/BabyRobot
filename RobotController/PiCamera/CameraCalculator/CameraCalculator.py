
from RobotController.PiCamera.CameraCalculator.PiCameraV2Parameters import *
from math import atan, cos, radians
W = 1
H = 0

#RECTANGLE_FORMAT -> (x1,y1,x2,y2)

class CameraCalculator:
	def __init__(self, pixel_size_in_um = SENSOR_PIXEL_SIZE_IN_UM, Sensor_dim_in_px = SENSOR_DIM_IN_PIXELS,
				 focal_in_mm = FOCAL_IN_MM, element_heigth_in_cm = ELEMENT_HEIGHT_IN_CM):
		"""
		Uses the Gauss Formula for Lens for calculating the distance to the baby in function of the parameters of the
		camera (focal length, sensor dimensions and pixel size), the height of the target and the height of its
		detected reflexion in the camera sensor.
		:param pixel_size_in_um: Float. Pixel size of the sensor in micrometers
		:param Sensor_dim_in_px: (Int, Int). Dimensions of the camera sensor.
		:param focal_in_mm: Float. Focal Lenght of the Camera in milimeters.
		:param element_heigth_in_cm: Float. Height of the element to measure in centimenters
		"""
		self.sensor_dimensions_in_cm = (Sensor_dim_in_px[0] * (pixel_size_in_um / 10000), Sensor_dim_in_px[1] * (pixel_size_in_um / 10000))
		self.focal_in_cm = focal_in_mm / 10
		self.element_height_in_cm = element_heigth_in_cm
		self.sensor_aperture_in_degrees = 2 * atan(self.sensor_dimensions_in_cm[0] / (2 * self.focal_in_cm))
		self.cos_of_half_aperture = cos(radians(self.sensor_aperture_in_degrees / 2.0))

	def rectangleToRealWorldXY(self, rectangle, h, w, element_height_in_cm = None, x_in_world_space = False):
		"""
		Calculates which is the distance to an element in the real world for the size of a given rectangle.
		:param rectangle: (Int, Int, Int, Int). Rectangle with the format (x1, y1, x2, y2).
		:param h: Int. Height of the camera sensor.
		:param w: Int. Width of the camera sensor.
		:param element_height_in_cm: Float. Height of the element to which measure distances (in the real world) in cm.
		:param x_in_world_space: Boolean. If True, return X in real world coordinates. If False, return the deviation
										  in pixels of the rectangle with respect to the center of the image.
		:return: (Float, Float or Int). Distance to the object in the real world. In format (x, y)
		"""
		if element_height_in_cm is None:
			element_height_in_cm = self.element_height_in_cm
		y = self.getDistance(rectangle=rectangle, h=h, element_height_in_cm=element_height_in_cm)
		x = self.getXPos(rectangle=rectangle, w=w, distance=(y if x_in_world_space else None))
		return (x,y)

	def getDistance(self, rectangle, h, element_height_in_cm = None):
		"""
		Calculates the distance to an element in the real world, with respect with the height of the rectangle
		of the subimage that contains it and the real height of the object in the real world.
		:param rectangle: (Int, Int, Int, Int). Rectangle with the format (x1, y1, x2, y2).
		:param h: Int. Height of the camera sensor.
		:param element_height_in_cm: Float. Height of the element to which measure distances (in the real world) in cm.
		:return: Float. Distance to the element.
		"""
		if element_height_in_cm is None:
			element_height_in_cm = self.element_height_in_cm
		(x1, y1, x2, y2) = rectangle
		percent_of_image_occuped = (max(y1,y2)-min(y1,y2)) / h
		cm_of_sensor_occuped = self.sensor_dimensions_in_cm[0] * percent_of_image_occuped
		division_of_triangles = self.focal_in_cm / cm_of_sensor_occuped
		distance_to_objective = division_of_triangles * element_height_in_cm
		return distance_to_objective

	def getXPos(self, rectangle, w, distance=None):
		"""
		Returns the X deviation to an object. If distance is not None in real world coordinates, elsewhere in pixels.
		:param rectangle: (Int, Int, Int, Int). Rectangle with the format (x1, y1, x2, y2).
		:param w: Int. Width of the camera sensor.
		:param x_in_world_space: Boolean. If True, return X in real world coordinates. If False or None,
										  return the deviation in pixels of the rectangle with respect to the
										  center of the image.
		:return: Float or Int. Distance to the real object in real world coordinates or pixels.
		"""
		(x1, y1, x2, y2) = rectangle
		#Calcs the distance ideal in img, to be X=0
		center = w / 2.0# - rectangle[Tracker.W]/2.0
		#Calcs the real distances in img
		real_distance_left = x1 + (max(x1,x2)-min(x1,x2)) / 2.0
		real_distance_right = w - real_distance_left
		if distance is not None:
			#The weight of all visual camp of half image
			real_world_half_weight_at_person = distance / self.cos_of_half_aperture
			if real_distance_left<real_distance_right:
				divisor_triangle = real_distance_left/center
				distance_to_center = -real_world_half_weight_at_person * (1 - divisor_triangle)

			else:
				divisor_triangle = real_distance_right/center
				distance_to_center = real_world_half_weight_at_person * (1 - divisor_triangle)

			return distance_to_center
		else:
			half_point = (x1 + x2) / 2
			return half_point - center




def map(x, in_min, in_max, out_min, out_max):
	"""
	Finds the equivalent interpolation of the point x in the 'in' range for the 'out' range.
	:param x: Float. Point in the 'in' range
	:param in_min: Float. Minimum of the 'in' range.
	:param in_max: Float. Maximum of the 'in' range.
	:param out_min: Float. Minimum of the 'out' range.
	:param out_max: Float. Maximum of the 'out' range.
	:return: Float. Wquivalent interpolation of the point x for the 'out' range.
	"""
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min