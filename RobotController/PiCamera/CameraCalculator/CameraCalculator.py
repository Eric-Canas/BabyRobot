
from RobotController.PiCamera.CameraCalculator.PiCameraV2Parameters import *
from math import atan, cos, radians
W = 1
H = 0

#RECTANGLE_FORMAT -> (x1,y1,x2,y2)

class CameraCalculator:
	def __init__(self, pixel_size_in_um = SENSOR_PIXEL_SIZE_IN_UM, Sensor_dim_in_px = SENSOR_DIM_IN_PIXELS,
				 focal_in_mm = FOCAL_IN_MM, element_heigth_in_cm = ELEMENT_HEIGHT_IN_CM):
		self.sensor_dimensions_in_cm = (Sensor_dim_in_px[0] * (pixel_size_in_um / 10000), Sensor_dim_in_px[1] * (pixel_size_in_um / 10000))
		self.focal_in_cm = focal_in_mm / 10
		self.element_height_in_cm = element_heigth_in_cm
		self.sensor_aperture_in_degrees = 2 * atan(self.sensor_dimensions_in_cm[0] / (2 * self.focal_in_cm))
		self.cos_of_half_aperture = cos(radians(self.sensor_aperture_in_degrees / 2.0))

	def rectangleToRealWorldXY(self,rectangle, h, w):
		y = self.getDistance(rectangle=rectangle, h=h)
		x = self.getXPos(rectangle=rectangle, w=w, distance=y)
		return (x,y)

	def getDistance(self, rectangle, h):
		(x1, y1, x2, y2) = rectangle
		percent_of_image_occuped = (max(y1,y2)-min(y1,y2)) / h
		cm_of_sensor_occuped = self.sensor_dimensions_in_cm[0] * percent_of_image_occuped
		division_of_triangles = self.focal_in_cm / cm_of_sensor_occuped
		distance_to_objective = division_of_triangles*self.element_height_in_cm
		return distance_to_objective

	def getXPos(self, rectangle, w, distance):
		(x1, y1, x2, y2) = rectangle
		#Calcs the distance ideal in img, to be X=0
		center = w / 2.0# - rectangle[Tracker.W]/2.0
		#Calcs the real distances in img
		real_distance_left = x1 + (max(x1,x2)-min(x1,x2)) / 2.0
		real_distance_right = w - real_distance_left
		#The weight of all visual camp of half image
		real_world_half_weight_at_person = distance / self.cos_of_half_aperture
		if real_distance_left<real_distance_right:
			divisor_triangle = real_distance_left/center
			distance_to_center = -real_world_half_weight_at_person * (1 - divisor_triangle)

		else:
			divisor_triangle = real_distance_right/center
			distance_to_center = real_world_half_weight_at_person * (1 - divisor_triangle)

		return distance_to_center




def map(x, in_min, in_max, out_min, out_max):
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min