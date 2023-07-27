import cv2
import tensorflow as tf
import numpy as np

from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img, KittiSet

# Select model type
model_type = ModelType.middlebury
# model_type = ModelType.flyingthings
# model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400.pb"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl.pb"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d.pb"

# Select Kitti Dataset
#dataset = KittiSet.kitti_2012
dataset = KittiSet.kitti_2015
#dataset = KittiSet.atieth_2306

# Enable supervision
supervision = False

# Initialize model
hitnet_depth = HitNet(model_path, model_type)

# Load images
#left_img = load_img("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
#right_img = load_img("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

if (dataset == KittiSet.kitti_2012):
	images_count = 194
	rel_path = "/home/jimtete/data/KITTI_2012/testing/"
	save_path = "KITTI/predictions/testing/2012/"
elif (dataset == KittiSet.kitti_2015):
	images_count = 200
	rel_path = "/home/jimtete/data/KITTI_2015/testing/"
	save_path = "KITTI/predictions/testing/2015/"
elif (dataset == KittiSet.atieth_2306):
	images_count = 286
	rel_path = "./ateith_dataset/ATEITH_2306/"
	save_path = rel_path + "predictions/"

for i in range(images_count):

	#read paths
	if (dataset != KittiSet.atieth_2306):
		left_image_path = rel_path + "image_2/" + str(i).zfill(6) + "_10.png"
		right_image_path = rel_path + "image_3/" + str(i).zfill(6) + "_10.png"
	else:
		left_image_path = rel_path + "image_left_" + str(i).zfill(5) + ".png"
		right_image_path = rel_path + "image_right_" + str(i).zfill(5) + ".png"


	#load kitti eval images
	left_img = cv2.imread(left_image_path)
	right_img = cv2.imread(right_image_path)

	#generate predictions
	disparity_map = hitnet_depth(left_img, right_img)
	color_disparity = draw_disparity(disparity_map)

	if (supervision):
		cobined_image = np.hstack((left_img, right_img, color_disparity))
		cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
		cv2.imshow("Estimated disparity", cobined_image)
		cv2.waitKey(0)

	cv2.imwrite(save_path+str(i).zfill(6)+".png", color_disparity)

	cv2.destroyAllWindows()
