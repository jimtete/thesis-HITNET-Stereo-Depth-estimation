import time
import cv2
import pafy
import tensorflow as tf
import numpy as np

from hitnet import HitNet, ExportType, ModelType, draw_disparity, draw_depth, CameraConfig

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

#videoUrl = 'https://youtu.be/Yui48w71SG0'
#videoPafy = pafy.new(videoUrl)
#print(videoPafy.streams)
#cap = cv2.VideoCapture(videoPafy.getbestvideo().url)

cap = cv2.VideoCapture("first_video.mkv")
#cap.set(cv2.CAP_PROP_POS_FRAMES, 250)

# Select model type
# model_type = ModelType.middlebury
model_type = ModelType.flyingthings
# model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400.pb"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl.pb"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d.pb"

# Store baseline (m) and focal length (pixel)
camera_config = CameraConfig(0.15, 1024)
max_distance = 100

# Select export type
# export_type = ExportType.images
export_type = ExportType.video

# video initializer
videoname = str(time.time())+"_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
frame_size = (1200*2, (720-223)*2)
out = cv2.VideoWriter(videoname, fourcc, fps, frame_size)

# Initialize model
hitnet_depth = HitNet(model_path, model_type, camera_config)
i = 0
cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
while cap.isOpened():
	i = i + 1
	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue

	if (i % 12 == 0):

		# Extract the left and right images
		left_img  = frame[200:-23,:frame.shape[1]//2-80]
		right_img = frame[223:,80+frame.shape[1]//2:]
		#color_real_depth = frame[:,frame.shape[1]*2//3:]

		# Estimate the depth
		disparity_map = hitnet_depth(left_img, right_img)
		depth_map = hitnet_depth.get_depth()

		color_disparity = draw_disparity(disparity_map)
		color_depth = draw_depth(depth_map, max_distance)
		cobined_image = np.vstack((np.hstack((left_img,right_img)),
								  np.hstack((color_disparity,color_depth))))

		cv2.imshow("Estimated depth", cobined_image)

		if (export_type == export_type.video):
			out.write(cobined_image)
		else:
			filename = "0.15_1024_10_flyingthings_"+str(i).zfill(8)+".jpeg"
			cv2.imwrite(filename,cobined_image)

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

out.release()
cap.release()
cv2.destroyAllWindows()
