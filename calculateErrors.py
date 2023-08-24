import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

def show_image(img):
    img = (img.astype(np.float32) / 255.0)

    cv2.imshow("Image", img)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_gt_disparity(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_array = np.array(img)

    disp = img_array.astype(np.float32) / 256.0

    return disp

def generate_data(gt_disp, pred_path):
    pred_depth = np.array(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE))
    (h,w) = gt_disp.shape

    print(np.max(gt_disp))

    mask = gt_disp > 0

    print(width_to_focal[w] * 0.54 / (gt_disp + (1.0 - mask)))

    gt_depth = width_to_focal[w] * 0.54 / (gt_disp + (3.0 - mask))
    print(np.max(gt_depth))

    return (gt_depth,pred_depth)





image_path = '/home/jimtete/data/KITTI_2015/training/disp_occ_0/000000_10.png'
pred_path = './KITTI/predictions/training/2015/000000.png'

gt_disp = load_gt_disparity(image_path)
gt_depth, pred_depth = generate_data(gt_disp, pred_path)
show_image(gt_depth)