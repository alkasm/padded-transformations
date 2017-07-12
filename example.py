import numpy as np 
import cv2
from padTransf import *

# load data
src = cv2.imread('test/img3.png')
dst = cv2.imread('test/img1.png')
transf = np.loadtxt('test/h1to3p')

# run provided algorithm (using inverse map since the homography is 1 to 3 and this is warping 3 to 1)
src_warped, dst_padded = warpPerspectivePadded(src, dst, transf, flags=cv2.WARP_INVERSE_MAP)
alpha = 0.5
beta = 1 - alpha
blended = cv2.addWeighted(src_warped, alpha, dst_padded, beta, 1.0)
cv2.imshow("Blended warp, with padding", blended)
cv2.waitKey(0)

# compare to standard OpenCV function
crop_warped = cv2.warpPerspective(src, transf, (dst.shape[1], dst.shape[0]), flags=cv2.WARP_INVERSE_MAP)
alpha = 0.5
beta = 1 - alpha
blended = cv2.addWeighted(crop_warped, alpha, dst, beta, 1.0)
cv2.imshow("Blended warp, standard crop", blended)
cv2.waitKey(0)
