import numpy as np 
import cv2
from padTransf import *


src = cv2.imread('test/img3.png')
dst = cv2.imread('test/img1.png')

# transf = cv2.invert(np.loadtxt('test/h1to3p'))[1]
transf = np.loadtxt('test/h1to3p')

src_warped, dst_padded = warpPerspectivePadded(src, dst, transf, cv2.WARP_INVERSE_MAP)
alpha = 0.5
beta = 1 - alpha
blended = cv2.addWeighted(src_warped, alpha, dst_padded, beta, 1.0)
cv2.imshow("Blended Warped Image", blended)
cv2.waitKey(0)

crop_warped = cv2.warpPerspective(src, transf, (dst.shape[1], dst.shape[0]), flags=cv2.WARP_INVERSE_MAP)
alpha = 0.5
beta = 1 - alpha
blended = cv2.addWeighted(crop_warped, alpha, dst, beta, 1.0)
cv2.imshow("Blended Warped Image", blended)
cv2.waitKey(0)
