from __future__ import print_function
import cv2
import numpy as np



__author__ = "Alexander Reynolds"
__email__ = "ar@reynoldsalexander.com"



def warpPerspectivePadded(src, dst, transf, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    """warpPerspectivePadded(...)
    warpPerspectivePadded(src, dst, transf, dsize[, flags[, borderMode[, borderValue]]]) -> src_warped, dst_padded

    This function takes in a source image to be warped onto the destination image.
    The only difference from the provided OpenCV functions is that this function 
    calculates the extent of the warped image and pads both the destination and
    the warped image so that the full extent of both images can be displayed together.

    Required arguments:
    src --- source image (to be warped)
    dst --- destination image (to be padded)
    transf --- (3, 3) transformation matrix

    Optional keyword arguments:
    flags, borderMode, borderValue --- the optional arguments to cv2.warpPerspective(). See OpenCV docs for usage.
    By default, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.

    Outputs:
    src_warped --- warped source image (uncropped)
    dst_padded --- padded destination image (same h, w as src_warped with dst placed to match src_warped)

    See also:
    warpAffinePadded() --- provides the same functionality but for affine transformations

    Contribute:
    GitHub --- https://github.com/alkasm/padded-transformations
    """

    assert transf.shape == (3, 3), 'Perspective transformation shape should be (3, 3).\nUse warpAffinePadded() for (2, 3) Euclidean or affine transformations.'

    transf = transf/transf[2,2] # ensure a legal homography
    if flags in (cv2.WARP_INVERSE_MAP, cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP, cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP):
        transf = cv2.invert(transf)[1]
        flags -= cv2.WARP_INVERSE_MAP

    # it is enough to find where the corners of the image go to find the padding bounds
    # points in clockwise order from origin
    src_h, src_w = src.shape[:2]
    lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])

    # transform points
    transf_lin_homg_pts = transf.dot(lin_homg_pts)
    transf_lin_homg_pts /= transf_lin_homg_pts[2,:]

    # find min and max points
    min_x = np.floor(np.min(transf_lin_homg_pts[0])).astype(int)
    min_y = np.floor(np.min(transf_lin_homg_pts[1])).astype(int)
    max_x = np.ceil(np.max(transf_lin_homg_pts[0])).astype(int)
    max_y = np.ceil(np.max(transf_lin_homg_pts[1])).astype(int)

    # add translation to the transformation matrix to shift to positive values
    anchor_x, anchor_y = 0, 0
    transl_transf = np.eye(3,3)
    if min_x < 0: 
        anchor_x = -min_x
        transl_transf[0,2] += anchor_x
    if min_y < 0:
        anchor_y = -min_y
        transl_transf[1,2] += anchor_y
    shifted_transf = transl_transf.dot(transf)
    shifted_transf /= shifted_transf[2,2]

    # create padded destination image
    dst_shape = dst.shape
    dst_h, dst_w = dst_shape[:2]
    if len(dst_shape) == 3: # 3-ch image, don't pad the third dimension
        pad_widths = ((anchor_y, max(max_y, dst_h)-dst_h), (anchor_x, max(max_x, dst_w)-dst_w), (0, 0))
    else:
        pad_widths = ((anchor_y, max(max_y, dst_h)-dst_h), (anchor_x, max(max_x, dst_w)-dst_w))
    dst_padded = np.pad(dst, pad_widths, mode='constant', constant_values=0) 

    dst_pad_h, dst_pad_w = dst_padded.shape[:2]
    src_warped = cv2.warpPerspective(src, shifted_transf, (dst_pad_w, dst_pad_h), 
        flags=flags, borderMode=borderMode, borderValue=borderValue)

    return dst_padded, src_warped


def warpAffinePadded(src, dst, transf, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    """warpPerspectivePadded(...)
    warpPerspectivePadded(src, dst, transf, dsize[, flags[, borderMode[, borderValue]]]) -> src_warped, dst_padded

    This function takes in a source image to be warped onto the destination image.
    The only difference from the provided OpenCV functions is that this function 
    calculates the extent of the warped image and pads both the destination and
    the warped image so that the full extent of both images can be displayed together.

    Required arguments:
    src --- source image (to be warped)
    dst --- destination image (to be padded)
    transf --- (2, 3) transformation matrix

    Optional keyword arguments:
    flags, borderMode, borderValue --- the optional arguments to cv2.warpAffine(). See OpenCV docs for usage.
    By default, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.

    Outputs:
    src_warped --- warped source image (uncropped)
    dst_padded --- padded destination image (same h, w as src_warped with dst placed to match src_warped)

    See also:
    warpPerspectivePadded() --- provides the same functionality but for perspective transformations

    Contribute:
    GitHub --- https://github.com/alkasm/padded-transformations
    """

    assert transf.shape == (2, 3), 'Affine transformation shape should be (2, 3).\nUse warpPerspectivePadded() for (3, 3) homography transformations.'

    if flags in (cv2.WARP_INVERSE_MAP, cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP, cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP):
        transf = cv2.invertAffineTransform(transf)
        flags -= cv2.WARP_INVERSE_MAP

    # it is enough to find where the corners of the image go to find the padding bounds
    # points in clockwise order from origin
    src_h, src_w = src.shape[:2]
    lin_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h]])

    # transform points
    transf_lin_pts = transf[:,:2].dot(lin_pts) + transf[:,2].reshape(2,1)

    # find min and max points
    min_x = np.floor(np.min(transf_lin_pts[0])).astype(int)
    min_y = np.floor(np.min(transf_lin_pts[1])).astype(int)
    max_x = np.ceil(np.max(transf_lin_pts[0])).astype(int)
    max_y = np.ceil(np.max(transf_lin_pts[1])).astype(int)

    # add translation to the transformation matrix to shift to positive values
    anchor_x, anchor_y = 0, 0
    if min_x < 0: 
        anchor_x = -min_x
    if min_y < 0:
        anchor_y = -min_y
    shifted_transf = transf + [[0, 0, anchor_x], [0, 0, anchor_y]]

    # create padded destination image
    dst_shape = dst.shape
    dst_h, dst_w = dst_shape[:2]
    if len(dst_shape) == 3: # 3-ch image, don't pad the third dimension
        pad_widths = ((anchor_y, max(max_y, dst_h)-dst_h), (anchor_x, max(max_x, dst_w)-dst_w), (0, 0))
    else:
        pad_widths = ((anchor_y, max(max_y, dst_h)-dst_h), (anchor_x, max(max_x, dst_w)-dst_w))
    dst_padded = np.pad(dst, pad_widths, mode='constant', constant_values=0) 

    dst_pad_h, dst_pad_w = dst_padded.shape[:2]
    src_warped = cv2.warpAffine(src, shifted_transf, (dst_pad_w, dst_pad_h), 
        flags=flags, borderMode=borderMode, borderValue=borderValue)

    return dst_padded, src_warped
