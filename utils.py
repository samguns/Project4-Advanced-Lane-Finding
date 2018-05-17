import pickle
import cv2
import numpy as np

dist_pickle = pickle.load(open("cal_params.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def abs_sobel_thresh(input, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.abs(cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.abs(cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(input, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return mag_binary


def dir_threshold(input, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.sqrt(np.power(sobelx, 2))
    abs_sobely = np.sqrt(np.power(sobely, 2))
    dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(dir)
    dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return dir_binary


def color_select(img, r_thresh=(0, 255), s_thresh=(0, 255)):
    r = img[:,:,0]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]

    r_binary = np.zeros_like(r)
    r_binary[(r > r_thresh[0]) & (r <= r_thresh[1])] = 1

    s_binary = np.zeros_like(s)
    s_binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1

    binary = np.zeros_like(r)
    binary[(r_binary == 1) & (s_binary == 1)] = 1
    return binary


def red_select(img, r_thresh=(0, 255)):
    r = img[:, :, 0]

    r_binary = np.zeros_like(r)
    r_binary[(r > r_thresh[0]) & (r <= r_thresh[1])] = 1
    return r_binary


def hls_select(img, s_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    s_binary = np.zeros_like(s)
    s_binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1
    return s_binary


def hsv_select(img, v_thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]

    v_binary = np.zeros_like(v)
    v_binary[(v > v_thresh[0]) & (v <= v_thresh[1])] = 1
    return v_binary


def warp(img, bottom_width, mid_width, height_pct):
    img_size = (img.shape[1], img.shape[0])
    bot_width = bottom_width
    mid_width = mid_width
    height_pct = height_pct
    bottom_trim = 0.935
    offset = img_size[0] * 0.25
    src = np.float32([[img.shape[1]*(0.5-mid_width/2), img.shape[0]*height_pct],
                      [img.shape[1]*(0.5+mid_width/2), img.shape[0]*height_pct],
                      [img.shape[1]*(0.5+bot_width/2), img.shape[0]*bottom_trim],
                      [img.shape[1]*(0.5-bot_width/2), img.shape[0]*bottom_trim]])
    dst = np.float32([[offset, 0],
                      [img_size[0]-offset, 0],
                      [img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv


def roi_area(img, bottom_width, mid_width, height_pct):
    img_size = (img.shape[1], img.shape[0])
    bot_width = bottom_width
    mid_width = mid_width
    height_pct = height_pct
    bottom_trim = 0.935
    offset = img_size[0] * 0.25
    src = np.float32([[img.shape[1]*(0.5-mid_width/2), img.shape[0]*height_pct],
                      [img.shape[1]*(0.5+mid_width/2), img.shape[0]*height_pct],
                      [img.shape[1]*(0.5+bot_width/2), img.shape[0]*bottom_trim],
                      [img.shape[1]*(0.5-bot_width/2), img.shape[0]*bottom_trim]])
    dst = np.float32([[offset, 0],
                      [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]]])

    print("SRC:")
    print(src)
    print("DST:")
    print(dst)

    area = np.zeros_like(img)
    cv2.fillPoly(area, np.int_([src]), (255, 0, 0))
    result = cv2.addWeighted(img, 1, area, 0.5, 0)
    return result, dst


def calculate_curvature(ploty, leftx, rightx, xm_per_pix=3.7 / 500, ym_per_pix=12 / 720):
    # Fit new polynomials to x,y in world space
    curve_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)

    y_eval = np.max(ploty)
    # Calculate the new radii of curvature
    curverad = ((1 + (2 * curve_fit_cr[0] * y_eval * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit_cr[0])

    # Caluclate the offset of the car on the road
    camera_center = (leftx[-1] + rightx[-1]) / 2
    center_diff = (camera_center - 640) * xm_per_pix

    return curverad, center_diff