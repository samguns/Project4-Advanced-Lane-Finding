import numpy as np
import cv2
import glob
import pickle
import matplotlib.image as mpimg

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y
objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


images = glob.glob('camera_cal/calibration*.jpg')


def collect_points(images):
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            print("corners found on " + fname)
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints


def calibrate_camera(objpoints, imgpoints):
    img = mpimg.imread('camera_cal/calibration10.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)

    # Find the warp matrix & inversion
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
    if ret == True:
        return ret, mtx, dist

    return False, None, None


if __name__ == "__main__":
    objpoints, imgpoints = collect_points(images)
    ret, mtx, dist = calibrate_camera(objpoints, imgpoints)
    # Save the camera calibration result for later use
    if ret == True:
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open('cal_params.p', 'wb'))
