import cv2
import numpy as np
import os
import glob
# Define the dimensions of the checkerboard
CHECKERBOARD = (7,10) # Number of inner corners per row and column of the checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Create a vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Create a vector to store vectors of 2D points for each checkerboard image
imgpoints = []
# Define the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
# Extract paths of individual images stored in the given directory
images = glob.glob('/home/highsky/Pictures/Webcam/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the checkerboard corners
    # ret = true if the desired number of corners is found in the image
    ret, corners = cv2.findChessboardCorners(gray,
                                             CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If the desired number of corners is detected,
    # refine pixel coordinates -> display the checkerboard image
    if ret == True:
        objpoints.append(objp)
        # Refine pixel coordinates for the given 2D points
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.imshow('img',img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
h,w = img.shape[:2] # e.g., 480, 640
# Perform camera calibration by passing known 3D point values (objpoints) and their corresponding pixel coordinates of detected corners (imgpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix : \n") # Intrinsic camera matrix
print(mtx)
print("dist : \n") # Lens distortion coefficients
print(dist)
print("rvecs : \n") # Rotation vectors
print(rvecs)
print("tvecs : \n") # Translation vectors
print(tvecs)