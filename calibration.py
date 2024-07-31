import cv2
import numpy as np
import glob

# Define the chessboard size (inner corners)
nx = 9
ny = 6

# Prepare object points, assuming a grid of nx * ny corners
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load calibration images from folder
images = glob.glob("camera_cal/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If corners are found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Define a function to undistort images using the calibration parameters
def undistort_image(distorted_image):
    return cv2.undistort(distorted_image, mtx, dist, None, mtx)

# Example usage: undistort an image and display original vs. undistorted
img = cv2.imread("camera_cal/calibration1.jpg")
undistorted_img = undistort_image(img)

# Display original and undistorted images side by side
combined = np.hstack((img, undistorted_img))
imgresize = cv2.resize(combined, (800, 300))
cv2.imshow("Original vs Undistorted", imgresize)
cv2.waitKey(0)

src = np.float32([(550, 460),
                  (150, 720),
                  (1200, 720),
                  (770, 460)])

dst = np.float32([(100, 0),
                  (100, 720),
                  (1100, 720),
                  (1100, 0)])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

def front_to_top(img):
    size = (1280, 720)
    return cv2.warpPerspective(img, M, size, flags = cv2.INTER_LINEAR)

def top_to_front(img):
    size = (1280, 720)
    return cv2.warpPerspective(img, M_inv, size, flags = cv2.INTER_LINEAR)

image = cv2.imread("road.jpg")
output_top = front_to_top(image)
output_front = top_to_front(image)

combined = np.hstack((image, output_top, output_front))
imgresize = cv2.resize(combined, (900, 300))
cv2.imshow("Original vs Undistorted", imgresize)
cv2.waitKey(0)
cv2.destroyAllWindows()
