import numpy as np
import cv2
import glob


NX, NY = (9, 6)
CALIBRATION_IMAGES_PATH = '../camera_cal/'
MARGINX_OF_OBJECT_IMAGE,  MARGINY_OF_OBJECT_IMAGE = 136, 100

def undistort(img):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NX*NY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    # Make a list of calibration images
    path_pattern = CALIBRATION_IMAGES_PATH+'calibration*.jpg'
    print(path_pattern)
    images = glob.glob(path_pattern)
    print(images)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        # print(fname)
    
        curr_image = cv2.imread(fname)
        gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    if ret:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        # cv2.imshow("undist", dst)

        return dst
    else:
        raise("ERROR while calibration")

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


def visualize_points(points, img):
    for point in points:
        cv2.circle(img, point, radius=7, color=(0, 0, 255), thickness=-20)
    cv2.imshow("points", img)
    cv2.waitKey(0)


         


def get_calibration_points(img,marginx=MARGINX_OF_OBJECT_IMAGE,marginy=MARGINY_OF_OBJECT_IMAGE):
    img_copy = np.copy(img)
    h, w = img_copy.shape[:2]
    NX, NY = (9, 6)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    corner_points = [(marginx-19, marginy+60), (w-marginx+96, marginy+71),
                     (marginx+110, h-marginy+23), (w-marginx-65, h-marginy+15)]
    ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
    dst = np.float32(corner_points)
    src = np.float32([corners[0], corners[NX-1], corners[-NY], corners[-1]])

    visualize_points(corner_points, img_copy)

    return src, dst

image_path ="../camera_cal/calibration1.jpg"
input_image = cv2.imread(image_path)
undist = undistort(input_image)
# src, dst = get_calibration_points(undist)


# warped = warper(undist, src, dst)
cv2.imshow("undistorted version of  "+image_path.split('/')[-1], undist)
# cv2.imshow("warped", warped)
cv2.waitKey(0)
