from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import numpy as np

img1 = cv2.imread("images/femfel1.png")
img2 = cv2.imread("images/femfel2.png")

stitcher = Stitcher()
(result, vis) = stitcher.stitch([img1, img2], showMatches=True)

img_diff = cv2.absdiff(result, img1)

kernel = np.ones((2,2), np.uint8)
img_eroded = cv2.erode(img_diff, kernel, iterations=1)

gray = cv2.cvtColor(img_eroded, cv2.COLOR_BGR2GRAY)
retval, dest = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(dest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for c in contours:
  (x, y, w, h) = cv2.boundingRect(c)
  cv2.rectangle(img_eroded, (x, y), (x + w, y + h), (0, 0, 255), 1)


# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
feature_image = None
feature_image = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], feature_image, flags=2)

# show the images
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Feature Points", feature_image)
cv2.imshow("Difference", img_diff)
cv2.imshow("Result", img_eroded)
cv2.waitKey(0)

# Save the images
cv2.imwrite("../bilder/feature_points.png", feature_image)
cv2.imwrite("../bilder/img_diff.png", img_diff)
cv2.imwrite("../bilder/result.png", img_eroded)