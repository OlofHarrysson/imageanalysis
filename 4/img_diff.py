from skimage.measure import compare_ssim
import imutils
import cv2
import sys
import scipy.io
import numpy as np
from pyimagesearch.panorama import Stitcher

img1 = cv2.imread("femfel1.png")
img2 = cv2.imread("femfel2.png")

# cv2.imshow("img1", imageA)
# cv2.imshow("img2", imageB)
# cv2.waitKey(0)



orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = None
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20], img3, flags=2)

# cv2.imshow("hej",img3)
# cv2.waitKey(0)

srcPoints = np.float32([kpsA[i] for (_, i) in matches])
dstPoints = np.float32([kpsB[i] for (i, _) in matches])

# compute the homography between the two sets of points
reprojThresh = 4.0
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)












sys.exit(1)

diff = cv2.absdiff(imageA, imageB)
kernel = np.ones((3,3), np.uint8)

img_erosion = cv2.erode(diff, kernel, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)


cv2.imshow("Diff", diff)
cv2.imshow("Canny", img_erosion)
cv2.imshow("Dilation", img_dilation)
cv2.imshow("img1", imageA)
cv2.imshow("img2", imageb)

cv2.waitKey(0)













sys.exit(1)

(score, diff) = compare_ssim(imageA, imageB, full=True, multichannel=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

cv2.imshow("Diff", diff)
cv2.waitKey(0)
sys.exit(1)

thresh = cv2.threshold(diff, 0, 255,
  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
  cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


for c in cnts:
  # compute the bounding box of the contour and then draw the
  # bounding box on both input images to represent where the two
  # images differ
  (x, y, w, h) = cv2.boundingRect(c)
  cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
  cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
