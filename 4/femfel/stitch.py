from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import numpy as np

imageA = cv2.imread("images/femfel1.png")
imageB = cv2.imread("images/femfel2.png")

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

diff = cv2.absdiff(result, imageA)

kernel = np.ones((2,2), np.uint8)
img_erosion = cv2.erode(diff, kernel, iterations=1)


gray = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2GRAY)
retval, dest = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(dest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# rectList, weights = cv2.groupRectangles(contours, 0.2)

for c in contours:
  (x, y, w, h) = cv2.boundingRect(c)
  cv2.rectangle(img_erosion, (x, y), (x + w, y + h), (0, 0, 255), 1)



cv2.imshow("nonZero", dest)
cv2.imshow("gray", gray)

# show the images
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
# cv2.imshow("Keypoint Matches", vis)
# cv2.imshow("Result", result)
# cv2.imshow("diff", diff)
cv2.imshow("erosion", img_erosion)



cv2.waitKey(0)
