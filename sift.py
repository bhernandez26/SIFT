import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import sys


img = cv2.imread('./blocks_L-150x150.png')
img2 = cv2.imread('./andreas_L-150x150.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)
keypoints, descriptor = sift.detectAndCompute(gray2, None)

img = cv2.drawKeypoints(gray, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints(gray2, keypoints, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoint.jpg', img)
cv2.imwrite('sift_keypoint2.jpg', img2)
