import cv2


def sift(gray):
	orb = cv2.ORB_create()
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray, None)

	return des
