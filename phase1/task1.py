import cv2
import json
import openCV

import os

import constants
import lbp
import sift


def starter():
	image_id, model = input("Enter image ID and model name (LBP or SIFT): ").split()

	constants_dict = constants.read_json()
	read_path = constants_dict["READ_PATH"]
	write_path = constants_dict["WRITE_PATH"]

	files = os.listdir(read_path)

	file = files[files.index("{}.jpg".format(image_id))]

	img = cv2.imread(read_path + file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	if model == "LBP":
		print(lbp.lbp(gray))
	else:
		print(sift.sift(gray))

	print("Done... ")
