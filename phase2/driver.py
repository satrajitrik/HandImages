import cv2
import json
import os

from descriptor import Descriptor


def read_json():
	with open("/Users/satrajitmaitra/HandImages/constants.json") as f:
		constants = json.load(f)

	return constants

def main():
	image_id = input("Enter image ID: ")

	constants_dict = read_json()
	read_path = constants_dict["READ_PATH"]

	files = os.listdir(read_path)

	file = files[files.index("{}.jpg".format(image_id))]

	img = cv2.imread(read_path + file)
	desc = Descriptor(img)

	print(desc.sift())
	print(desc.lbp())

if __name__ == '__main__':
	main()