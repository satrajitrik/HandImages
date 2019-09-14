import cv2
import json
import os

import constants
import lbp
import sift


def main():
	image_id = input("Enter image ID: ")

	constants_dict = constants.read_json()
	read_path = constants_dict["READ_PATH"]
	write_path = constants_dict["WRITE_PATH"]

	files = os.listdir(read_path)

	file = files[files.index(image_id+".jpg")]

	img = cv2.imread(read_path + file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	lbp_vector = {
		"name": file.replace(".jpg", ""),
		"lbp": lbp.lbp(gray).tolist()
	}

	sift_vector = {
		"name": file.replace(".jpg", ""),
		"sift": sift.sift(gray).tolist()
	}

	with open(write_path+file.replace(".jpg", "")+"_lbp.json", "w") as fp:
		json.dump(lbp_vector, fp, indent=4, sort_keys=True)

	with open(write_path+file.replace(".jpg", "")+"_sift.json", "w") as fp:
		json.dump(sift_vector, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
	main()
