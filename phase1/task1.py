import cv2
import json
import os
import sift
import lbp


READ_PATH = "/Users/satrajitmaitra/Downloads/Hands_smaller/"
WRITE_PATH = "/Users/satrajitmaitra/HandImages/results/"

def main():
	image_id = input("Enter image ID:")
	files = os.listdir(READ_PATH)

	file = files[files.index(image_id+".jpg")]

	img = cv2.imread(READ_PATH + file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	lbp_vector = {
		"name": file.replace(".jpg", ""),
		"lbp": lbp.lbp(gray).tolist()
	}

	sift_vector = {
		"name": file.replace(".jpg", ""),
		"lbp": sift.sift(gray).tolist()
	}

	with open(WRITE_PATH+file.replace(".jpg", "")+"_lbp.json", "w") as fp:
		json.dump(lbp_vector, fp, indent=4, sort_keys=True)

	with open(WRITE_PATH+file.replace(".jpg", "")+"_sift.json", "w") as fp:
		json.dump(sift_vector, fp, indent=4, sort_keys=True)

if __name__ == '__main__':
	main()
