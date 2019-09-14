import json


def read_json():
	with open("/Users/satrajitmaitra/HandImages/constants.json") as f:
		constants = json.load(f)

	return constants
