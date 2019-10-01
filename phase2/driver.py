import json
import os

import cv2

from descriptor import Descriptor

JSON_FILE_PATH = "/Users/satrajitmaitra/HandImages/constants.json"


def read_json():
    with open(JSON_FILE_PATH) as f:
        constants = json.load(f)
    
    return constants


def main():
    task = input("Input Task number:")
    if task == 1:
        feature_model = input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
        dimension_reduction = input("Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : ")
        k = input("Enter k: ")
    elif task == 2:
        feature_model = input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
        dimension_reduction = input("Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : ")
        k = input("Enter k: ")
        image_id = raw_input("Enter image ID: ")
    elif task == 3:
        feature_model = input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
        dimension_reduction = input("Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : ")
        k = input("Enter k: ")
        label = input("Select the label: 1. 2. 3. 4 ....")
    elif task == 4:
        pass
    elif task == 5:
        pass
    elif task == 6:
        pass
    elif task == 7:
        pass
    elif task == 8:
        pass
    else:
        print "Enter Task number (1-8)"
    
    image_id = raw_input("Enter image ID: ")
    label = input("Select the label: 1. 2. 3. 4 ....")
    feature_model = input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
    dimension_reduction = input("Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : ")
    k = input("Enter k: ")
    
    constants_dict = read_json()
    read_path = constants_dict["READ_PATH"]
    
    files = os.listdir(read_path)
    image_file = files[files.index("{}.jpg".format(image_id))]
    
    img = cv2.imread(read_path + image_file)
    desc = Descriptor(img)
    
    print(desc.sift())
    print(desc.lbp())


if __name__ == '__main__':
    main()
