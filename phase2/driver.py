import os

import cv2

import Task1, Task2, Task3, Task4, Task5, Task6, Task7, Task8
from config import Config
from descriptor import Descriptor

JSON_FILE_PATH = "/Users/satrajitmaitra/HandImages/constants.json"


def main():
    task = int(input("Input Task number:"))

    if task == 1:
        feature_model = int(
            input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
        )
        dimension_reduction = int(
            input(
                "Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : "
            )
        )
        k = int(input("Enter k: "))
        Task1.starter(feature_model, dimension_reduction, k)

    elif task == 2:
        feature_model = int(
            input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
        )
        dimension_reduction = int(
            input(
                "Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : "
            )
        )
        k = int(input("Enter k: "))
        image_id = input("Enter image ID: ")
        m = int(input("Enter m: "))
        Task2.starter(feature_model, dimension_reduction, k, image_id, m)

    elif task == 3:
        feature_model = int(
            input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
        )
        dimension_reduction = int(
            input(
                "Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : "
            )
        )
        k = int(input("Enter k: "))
        label = int(
            input(
                "Select the label:\n1. Left\t2. Right\t3. Dorsal\t4. Palmar\n5. With accessories\t6. Without accessories\t7. Male\t8. Female: "
            )
        )
        Task3.starter(feature_model, dimension_reduction, k, label)

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
        print("Enter Task number (1-8)")


if __name__ == "__main__":
    main()
