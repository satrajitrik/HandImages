import os

import cv2

import Task1, Task2, Task3, Task4, Task5, Task6, Task7, Task8, ExtraCredit
from descriptor import Descriptor


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
        choice = input(
            "Do you want to go ahead with task 1 input configurations? yes(y) or no(n) "
        )
        image_id = input("Enter image ID: ")
        m = int(input("Enter m: "))
        feature_model = dimension_reduction = k = None

        if choice == "n":
            feature_model = int(
                input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
            )
            dimension_reduction = int(
                input(
                    "Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : "
                )
            )
            k = int(input("Enter k: "))

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
        choice = input(
            "Do you want to go ahead with task 3 input configurations? yes(y) or no(n) "
        )
        image_id = input("Enter image ID: ")
        m = int(input("Enter m: "))
        feature_model = dimension_reduction = k = label = None

        if choice == "n":
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

        Task4.starter(feature_model, dimension_reduction, k, label, image_id, m)

    elif task == 5:
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
        image_id = input("Enter image ID: ")
        Task5.starter(feature_model, dimension_reduction, k, label, image_id)

    elif task == 6:
        subject_id = int(input("Enter subject ID: "))
        Task6.starter(subject_id)

    elif task == 7:
        k = int(input("Enter k: "))
        Task7.starter(k)

    elif task == 8:
        k = int(input("enter k : "))
        Task8.starter(k)
    elif task == 9:
        feature_model = int(
            input("Select the Feature Model:\n1. CM\t2. LBP\t3. HOG\t4. SIFT : ")
        )
        dimension_reduction = int(
            input(
                "Select the Dimension Reduction Technique:\n1. PCA\t2. SVD\t3. NMF\t4. LDA : "
            )
        )
        k = int(input("Enter k: "))
        visualizer = int(input("Enter Visualizer:\n1. Data \t2.Feature "))
        ExtraCredit.starter(feature_model, dimension_reduction, k, visualizer)
    else:
        print("Enter Task number (1-9)")


if __name__ == "__main__":
    main()
