import task3, task4, task5, task6


def main():
    task = int(input("Input Task number:"))
    if task == 3:
        k = int(input("Enter k (k outgoing edges): "))
        K = int(input("Enter K (K most dominant images): "))
        image_ids = input("Enter 3 image file names with spaces: ")
        task3.starter(k, K, image_ids.replace(".jpg", "").split())
    elif task == 4:
        classifier = int(
            input("Select the classifier: 1. SVM\t2. Decision Tree\t3. PPR\t:")
        )
        task4.starter(classifier)
    elif task == 5:
        image_id = input("Enter the image ID: ")
        m = int(input("Enter m: "))
        k = int(input("Enter k: "))
        l = int(input("Enter l: "))
        task5.starter(image_id, m, k, l)
    elif task == 6:
        image_id = input("Enter the image ID: ")
        m = int(input("Enter m: "))
        k = int(input("Enter k: "))
        l = int(input("Enter l: "))
        algorithm = input("Enter feedback system: ")
        task6.starter(image_id, m, k, l, algorithm)


if __name__ == "__main__":
    main()
